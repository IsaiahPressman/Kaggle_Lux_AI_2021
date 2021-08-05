# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from contextlib import contextmanager
import logging
import math
import os
from pathlib import Path
import pprint
import threading
import time
import timeit
import traceback
from types import SimpleNamespace
from typing import *
import wandb
import warnings

import torch
from torch.cuda import amp
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from .core import prof, vtrace
from .core.buffer_utils import Buffers, create_buffers, fill_buffers_inplace, stack_buffers, buffers_apply
from ..lux_gym import create_env
from ..nns import create_model


LOCK_TIMEOUT = 2.
# TODO: Reformat logging
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def reduce(losses: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    else:
        raise ValueError(f"Reduction must be one of 'sum' or 'mean', was: {reduction}")


def compute_baseline_loss(advantages: torch.Tensor, reduction: str) -> torch.Tensor:
    return reduce(advantages ** 2, reduction=reduction)


def compute_entropy_loss(logits: torch.Tensor, reduction: str) -> torch.Tensor:
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    log_policy_masked_zeroed = torch.where(
        log_policy.isneginf(),
        torch.zeros_like(log_policy),
        log_policy
    )
    return reduce((policy * log_policy_masked_zeroed).sum(dim=-1), reduction)


def compute_policy_gradient_loss(
        logits: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        reduction: str
) -> torch.Tensor:
    cross_entropy = F.nll_loss(
        F.log_softmax(logits, dim=-1),
        target=actions,
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return reduce(cross_entropy * advantages.detach(), reduction)


# From https://stackoverflow.com/questions/16740104/python-lock-with-statement-and-timeout
@contextmanager
def acquire_timeout(lock: threading.Lock, timeout: float):
    result = lock.acquire(timeout=timeout)
    yield result
    if result:
        lock.release()


def act(
    flags: SimpleNamespace,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    actor_model: torch.nn.Module,
    buffers: Buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()

        env = create_env(flags, device=flags.actor_device)
        if flags.seed is not None:
            env.seed(flags.seed + actor_index * flags.n_actor_envs)
        else:
            env.seed()
        env_output = env.reset(force=True)
        agent_output = actor_model(env_output)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            fill_buffers_inplace(buffers[index], dict(**env_output, **agent_output), 0)

            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output = actor_model(env_output)
                timings.time("model")

                env_output = env.step(agent_output["actions"])
                if env_output["done"].any():
                    # Cache reward, done, and info["actions_taken"] from the terminal step
                    cached_reward = env_output["reward"]
                    cached_done = env_output["done"]
                    cached_info_actions_taken = env_output["info"]["actions_taken"]
                    cached_info_logging = {
                        key: val for key, val in env_output["info"].items() if key.startswith("logging_")
                    }

                    env_output = env.reset()
                    env_output["reward"] = cached_reward
                    env_output["done"] = cached_done
                    env_output["info"]["actions_taken"] = cached_info_actions_taken
                    env_output["info"].update(cached_info_logging)
                timings.time("step")

                fill_buffers_inplace(buffers[index], dict(**env_output, **agent_output), t + 1)
                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    # For debugging:
    # except AssertionError as e:
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags: SimpleNamespace,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    timings: prof.Timings,
    lock=threading.Lock(),
):
    with acquire_timeout(lock, LOCK_TIMEOUT):
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size // flags.n_actor_envs)]
        timings.time("dequeue")
    batch = stack_buffers([buffers[m] for m in indices])
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = buffers_apply(batch, lambda x: x.to(device=flags.learner_device, non_blocking=True))
    timings.time("device")
    return batch


def learn(
    flags: SimpleNamespace,
    actor_model: nn.Module,
    learner_model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    grad_scaler: amp.grad_scaler,
    lr_scheduler: torch.optim.lr_scheduler,
    lock=threading.Lock(),
):
    """Performs a learning (optimization) step."""
    with acquire_timeout(lock, LOCK_TIMEOUT):
        with amp.autocast(enabled=flags.use_mixed_precision):
            flattened_batch = buffers_apply(batch, lambda x: torch.flatten(x, start_dim=0, end_dim=1))
            learner_outputs = learner_model(flattened_batch)
            learner_outputs = buffers_apply(learner_outputs, lambda x: x.view(flags.unroll_length + 1,
                                                                              flags.batch_size,
                                                                              *x.shape[1:]))

            # Take final value function slice for bootstrapping.
            bootstrap_value = learner_outputs["baseline"][-1]

            # Move from obs[t] -> action[t] to action[t] -> obs[t].
            batch = buffers_apply(batch, lambda x: x[1:])
            learner_outputs = buffers_apply(learner_outputs, lambda x: x[:-1])

            losses = {
                "pg": {},
                "baseline": {},
                "entropy": {}
            }
            discounts = (~batch["done"]).float() * flags.discounting
            for act_space in batch["actions"].keys():
                behavior_policy_logits = batch["policy_logits"][act_space]
                learner_policy_logits = learner_outputs["policy_logits"][act_space]
                actions = batch["actions"][act_space]
                discounts_expanded = discounts[..., None, None, None, None].expand_as(actions)
                reward = batch["reward"][..., None, :, None, None].expand_as(actions)
                values = learner_outputs["baseline"][..., None, :, None, None].expand_as(actions)
                bootstrap_value_expanded = bootstrap_value[:, None, :, None, None].expand(*actions.shape[1:])

                vtrace_returns = vtrace.from_logits(
                    behavior_policy_logits=behavior_policy_logits,
                    target_policy_logits=learner_policy_logits,
                    actions=actions,
                    discounts=discounts_expanded,
                    rewards=reward,
                    values=values,
                    bootstrap_value=bootstrap_value_expanded,
                )
                actions_taken_mask = batch["info"]["actions_taken"][act_space]
                if actions_taken_mask.sum() == 0:
                    # Mean of an empty tensor will be NaN, so we cannot compute the loss when no actions of the 
                    # given act_space were taken
                    loss = torch.zeros(1, device=flags.learner_device).sum()
                    losses["pg"][act_space] = loss
                    losses["baseline"][act_space] = loss
                    losses["entropy"][act_space] = loss
                else:
                    losses["pg"][act_space] = compute_policy_gradient_loss(
                        learner_policy_logits[actions_taken_mask],
                        actions[actions_taken_mask],
                        vtrace_returns.pg_advantages[actions_taken_mask],
                        reduction=flags.reduction
                    )
                    losses["baseline"][act_space] = flags.baseline_cost * compute_baseline_loss(
                        vtrace_returns.vs[actions_taken_mask] - values[actions_taken_mask],
                        reduction=flags.reduction
                    )
                    losses["entropy"][act_space] = flags.entropy_cost * compute_entropy_loss(
                        learner_policy_logits[actions_taken_mask],
                        reduction=flags.reduction
                    )
            total_loss = torch.stack([loss for d in losses.values() for loss in d.values()]).sum()

            last_lr = lr_scheduler.get_last_lr()
            assert len(last_lr) == 1, 'Logging per-parameter LR still needs support'
            last_lr = last_lr[0]
            losses_by_loss_type = {
                loss_type: torch.stack([v for v in val.values()]).sum().item()
                for loss_type, val in losses.items()
            }
            losses_by_act_space = {
                act_space: torch.stack([d[act_space] for d in losses.values()]).sum().item()
                for act_space in batch["actions"].keys()
            }
            stats = {
                "Env": {
                    key[8:]: val[batch["done"]].mean().item()
                    for key, val in batch["info"].items() if key.startswith("logging_")
                },
                "Loss": {
                    "total_loss": total_loss.item(),
                },
                "Misc": {
                    "learning_rate": last_lr,
                },
            }
            stats["Loss"].update({
                f"total_{key}_loss": val for key, val in losses_by_loss_type.items()
            })
            stats["Loss"].update({
                f"total_{key}_loss": val for key, val in losses_by_act_space.items()
            })
            stats["Loss"].update({
                f"{act_space}_{loss_type}_loss": val.item()
                for loss_type, d in losses.items()
                for act_space, val in d.items()
            })

            optimizer.zero_grad()
            if flags.use_mixed_precision:
                grad_scaler.scale(total_loss).backward()
                if flags.clip_grads is not None:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(learner_model.parameters(), flags.clip_grads)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                total_loss.backward()
                if flags.clip_grads is not None:
                    torch.nn.utils.clip_grad_norm_(learner_model.parameters(), flags.clip_grads)
                optimizer.step()
            if lr_scheduler is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    lr_scheduler.step()

        # noinspection PyTypeChecker
        actor_model.load_state_dict(learner_model.state_dict())
        return stats


def train(flags):
    # Necessary for multithreading and multiprocessing
    os.environ["OMP_NUM_THREADS"] = "1"

    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size // flags.n_actor_envs:
        raise ValueError("num_buffers should be larger than batch_size // n_actor_envs")

    t = flags.unroll_length
    b = flags.batch_size
    n = flags.n_actor_envs

    example_info = create_env(flags, torch.device("cpu")).reset(force=True)["info"]
    buffers = create_buffers(flags, example_info)

    if flags.load_dir:
        raise NotImplementedError

    actor_model = create_model(flags, flags.actor_device)
    actor_model.eval()
    actor_model.share_memory()

    actor_processes = []
    free_queue = mp.SimpleQueue()
    full_queue = mp.SimpleQueue()

    for i in range(flags.num_actors):
        # For debugging:
        # actor = threading.Thread(
        actor = mp.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                actor_model,
                buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)
        time.sleep(0.5)

    learner_model = create_model(flags, flags.learner_device)
    learner_model.train()
    learner_model = learner_model.share_memory()
    if not flags.disable_wandb:
        wandb.watch(learner_model, flags.model_log_freq, log="all", log_graph=True)

    optimizer = flags.optimizer_class(
        learner_model.parameters(),
        **flags.optimizer_kwargs
    )

    def lr_lambda(epoch):
        min_pct = flags.min_lr_mod
        pct_complete = min(epoch * t * b * n, flags.total_steps) / flags.total_steps
        scaled_pct_complete = pct_complete * (1. - min_pct) + min_pct
        return 1. - scaled_pct_complete

    grad_scaler = amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step, stats = 0, {}

    def batch_and_learn(learner_idx, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                timings,
            )
            stats = learn(
                flags, actor_model, learner_model, batch, optimizer, grad_scaler, scheduler
            )
            timings.time("learn")
            with lock:
                step += t * b
                if not flags.disable_wandb:
                    wandb.log(stats, step=step)
        if learner_idx == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    learner_threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name=f"batch-and-learn-{i}", args=(i,)
        )
        thread.start()
        learner_threads.append(thread)

    def checkpoint(checkpoint_path: Union[str, Path]):
        logging.info(f"Saving checkpoint to {checkpoint_path}")
        torch.save(
            {
                "model_state_dict": actor_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpoint_path,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            # Save every checkpoint_freq minutes
            if timer() - last_checkpoint_time > flags.checkpoint_freq * 60:
                cp_path = str(step).zfill(int(math.log10(flags.total_steps)) + 1) + ".pt"
                checkpoint(cp_path)
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            bps = (step - start_step) / flags.batch_size / (timer() - start_time)
            logging.info(f"Steps {step:d} @ {sps:.1f} SPS / {bps:.1f} BPS. Stats:\n{pprint.pformat(stats)}")
    except KeyboardInterrupt:
        # Try checkpointing and joining actors then quit.
        return
    else:
        for thread in learner_threads:
            thread.join()
        logging.info(f"Learning finished after {step:d} steps.")
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)
        cp_path = str(step).zfill(int(math.log10(flags.total_steps)) + 1) + ".pt"
        checkpoint(cp_path)


"""
def test(flags: SimpleNamespace, num_episodes: int = 10):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    gym_env = create_env(flags)
    env = environment.Environment(gym_env)
    model = Net(gym_env.observation_space.shape, gym_env.action_space.n, flags.use_lstm)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    observation = env.initial()
    returns = []

    while len(returns) < num_episodes:
        if flags.mode == "test_render":
            env.gym_env.render()
        agent_outputs = model(observation)
        policy_outputs, _ = agent_outputs
        observation = env.step(policy_outputs["action"])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
    )
"""
