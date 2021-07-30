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

import logging
import math
import os
from pathlib import Path
import pprint
import threading
import time
import timeit
import traceback
from typing import *
import wandb
import warnings

import torch
from torch.cuda import amp
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from .core import prof, vtrace
from .core.buffer_utils import Buffers, create_buffers, fill_buffers_inplace, stack_buffers, slice_buffers
from ..lux_gym import create_env
from ..nns import create_model


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
    logits_masked_zeroed = torch.where(
        logits.detach().isneginf(),
        torch.zeros_like(logits),
        logits
    )
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits_masked_zeroed, dim=-1)
    return reduce((policy * log_policy).sum(dim=-1), reduction)


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


def act(
    flags,
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
            env.seed(flags.seed + actor_index)
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
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    timings: prof.Timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size // flags.n_actor_envs)]
        timings.time("dequeue")
    batch = stack_buffers([buffers[m] for m in indices])
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    timings.time("device")
    return batch


def learn(
    flags,
    actor_model: nn.Module,
    learner_model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.optimizer.Optimizer,
    grad_scaler: amp.grad_scaler,
    lr_scheduler: torch.optim.lr_scheduler,
    lock=threading.Lock(),
):
    """Performs a learning (optimization) step."""
    with lock:
        with amp.autocast(enabled=flags.use_mixed_precision):
            learner_outputs = learner_model(batch)

            # Take final value function slice for bootstrapping.
            bootstrap_value = learner_outputs["baseline"][-1]

            # Move from obs[t] -> action[t] to action[t] -> obs[t].
            # batch = {key: tensor[1:] for key, tensor in batch.items()}
            # learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}
            batch = slice_buffers(batch, slice(1, None))
            learner_outputs = slice_buffers(learner_outputs, slice(None, -1))

            behavior_policy_logits = flags.act_space.from_dict(batch["policy_logits"], expanded=True)
            learner_policy_logits = flags.act_space.from_dict(learner_outputs["policy_logits"], expanded=True)
            actions = flags.act_space.from_dict(batch["actions"], expanded=False)
            discounts = (~batch["done"]).float() * flags.discounting
            discounts = discounts[..., None, None, None].expand_as(actions)
            reward = batch["reward"][..., None, None].expand_as(actions)
            values = learner_outputs["baseline"][..., None, None].expand_as(actions)
            bootstrap_value = bootstrap_value[..., None, None].expand(*actions.size()[1:])

            vtrace_returns = vtrace.from_logits(
                behavior_policy_logits=behavior_policy_logits,
                target_policy_logits=learner_policy_logits,
                actions=actions,
                discounts=discounts,
                rewards=reward,
                values=values,
                bootstrap_value=bootstrap_value,
            )
            actions_taken_mask = flags.act_space.from_dict(batch["actions_taken"], expanded=False)

            pg_loss = compute_policy_gradient_loss(
                learner_policy_logits[actions_taken_mask],
                actions[actions_taken_mask],
                vtrace_returns.pg_advantages[actions_taken_mask],
                reduction=flags.reduction
            )
            baseline_loss = flags.baseline_cost * compute_baseline_loss(
                vtrace_returns.vs[actions_taken_mask] - values[actions_taken_mask],
                reduction=flags.reduction
            )
            entropy_loss = flags.entropy_cost * compute_entropy_loss(
                learner_policy_logits[actions_taken_mask],
                reduction=flags.reduction
            )
            total_loss = pg_loss + baseline_loss + entropy_loss

            last_lr = lr_scheduler.get_last_lr()
            assert len(last_lr) == 1, 'Logging per-parameter LR still needs support'
            last_lr = last_lr[0]
            stats = {
                key[8:]: val[batch["done"]].mean.item() for key, val in batch["info"] if key.startswith("logging_")
            }
            stats.update({
                "total_loss": total_loss.item(),
                "pg_loss": pg_loss.item(),
                "baseline_loss": baseline_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "learning_rate": last_lr
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
    mp.set_start_method("spawn")

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    t = flags.unroll_length
    b = flags.batch_size
    n = flags.n_actor_envs

    _, _, _, example_info = create_env(flags, torch.device("cpu")).reset(force=True)
    buffers = create_buffers(flags, example_info)

    if flags.resume_from_checkpoint is not None:
        raise NotImplementedError

    actor_model = create_model(flags, flags.actor_device)
    actor_model.share_memory()

    actor_processes = []
    free_queue = mp.SimpleQueue()
    full_queue = mp.SimpleQueue()

    for i in range(flags.num_actors):
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

    learner_model = create_model(flags, flags.learner_device)
    learner_model = learner_model.share_memory()

    optimizer = flags.optimizer_class(
        learner_model.parameters(),
        **flags.optimizer_kwargs
    )

    def lr_lambda(epoch):
        min_pct = flags.min_lr / flags.lr
        pct_complete = min(epoch * t * b * n, flags.total_steps) / flags.total_steps
        scaled_pct_complete = pct_complete * (1. - min_pct) + min_pct
        return 1. - scaled_pct_complete

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(learner_idx, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                timings,
            )
            stats = learn(
                flags, actor_model, learner_model, batch, agent_state, optimizer, scheduler
            )
            timings.time("learn")
            with lock:
                step += t * b
                wandb.log(stats, step=stats["step"])
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
                cp_path = Path(flags.exp_folder) / str(step).zfill(int(math.log10(flags.total_steps)) + 1)
                checkpoint(cp_path)
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            logging.info(f"Steps {step:d} @ {sps:.1f} SPS. Stats:\n{pprint.pformat(stats)}")
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
        cp_path = Path(flags.exp_folder) / str(step).zfill(int(math.log10(flags.total_steps)) + 1)
        checkpoint(cp_path)


"""
def test(flags, num_episodes: int = 10):
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
