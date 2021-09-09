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
from omegaconf import OmegaConf
import os
from pathlib import Path
import pprint
import threading
import time
import timeit
import traceback
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, Union
import wandb
import warnings

import torch
from torch.cuda import amp
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from .core import prof, td_lambda, upgo, vtrace
from .core.buffer_utils import Buffers, create_buffers, fill_buffers_inplace, stack_buffers, split_buffers, \
    buffers_apply
from ..lux_gym import create_env
from ..lux_gym.act_spaces import ACTION_MEANINGS
from ..nns import create_model
from ..utils import flags_to_namespace


KL_DIV_LOSS = nn.KLDivLoss(reduction="none")
# TODO: Reformat logging
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def combine_policy_logits_to_log_probs(
        behavior_policy_logits: torch.Tensor,
        actions: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    """
    Combines all policy_logits at a given step to get a single action_log_probs value for that step

    Initial shape: time, batch, 1, players, x, y
    Returned shape: time, batch, players
    """
    # DEPRECATED incorrect (but effective?) way of doing things:
    """
    # Get the log probs
    log_probs = F.log_softmax(behavior_policy_logits, dim=-1)
    # Ignore log probs for actions that were not used
    log_probs = actions_taken_mask * torch.where(
        ~torch.isneginf(log_probs),
        log_probs,
        torch.zeros_like(log_probs)
    )
    # Select the log probs for the actions that were taken
    selected_log_probs = torch.gather(log_probs, -1, actions)
    # Sum over actions, y, and x dimensions to combine log_probs from different actions
    # Squeeze out action_planes dimension as well
    ## This is the problem step - we are summing log probs,
    ## which is not the same as summing probs and then taking the log
    return torch.flatten(selected_log_probs, start_dim=-3, end_dim=-1).sum(dim=-1).squeeze(dim=-2)
    """

    # Get the action probabilities
    probs = F.softmax(behavior_policy_logits, dim=-1)
    # Ignore probabilities for actions that were not used
    probs = actions_taken_mask * probs
    # Select the probabilities for the overlapping actions that were taken and sum these
    selected_probs = torch.gather(probs, -1, actions).sum(dim=-1)
    # Remove 0-valued selected_probs in order to eliminate neg-inf valued log_probs
    selected_probs = selected_probs + torch.where(
        selected_probs == 0,
        torch.ones_like(selected_probs),
        torch.zeros_like(selected_probs)
    )
    log_probs = torch.log(selected_probs)
    # Sum over y and x dimensions to combine log_probs from different actions
    # Squeeze out action_planes dimension as well
    return torch.flatten(log_probs, start_dim=-2, end_dim=-1).sum(dim=-1).squeeze(dim=-2)


def combine_policy_entropy(
        policy_logits: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes and combines policy entropy for a given step.
    NB: We are just computing the sum of individual entropies, not the joint entropy, because I don't think there is
    an efficient way to compute the joint entropy?

    Initial shape: time, batch, action_planes, players, x, y
    Returned shape: time, batch, players
    """
    policy = F.softmax(policy_logits, dim=-1)
    log_policy = F.log_softmax(policy_logits, dim=-1)
    log_policy_masked_zeroed = torch.where(
        log_policy.isneginf(),
        torch.zeros_like(log_policy),
        log_policy
    )
    entropies = (policy * log_policy_masked_zeroed).sum(dim=-1)
    assert actions_taken_mask.shape == entropies.shape
    entropies_masked = entropies * actions_taken_mask.float()
    # Sum over y, x, and action_planes dimensions to combine entropies from different actions
    return entropies_masked.sum(dim=-1).sum(dim=-1).squeeze(dim=-2)


def compute_teacher_kl_loss(
        learner_policy_logits: torch.Tensor,
        teacher_policy_logits: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    learner_policy_log_probs = F.log_softmax(learner_policy_logits, dim=-1)
    teacher_policy = F.softmax(teacher_policy_logits, dim=-1)
    kl_div = F.kl_div(
        learner_policy_log_probs,
        teacher_policy.detach(),
        reduction="none",
        log_target=False
    ).sum(dim=-1)
    assert actions_taken_mask.shape == kl_div.shape
    kl_div_masked = kl_div * actions_taken_mask.float()
    # Sum over y, x, and action_planes dimensions to combine kl divergences from different actions
    return kl_div_masked.sum(dim=-1).sum(dim=-1).squeeze(dim=-2)


def reduce(losses: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    else:
        raise ValueError(f"Reduction must be one of 'sum' or 'mean', was: {reduction}")


def compute_baseline_loss(value_targets: torch.Tensor, values: torch.Tensor, reduction: str) -> torch.Tensor:
    baseline_loss = F.smooth_l1_loss(values, value_targets.detach(), reduction="none")
    return reduce(baseline_loss, reduction=reduction)


def compute_entropy_loss(combined_learner_entropy: torch.Tensor, reduction: str) -> torch.Tensor:
    return reduce(combined_learner_entropy, reduction)


def compute_policy_gradient_loss(
        action_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        reduction: str
) -> torch.Tensor:
    cross_entropy = -action_log_probs.view_as(advantages)
    return reduce(cross_entropy * advantages.detach(), reduction)


@torch.no_grad()
def act(
    flags: SimpleNamespace,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    actor_model: torch.nn.Module,
    buffers: Buffers,
):
    if flags.debug:
        catch_me = AssertionError
    else:
        catch_me = Exception
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

                agent_output = actor_model(env_output)
                timings.time("model")

                env_output = env.step(agent_output["actions"])
                if env_output["done"].any():
                    # Cache reward, done, and info["actions_taken"] from the terminal step
                    cached_reward = env_output["reward"]
                    cached_done = env_output["done"]
                    cached_info_actions_taken = env_output["info"]["actions_taken"]
                    cached_info_logging = {
                        key: val for key, val in env_output["info"].items() if key.startswith("LOGGING_")
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
    except catch_me as e:
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
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(max(flags.batch_size // flags.n_actor_envs, 1))]
        timings.time("dequeue")
    batch = stack_buffers([buffers[m] for m in indices], dim=1)
    timings.time("batch")
    batch = buffers_apply(batch, lambda x: x.to(device=flags.learner_device, non_blocking=True))
    timings.time("device")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    return batch


def learn(
        flags: SimpleNamespace,
        actor_model: nn.Module,
        learner_model: nn.Module,
        teacher_model: Optional[nn.Module],
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        grad_scaler: amp.grad_scaler,
        lr_scheduler: torch.optim.lr_scheduler,
        total_games_played: int,
        baseline_only: bool = False,
        lock=threading.Lock(),
) -> Tuple[Dict, int]:
    """Performs a learning (optimization) step."""
    with lock:
        with amp.autocast(enabled=flags.use_mixed_precision):
            flattened_batch = buffers_apply(batch, lambda x: torch.flatten(x, start_dim=0, end_dim=1))
            learner_outputs = learner_model(flattened_batch)
            learner_outputs = buffers_apply(learner_outputs, lambda x: x.view(flags.unroll_length + 1,
                                                                              flags.batch_size,
                                                                              *x.shape[1:]))
            if flags.use_teacher:
                with torch.no_grad():
                    teacher_outputs = teacher_model(flattened_batch)
                    teacher_outputs = buffers_apply(teacher_outputs, lambda x: x.view(flags.unroll_length + 1,
                                                                                      flags.batch_size,
                                                                                      *x.shape[1:]))
            else:
                teacher_outputs = None

            # Take final value function slice for bootstrapping.
            bootstrap_value = learner_outputs["baseline"][-1]

            # Move from obs[t] -> action[t] to action[t] -> obs[t].
            batch = buffers_apply(batch, lambda x: x[1:])
            learner_outputs = buffers_apply(learner_outputs, lambda x: x[:-1])
            if flags.use_teacher:
                teacher_outputs = buffers_apply(teacher_outputs, lambda x: x[:-1])

            combined_behavior_action_log_probs = torch.zeros(
                (flags.unroll_length, flags.batch_size, 2),
                device=flags.learner_device
            )
            combined_learner_action_log_probs = torch.zeros_like(combined_behavior_action_log_probs)
            combined_teacher_kl_loss = torch.zeros_like(combined_behavior_action_log_probs)
            teacher_kl_losses = {}
            combined_learner_entropy = torch.zeros_like(combined_behavior_action_log_probs)
            entropies = {}
            for act_space in batch["actions"].keys():
                actions = batch["actions"][act_space]
                actions_taken_mask = batch["info"]["actions_taken"][act_space]

                behavior_policy_logits = batch["policy_logits"][act_space]
                behavior_action_log_probs = combine_policy_logits_to_log_probs(
                    behavior_policy_logits,
                    actions,
                    actions_taken_mask
                )
                combined_behavior_action_log_probs = combined_behavior_action_log_probs + behavior_action_log_probs

                learner_policy_logits = learner_outputs["policy_logits"][act_space]
                learner_action_log_probs = combine_policy_logits_to_log_probs(
                    learner_policy_logits,
                    actions,
                    actions_taken_mask
                )
                combined_learner_action_log_probs = combined_learner_action_log_probs + learner_action_log_probs

                # Only take entropy and KL loss for tiles where at least one action was taken
                any_actions_taken = actions_taken_mask.any(dim=-1)
                if flags.use_teacher:
                    teacher_kl_loss = compute_teacher_kl_loss(
                        learner_policy_logits,
                        teacher_outputs["policy_logits"][act_space],
                        any_actions_taken
                    )
                else:
                    teacher_kl_loss = torch.zeros_like(combined_teacher_kl_loss)
                combined_teacher_kl_loss = combined_teacher_kl_loss + teacher_kl_loss
                teacher_kl_losses[act_space] = (reduce(
                    teacher_kl_loss,
                    reduction="sum",
                ) / any_actions_taken.sum()).detach().cpu().item()

                learner_policy_entropy = combine_policy_entropy(
                    learner_policy_logits,
                    any_actions_taken
                )
                combined_learner_entropy = combined_learner_entropy + learner_policy_entropy
                entropies[act_space] = -(reduce(
                    learner_policy_entropy,
                    reduction="sum"
                ) / any_actions_taken.sum()).detach().cpu().item()

            discounts = (~batch["done"]).float() * flags.discounting
            discounts = discounts.unsqueeze(-1).expand_as(combined_behavior_action_log_probs)
            values = learner_outputs["baseline"]
            vtrace_returns = vtrace.from_action_log_probs(
                behavior_action_log_probs=combined_behavior_action_log_probs,
                target_action_log_probs=combined_learner_action_log_probs,
                discounts=discounts,
                rewards=batch["reward"],
                values=values,
                bootstrap_value=bootstrap_value
            )
            td_lambda_returns = td_lambda.td_lambda(
                rewards=batch["reward"],
                values=values,
                bootstrap_value=bootstrap_value,
                discounts=discounts,
                lmb=flags.lmb
            )
            upgo_returns = upgo.upgo(
                rewards=batch["reward"],
                values=values,
                bootstrap_value=bootstrap_value,
                discounts=discounts,
                lmb=flags.lmb
            )

            vtrace_pg_loss = compute_policy_gradient_loss(
                combined_learner_action_log_probs,
                vtrace_returns.pg_advantages,
                reduction=flags.reduction
            )
            upgo_clipped_importance = torch.minimum(
                vtrace_returns.log_rhos.exp(),
                torch.ones_like(vtrace_returns.log_rhos)
            ).detach()
            upgo_pg_loss = compute_policy_gradient_loss(
                combined_learner_action_log_probs,
                upgo_clipped_importance * upgo_returns.advantages,
                reduction=flags.reduction
            )
            baseline_loss = compute_baseline_loss(
                td_lambda_returns.vs,
                values,
                reduction=flags.reduction
            )
            teacher_kl_loss = flags.teacher_kl_cost * reduce(
                combined_teacher_kl_loss,
                reduction=flags.reduction
            )
            entropy_loss = flags.entropy_cost * reduce(
                combined_learner_entropy,
                reduction=flags.reduction
            )
            if baseline_only:
                total_loss = baseline_loss
                vtrace_pg_loss, upgo_pg_loss, teacher_kl_loss, entropy_loss = torch.zeros(4) + float("nan")
            else:
                total_loss = vtrace_pg_loss + upgo_pg_loss + baseline_loss + teacher_kl_loss + entropy_loss

            last_lr = lr_scheduler.get_last_lr()
            assert len(last_lr) == 1, 'Logging per-parameter LR still needs support'
            last_lr = last_lr[0]
            action_distributions_flat = {
                key[16:]: val[batch["done"]][~val[batch["done"]].isnan()].sum().item()
                for key, val in batch["info"].items()
                if key.startswith("LOGGING_") and "ACTIONS_" in key
            }
            action_distributions = {space: {} for space in ACTION_MEANINGS.keys()}
            for flat_name, n in action_distributions_flat.items():
                space, meaning = flat_name.split(".")
                action_distributions[space][meaning] = n
            action_distributions_aggregated = {}
            for space, dist in action_distributions.items():
                if space == "city_tile":
                    action_distributions_aggregated[space] = dist
                elif space in ("cart", "worker"):
                    aggregated = {
                        a: n for a, n in dist.items() if "TRANSFER" not in a and "MOVE" not in a
                    }
                    aggregated["TRANSFER"] = sum({a: n for a, n in dist.items() if "TRANSFER" in a}.values())
                    aggregated["MOVE"] = sum({a: n for a, n in dist.items() if "MOVE" in a}.values())
                    action_distributions_aggregated[space] = aggregated
                else:
                    raise RuntimeError(f"Unrecognized action_space: {space}")
                n_actions = sum(action_distributions_aggregated[space].values())
                if n_actions == 0:
                    action_distributions_aggregated[space] = {
                        key: float("nan") for key in action_distributions_aggregated[space].keys()
                    }
                else:
                    action_distributions_aggregated[space] = {
                        key: val / n_actions for key, val in action_distributions_aggregated[space].items()
                    }

            total_games_played += batch["done"].sum().item()
            stats = {
                "Env": {
                    key[8:]: val[batch["done"]][~val[batch["done"]].isnan()].mean().item()
                    for key, val in batch["info"].items()
                    if key.startswith("LOGGING_") and "ACTIONS_" not in key
                },
                "Actions": action_distributions_aggregated,
                "Loss": {
                    "vtrace_pg_loss": vtrace_pg_loss.detach().item(),
                    "upgo_pg_loss": upgo_pg_loss.detach().item(),
                    "baseline_loss": baseline_loss.detach().item(),
                    "teacher_kl_loss": teacher_kl_loss.detach().item(),
                    "entropy_loss": entropy_loss.detach().item(),
                    "total_loss": total_loss.detach().item(),
                },
                "Entropy": {
                    "overall": sum(e for e in entropies.values() if not math.isnan(e)),
                    **entropies
                },
                "Teacher_KL_Divergence": {
                    "overall": sum(tkld for tkld in teacher_kl_losses.values() if not math.isnan(tkld)),
                    **teacher_kl_losses
                },
                "Misc": {
                    "learning_rate": last_lr,
                    "total_games_played": total_games_played
                },
            }

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
        return stats, total_games_played


def train(flags):
    # Necessary for multithreading and multiprocessing
    os.environ["OMP_NUM_THREADS"] = "1"

    if flags.num_buffers < flags.num_actors:
        raise ValueError("num_buffers should >= num_actors")
    if flags.num_buffers < flags.batch_size // flags.n_actor_envs:
        raise ValueError("num_buffers should be larger than batch_size // n_actor_envs")

    t = flags.unroll_length
    b = flags.batch_size

    example_info = create_env(flags, torch.device("cpu")).reset(force=True)["info"]
    buffers = create_buffers(flags, example_info)

    if flags.load_dir:
        checkpoint_state = torch.load(Path(flags.load_dir) / flags.checkpoint_file, map_location=torch.device("cpu"))
    else:
        checkpoint_state = None

    actor_model = create_model(flags, flags.actor_device)
    if checkpoint_state is not None:
        actor_model.load_state_dict(checkpoint_state["model_state_dict"])
    actor_model.eval()
    actor_model.share_memory()
    n_trainable_params = sum(p.numel() for p in actor_model.parameters() if p.requires_grad)
    logging.info(f'Training model with {n_trainable_params:,d} parameters.')

    actor_processes = []
    free_queue = mp.SimpleQueue()
    full_queue = mp.SimpleQueue()

    for i in range(flags.num_actors):
        actor_start = threading.Thread if flags.debug else mp.Process
        actor = actor_start(
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
    if checkpoint_state is not None:
        learner_model.load_state_dict(checkpoint_state["model_state_dict"])
    learner_model.train()
    learner_model = learner_model.share_memory()
    if not flags.disable_wandb:
        wandb.watch(learner_model, flags.model_log_freq, log="all", log_graph=True)

    optimizer = flags.optimizer_class(
        learner_model.parameters(),
        **flags.optimizer_kwargs
    )
    if checkpoint_state is not None and not flags.weights_only:
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

    # Load teacher model for KL loss
    if flags.use_teacher:
        if flags.teacher_kl_cost <= 0.:
            raise ValueError("It does not make sense to use teacher when teacher_kl_cost <= 0.")
        teacher_flags = OmegaConf.load(Path(flags.teacher_load_dir) / "config.yaml")
        teacher_flags = flags_to_namespace(OmegaConf.to_container(teacher_flags))
        teacher_model = create_model(teacher_flags, flags.learner_device)
        teacher_model.load_state_dict(
            torch.load(
                Path(flags.teacher_load_dir) / flags.teacher_checkpoint_file,
                map_location=torch.device("cpu")
            )["model_state_dict"]
        )
        teacher_model.eval()
    else:
        teacher_model = None
        if flags.teacher_kl_cost > 0.:
            logging.warning(f"flags.teacher_kl_cost is {flags.teacher_kl_cost}, but use_teacher is False. "
                            f"Setting flags.teacher_kl_cost to 0.")
        flags.teacher_kl_cost = 0.

    def lr_lambda(epoch):
        min_pct = flags.min_lr_mod
        pct_complete = min(epoch * t * b, flags.total_steps) / flags.total_steps
        scaled_pct_complete = pct_complete * (1. - min_pct) + min_pct
        return 1. - scaled_pct_complete

    grad_scaler = amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if checkpoint_state is not None and not flags.weights_only:
        scheduler.load_state_dict(checkpoint_state["scheduler_state_dict"])

    step, total_games_played, stats = 0, 0, {}
    if checkpoint_state is not None and not flags.weights_only:
        if "step" in checkpoint_state.keys():
            step = checkpoint_state["step"]
        # Backwards compatibility
        else:
            logging.warning("Loading old checkpoint_state without 'step' saved. Starting at step 0.")
        if "total_games_played" in checkpoint_state.keys():
            total_games_played = checkpoint_state["total_games_played"]
        # Backwards compatibility
        else:
            logging.warning("Loading old checkpoint_state without 'total_games_played' saved. Starting at step 0.")

    def batch_and_learn(learner_idx, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, total_games_played, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            full_batch = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                timings,
            )
            if flags.batch_size < flags.n_actor_envs:
                batches = split_buffers(full_batch, flags.batch_size, dim=1, contiguous=True)
            else:
                batches = [full_batch]
            for batch in batches:
                stats, total_games_played = learn(
                    flags=flags,
                    actor_model=actor_model,
                    learner_model=learner_model,
                    teacher_model=teacher_model,
                    batch=batch,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    lr_scheduler=scheduler,
                    total_games_played=total_games_played,
                    baseline_only=step / (t * b) < flags.n_value_warmup_batches,
                )
                with lock:
                    step += t * b
                    if not flags.disable_wandb:
                        wandb.log(stats, step=step)
            timings.time("learn")
        if learner_idx == 0:
            logging.info(f"Batch and learn timing statistics: {timings.summary()}")

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
                "step": step,
                "total_games_played": total_games_played,
            },
            checkpoint_path + ".pt",
        )
        torch.save(
            {
                "model_state_dict": actor_model.state_dict(),
            },
            checkpoint_path + "_weights.pt"
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
                cp_path = str(step).zfill(int(math.log10(flags.total_steps)) + 1)
                checkpoint(cp_path)
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            bps = (step - start_step) / (t * b) / (timer() - start_time)
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
        cp_path = str(step).zfill(int(math.log10(flags.total_steps)) + 1)
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
