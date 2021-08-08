from contextlib import redirect_stdout
import io
# Silence "Loading environment football failed: No module named 'gfootball'" message
with redirect_stdout(io.StringIO()):
    import kaggle_environments

import hydra
import logging
import os
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from torch import multiprocessing as mp
import wandb

from lux_ai.utils import flags_to_namespace
from lux_ai.torchbeast.monobeast import train


os.environ["OMP_NUM_THREADS"] = "1"

# TODO Reformat logging basic config
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def get_default_flags(flags: DictConfig) -> DictConfig:
    flags = OmegaConf.to_container(flags)
    # Env params
    flags.setdefault("seed", 42)
    flags.setdefault("num_buffers", max(flags["num_actors"], flags["batch_size"] // flags["n_actor_envs"]))
    flags.setdefault("obs_space_kwargs", {})
    flags.setdefault("reward_space_kwargs", {})

    # Training params
    flags.setdefault("use_mixed_precision", True)
    flags.setdefault("discounting", 0.999)
    flags.setdefault("reduction", "mean")
    flags.setdefault("clip_grads", 10.)
    flags.setdefault("checkpoint_freq", 5.)
    flags.setdefault("num_learner_threads", 1)

    # Miscellaneous params
    flags.setdefault("load_dir", None)
    flags.setdefault("disable_wandb", False)

    return OmegaConf.create(flags)


@hydra.main(config_path="conf", config_name="conv_config")
def main(flags: DictConfig):
    cli_conf = OmegaConf.from_cli()
    if Path("config.yaml").exists():
        new_flags = OmegaConf.load("config.yaml")
        flags = OmegaConf.merge(new_flags, cli_conf)

    if flags.get("load_dir", None):
        # this ignores the local config.yaml and replaces it completely with saved one
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_steps=N before and want to increase it
        logging.info("Loading existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load(Path(flags.load_dir) / "config.yaml")
        flags = OmegaConf.merge(new_flags, cli_conf)

    flags = get_default_flags(flags)
    logging.info(OmegaConf.to_yaml(flags, resolve=True))
    OmegaConf.save(flags, "config.yaml")
    if not flags.disable_wandb:
        wandb.init(
            project=flags.project,
            config=vars(flags),
            group=flags.group,
            entity=flags.entity,
        )

    flags = flags_to_namespace(OmegaConf.to_container(flags))
    mp.set_sharing_strategy(flags.sharing_strategy)
    train(flags)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
