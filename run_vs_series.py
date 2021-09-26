import fire
import math
import multiprocessing as mp
import os
from pathlib import Path
from subprocess import Popen
import tqdm
from typing import NoReturn, Optional

MAP_SIZES = (12, 16, 24, 32)


def generate_game_command(
        agent_1: str,
        agent_2: str,
        game_name: str,
        map_size: int,
        out_dir: Path,
) -> str:
    replay_file = f"{out_dir / game_name}.json"
    cli_args = [
        "--loglevel 0",
        "--memory 8000",
        "--maxtime 20000",
        "storeLogs false",
        f"--width {map_size:d}",
        f"--height {map_size:d}",
        f"--out {replay_file}"
    ]
    game_commands_list = [
        "lux-ai-2021",
        agent_1,
        agent_2,
        *cli_args
    ]
    return " ".join(game_commands_list)


def run_game(game_command: str) -> NoReturn:
    proc = Popen(game_command, shell=True, stdout=open(os.devnull, 'wb'))
    proc.wait()


def main(
        agent_1: str,
        agent_2: str,
        out_dir: Optional[str] = None,
        n_workers: int = 3,
        n_games: int = 100,
        cuda_visible_devices: str = "0",
) -> NoReturn:
    agent_1 = Path(agent_1)
    agent_2 = Path(agent_2)
    agent_1_name = f"{agent_1.parent.name}/{agent_1.stem}"
    agent_2_name = f"{agent_2.parent.name}/{agent_2.stem}"
    if out_dir is None:
        out_dir = f"{agent_1_name.replace('/', '_')}__vs__{agent_2_name.replace('/', '_')}"
    out_dir = Path(out_dir)

    print(f"Saving replays to: {out_dir}")
    if out_dir.exists():
        assert out_dir.is_dir() and not list(out_dir.iterdir()), "out_dir must be an empty directory or not exist"
    else:
        out_dir.mkdir()

    print(f"Running tournament between {agent_1_name}.py and {agent_2_name}.py")
    assert n_games % len(MAP_SIZES) == 0, f"n_games must be evenly divisible by {len(MAP_SIZES)}, was {n_games}"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    n_games_per_map = n_games // len(MAP_SIZES)
    game_commands = []
    for i in range(n_games // len(MAP_SIZES)):
        if i % 2 == 0:
            a1, a2 = agent_1, agent_2
        else:
            a1, a2 = agent_2, agent_1
        for map_size in MAP_SIZES:
            game_commands.append(generate_game_command(
                agent_1=str(a1),
                agent_2=str(a2),
                game_name=f"{map_size}_{str(i).zfill(int(math.log10(n_games_per_map + 1)))}",
                map_size=map_size,
                out_dir=out_dir,
            ))

    with mp.Pool(processes=n_workers) as pool:
        _ = list(tqdm.tqdm(pool.imap(run_game, game_commands), total=len(game_commands)))


if __name__ == '__main__':
    fire.Fire(main)
