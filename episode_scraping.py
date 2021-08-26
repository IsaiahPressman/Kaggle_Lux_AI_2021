import argparse
import pandas as pd
import numpy as np
import os
import requests
import json
import datetime
import time

COMP = 'lux-ai-2021'
# You should not make more than 3600 API calls per day
MAX_CALLS_PER_DAY = 3600
LOWEST_SCORE_THRESH = 1000

META = "episode_scraping/metadata/"
MATCH_DIR = "episode_scraping/episodes/"
INFO_DIR = "episode_scraping/infos/"
base_url = "https://www.kaggle.com/requests/EpisodeService/"
get_url = base_url + "GetEpisodeReplay"
BUFFER = 1
COMPETITIONS = {
    "lux-ai-2021": 30067,
    "hungry-geese": 25401,
    "rock-paper-scissors": 22838,
    "santa-2020": 24539,
    "halite": 18011,
    "google-football": 21723
}


def create_info_json(epid: int, episodes_df: pd.DataFrame, epagents_df: pd.DataFrame):
    create_seconds = int((episodes_df[episodes_df.index == epid]["CreateTime"].values[0]).item() / 1e9)
    end_seconds = int((episodes_df[episodes_df.index == epid]["CreateTime"].values[0]).item() / 1e9)

    agents = []
    for index, row in epagents_df[epagents_df["EpisodeId"] == epid].sort_values(by=["Index"]).iterrows():
        agent = {
            "id": int(row["Id"]),
            "state": int(row["State"]),
            "submissionId": int(row["SubmissionId"]),
            "reward": int(row["Reward"]),
            "index": int(row["Index"]),
            "initialScore": float(row["InitialScore"]),
            "initialConfidence": float(row["InitialConfidence"]),
            "updatedScore": float(row["UpdatedScore"]),
            "updatedConfidence": float(row["UpdatedConfidence"]),
            "teamId": int(99999)
        }
        agents.append(agent)

    info = {
        "id": int(epid),
        "competitionId": int(COMPETITIONS[COMP]),
        "createTime": {
            "seconds": int(create_seconds)
        },
        "endTime": {
            "seconds": int(end_seconds)
        },
        "agents": agents
    }

    return info


def save_episode(epid: int, **kwargs):
    # request
    re = requests.post(get_url, json={"EpisodeId": epid})

    # save replay
    replay = re.json()["result"]["replay"]
    with open(MATCH_DIR + f"{epid:d}.json", "w") as f:
        f.write(replay)

    # save match info
    info = create_info_json(epid, **kwargs)
    with open(INFO_DIR + f"{epid:d}_info.json", "w") as f:
        json.dump(info, f)


def main(n_episodes: int):
    num_api_calls_today = 0

    # Load Episodes
    episodes_df = pd.read_csv(META + "Episodes.csv")

    # Load EpisodeAgents
    epagents_df = pd.read_csv(META + "EpisodeAgents.csv")

    print(f"Episodes.csv: {len(episodes_df)} rows before filtering.")
    print(f"EpisodeAgents.csv: {len(epagents_df)} rows before filtering.")

    episodes_df = episodes_df[episodes_df.CompetitionId == COMPETITIONS[COMP]]
    epagents_df = epagents_df[epagents_df.EpisodeId.isin(episodes_df.Id)]

    print(f"Episodes.csv: {len(episodes_df)} rows after filtering for {COMP}.")
    print(f"EpisodeAgents.csv: {len(epagents_df)} rows after filtering for {COMP}.")

    # Prepare dataframes
    episodes_df = episodes_df.set_index(["Id"])
    episodes_df["CreateTime"] = pd.to_datetime(episodes_df["CreateTime"])
    episodes_df["EndTime"] = pd.to_datetime(episodes_df["EndTime"])

    epagents_df.fillna(0, inplace=True)
    epagents_df = epagents_df.sort_values(by=["Id"], ascending=False)

    latest_scores_df = epagents_df.loc[epagents_df.groupby("SubmissionId").EpisodeId.idxmax(), :].sort_values(
        by=["UpdatedScore"])
    latest_scores_df["LatestScore"] = latest_scores_df.UpdatedScore
    latest_scores_df = latest_scores_df[["SubmissionId", "LatestScore"]]
    epagents_df = epagents_df.merge(latest_scores_df, left_on="SubmissionId", right_on="SubmissionId",
                                    how="outer").sort_values(by=["LatestScore"])

    # Get episodes with all agent scores > a given threshold
    episode_min_scores = epagents_df.groupby("EpisodeId").LatestScore.min()
    ep_to_score = episode_min_scores[episode_min_scores >= LOWEST_SCORE_THRESH].to_dict()
    print(f"{len(ep_to_score)} episodes with all agent scores over {LOWEST_SCORE_THRESH}")

    all_files = []
    for root, dirs, files in os.walk(MATCH_DIR, topdown=False):
        all_files.extend(files)
    seen_episodes = [int(f.split(".")[0]) for f in all_files
                     if "." in f and f.split(".")[0].isdigit() and f.split(".")[1] == "json"]
    remaining = np.setdiff1d([ep for ep in ep_to_score.keys()], seen_episodes)
    print(f"{len(remaining)} of these {len(ep_to_score)} episodes not yet saved")
    print(f"Total of {len(seen_episodes):d} games in existing library")

    r = BUFFER
    n_calls = min(n_episodes, MAX_CALLS_PER_DAY)
    print(f"Downloading {min(n_calls, remaining)} episodes")
    start_time = datetime.datetime.now()
    se = 0
    for epid, value in sorted(ep_to_score.items(), key=lambda kv: kv[1], reverse=True):
        if num_api_calls_today <= n_calls:
            if epid not in seen_episodes and num_api_calls_today < n_calls:
                try:
                    save_episode(int(epid), episodes_df=episodes_df, epagents_df=epagents_df)
                except requests.exceptions.ConnectionError:
                    pass
                r += 1
                se += 1
                try:
                    print(f"{num_api_calls_today + 1}: saved episode #{epid:d} with score {value:.2f}")
                    seen_episodes.append(epid)
                    num_api_calls_today += 1
                except:
                    print(f"  file {epid:d}.json did not seem to save")
                    se -= 1
                if r > (datetime.datetime.now() - start_time).seconds:
                    time.sleep(r - (datetime.datetime.now() - start_time).seconds)
            if num_api_calls_today >= (min(3600, n_calls)):
                break
    print("")
    print(f"Episodes saved: {se}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download episodes from the best agents using Kaggle's API. "
                                                 "NB: Do not download more than 3600 episodes per day, or you may be "
                                                 "banned from using their API.")
    parser.add_argument(
        "-n",
        "--n_episodes",
        type=int,
        default=3550,
        help="How many episodes to download"
    )
    args = parser.parse_args()
    main(args.n_episodes)
