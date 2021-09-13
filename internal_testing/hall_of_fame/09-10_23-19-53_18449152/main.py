from contextlib import redirect_stdout
import io
# Silence "Loading environment football failed: No module named 'gfootball'" message
with redirect_stdout(io.StringIO()):
    import kaggle_environments
from typing import Dict
from lux_ai.rl_agent.rl_agent import agent
# from lux_ai.handcrafted_agents.needs_name_v0 import agent


if __name__ == "__main__":

    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)
    step = 0

    class Observation(Dict[str, any]):
        def __init__(self, player=0):
            self.player = player
            # self.updates = []
            # self.step = 0
    observation = Observation()
    observation["updates"] = []
    observation["step"] = 0
    observation["remainingOverageTime"] = 60.
    player_id = 0
    while True:
        inputs = read_input()
        observation["updates"].append(inputs)
        
        if step == 0:
            player_id = int(observation["updates"][0])
            observation.player = player_id
            """
            # fixes bug where updates array is shared, but the first update is agent dependent actually
            observation["updates"][0] = f"{observation.player}"
            """
        if inputs == "D_DONE":
            actions = agent(observation, None)
            observation["updates"] = []
            step += 1
            observation["step"] = step
            print(",".join(actions))
            print("D_FINISH")
