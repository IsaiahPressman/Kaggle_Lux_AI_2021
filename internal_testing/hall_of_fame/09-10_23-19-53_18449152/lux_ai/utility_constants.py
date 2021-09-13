import getpass

from .lux.constants import Constants
from .lux.game_constants import GAME_CONSTANTS

USER = getpass.getuser()
LOCAL_EVAL = USER in ['isaiah']

# Shorthand constants
DAY_LEN = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
NIGHT_LEN = GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
COLLECTION_RATES = GAME_CONSTANTS["PARAMETERS"]["WORKER_COLLECTION_RATE"]

# Derived constants
DN_CYCLE_LEN = DAY_LEN + NIGHT_LEN
MAX_CAPACITY = max(GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"].values())
MAX_RESEARCH = max(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"].values())
MAX_RESOURCE = {
    Constants.RESOURCE_TYPES.WOOD: GAME_CONSTANTS["PARAMETERS"]["MAX_WOOD_AMOUNT"],
    # https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/gen.ts#L253
    Constants.RESOURCE_TYPES.COAL: 425.,
    # https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/gen.ts#L269
    Constants.RESOURCE_TYPES.URANIUM: 350.
}
MAX_BOARD_SIZE = (32, 32)
