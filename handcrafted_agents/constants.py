import getpass

from ..lux_ai.lux.game_constants import GAME_CONSTANTS
from ..lux_ai.lux.constants import Constants

USER = getpass.getuser()
LOCAL_EVAL = USER in ['isaiah']
DIRECTIONS = (
    Constants.DIRECTIONS.NORTH,
    Constants.DIRECTIONS.EAST,
    Constants.DIRECTIONS.SOUTH,
    Constants.DIRECTIONS.WEST
)
MAX_RESEARCH = max(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"].values())
RESOURCE_TYPES = (
    Constants.RESOURCE_TYPES.WOOD,
    Constants.RESOURCE_TYPES.COAL,
    Constants.RESOURCE_TYPES.URANIUM
)
