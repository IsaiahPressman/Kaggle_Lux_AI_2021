class Constants:

    # noinspection PyPep8Naming
    class INPUT_CONSTANTS:
        RESEARCH_POINTS = "rp"
        RESOURCES = "r"
        UNITS = "u"
        CITY = "c"
        CITY_TILES = "ct"
        ROADS = "ccd"
        DONE = "D_DONE"

    class DIRECTIONS:
        NORTH = "n"
        WEST = "w"
        SOUTH = "s"
        EAST = "e"
        CENTER = "c"

        @staticmethod
        def astuple(move_only: bool = True):
            move_directions = (
                Constants.DIRECTIONS.NORTH,
                Constants.DIRECTIONS.EAST,
                Constants.DIRECTIONS.SOUTH,
                Constants.DIRECTIONS.WEST
            )
            if move_only:
                return move_directions
            else:
                return move_directions + (Constants.DIRECTIONS.CENTER,)

    # noinspection PyPep8Naming
    class UNIT_TYPES:
        WORKER = 0
        CART = 1

    # noinspection PyPep8Naming
    class RESOURCE_TYPES:
        WOOD = "wood"
        URANIUM = "uranium"
        COAL = "coal"

        @staticmethod
        def astuple():
            return Constants.RESOURCE_TYPES.WOOD, Constants.RESOURCE_TYPES.COAL, Constants.RESOURCE_TYPES.URANIUM
