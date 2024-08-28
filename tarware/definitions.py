from enum import Enum, IntEnum


class AgentType(Enum):
    AGV = 0
    PICKER = 1
    AGENT = 2

class Action(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    TOGGLE_LOAD = 4

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2

class ImageLayer(Enum):
    """
    Input layers of image-style observations
    """
    SHELVES = 0 # binary layer indicating shelves (also indicates carried shelves)
    REQUESTS = 1 # binary layer indicating requested shelves
    AGENTS = 2 # binary layer indicating agents in the environment (no way to distinguish agents)
    AGENT_DIRECTION = 3 # layer indicating agent directions as int (see Direction enum + 1 for values)
    AGENT_LOAD = 4 # binary layer indicating agents with load
    GOALS = 5 # binary layer indicating goal/ delivery locations
    ACCESSIBLE = 6 # binary layer indicating accessible cells (all but occupied cells/ out of map)
    PICKERS = 7 # binary layer indicating agents in the environment which only can_load
    PICKERS_DIRECTION = 8 # layer indicating agent directions as int (see Direction enum + 1 for values)

class CollisionLayers(IntEnum):
    AGVS = 0
    PICKERS = 1
    SHELVES = 2
    CARRIED_SHELVES = 3