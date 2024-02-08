import logging

from collections import defaultdict, OrderedDict
import gym
from gym import spaces
from astar.search import AStar

from rware.utils import MultiAgentActionSpace, MultiAgentObservationSpace
from enum import Enum
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
import copy
import networkx as nx
_AXIS_Z = 0
_AXIS_Y = 1
_AXIS_X = 2

_COLLISION_LAYERS = 3

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1
_LAYER_PICKERS = 2

class _VectorWriter:
    def __init__(self, size: int):
        self.vector = np.zeros(size, dtype=np.float32)
        self.idx = 0

    def write(self, data):
        data_size = len(data)
        self.vector[self.idx : self.idx + data_size] = data
        self.idx += data_size

    def skip(self, bits):
        self.idx += bits

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

DIR_TO_ENUM = {
    (0, -1): Direction.UP,
    (0, 1): Direction.DOWN,
    (-1, 0): Direction.LEFT,
    (1, 0): Direction.RIGHT,
    }

def get_next_micro_action(agent_x, agent_y, agent_direction, target):
    target_x, target_y = target
    target_direction =  DIR_TO_ENUM[(target_x - agent_x, target_y - agent_y)]

    turn_order = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    # Find the indices of the source and target directions in the turn order
    source_index = turn_order.index(agent_direction)
    target_index = turn_order.index(target_direction)

    # Calculate the difference in indices to determine the number of turns needed
    turn_difference = (source_index - target_index) % len(turn_order)

    # Determine the direction of the best next turn
    if turn_difference == 0:
        return Action.FORWARD
    elif turn_difference == 1:
        return Action.LEFT
    elif turn_difference == 2:
        return Action.RIGHT
    elif turn_difference == 3:
        return Action.RIGHT

class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


class ObserationType(Enum):
    DICT = 0
    FLATTENED = 1
    IMAGE = 2

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

class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y


class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, msg_bits: int, agent_type: AgentType):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.message = np.zeros(msg_bits)
        self.req_action: Optional[Action] = None
        self.carrying_shelf: Optional[Shelf] = None
        self.canceled_action = None
        self.has_delivered = False
        self.path = None
        self.busy = False
        self.fixing_clash = 0
        self.type = agent_type

    @property
    def collision_layers(self):
        if self.loaded:
            return (_LAYER_AGENTS, _LAYER_SHELFS)
        else:
            return (_LAYER_AGENTS,)

    def req_location(self, grid_size) -> Tuple[int, int]:
        if self.req_action != Action.FORWARD:
            return self.x, self.y
        elif self.dir == Direction.UP:
            return self.x, max(0, self.y - 1)
        elif self.dir == Direction.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)
        elif self.dir == Direction.LEFT:
            return max(0, self.x - 1), self.y
        elif self.dir == Direction.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == Action.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == Action.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir


class Shelf(Entity):
    counter = 0

    def __init__(self, x, y):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)

    @property
    def collision_layers(self):
        return (_LAYER_SHELFS,)


class Warehouse(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        global_observations: bool,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        n_agvs: int,
        n_pickers: int,
        n_goals: int,
        msg_bits: int,
        sensor_range: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
        layout: str = None,
        observation_type: ObserationType=ObserationType.FLATTENED,
        image_observation_layers: List[ImageLayer]=[
            ImageLayer.SHELVES,
            ImageLayer.REQUESTS,
            ImageLayer.AGENTS,
            ImageLayer.GOALS,
            ImageLayer.ACCESSIBLE
        ],
        image_observation_directional: bool=True,
        normalised_coordinates: bool=False,
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where agents are rewarded if
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param n_agents: Number of spawned and controlled agents
        :type n_agents: int
        :param msg_bits: Number of communication bits for each agent
        :type msg_bits: int
        :param sensor_range: Range of each agents observation
        :type sensor_range: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param layout: A string for a custom warehouse layout. X are shelve locations, dots are corridors, and g are the goal locations. Ignores shelf_columns, shelf_height and shelf_rows when used.
        :type layout: str
        :param observation_type: Specifies type of observations
        :param image_observation_layers: Specifies types of layers observed if image-observations
            are used
        :type image_observation_layers: List[ImageLayer]
        :param image_observation_directional: Specifies whether image observations should be
            rotated to be directional (agent perspective) if image-observations are used
        :type image_observation_directional: bool
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

        self.goals: List[Tuple[int, int]] = []

        self.n_agvs = n_agvs
        self.n_pickers = n_pickers
        self.n_agents_ = n_agvs + n_pickers
        self.num_goals = n_goals

        if not layout:
            self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        else:
            self._make_layout_from_str(layout)
        if n_pickers > 0:
            self._agent_types = [AgentType.AGV for _ in range(n_agvs)] + [AgentType.PICKER for _ in range(n_pickers)]
        else:
            self._agent_types = [AgentType.AGENT for _ in range(self.n_agents_)]
        assert msg_bits == False
        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self.reward_range = (0, 1)
        self.no_need_return_item = False
        self.global_observations = global_observations
        if self.global_observations:
            self.sensor_range = 0
        self.fixing_clash_time = 4
        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps
        
        self.normalised_coordinates = normalised_coordinates

        sa_action_space = [len(self.item_loc_dict) + 1, *msg_bits * (2,)]
        if len(sa_action_space) == 1:
            sa_action_space = spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = spaces.MultiDiscrete(sa_action_space)

        self.action_space_ = spaces.Tuple(tuple(self.n_agents_ * [sa_action_space]))

        self.request_queue_size = request_queue_size
        self.request_queue = []

        self.agents: List[Agent] = []
        
        self._targets = np.zeros(len(self.item_loc_dict), dtype=int)
        self.same_action_count = np.ones(self.n_agents_)
        self._same_action_threshold = 100
        # default values:
        self.fast_obs = None
        self.image_obs = None
        self.observation_space_ = None
        if observation_type == ObserationType.IMAGE:
            self._use_image_obs(image_observation_layers, image_observation_directional)
        else:
            # used for DICT observation type and needed as preceeding stype to generate
            # FLATTENED observations as well
            self._use_slow_obs()

        # for performance reasons we
        # can flatten the obs vector
        if observation_type == ObserationType.FLATTENED:
            self._use_fast_obs()

        self.renderer = None

    @property
    def observation_space(self):
        return self.observation_space_

    @property
    def action_space(self):
        return self.action_space_
    
    @property
    def n_agents(self):
        return self.n_agents_

    def _make_layout_from_params(self, shelf_columns, shelf_rows, column_height):
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        self._extra_rows = 0
        self.grid_size = (
            (column_height + 1) * shelf_rows + 2 + self._extra_rows,
            (2 + 1) * shelf_columns + 1,
        )
        self.column_height = column_height
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        if self.num_goals > self.grid_size[1]:
            self.num_goals = self.grid_size[1]
        self.goals = [
            (self.grid_size[1] // 2 - self.num_goals//2 + i, self.grid_size[0] - 1)
            for i in range(self.num_goals)
        ]

        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        highway_func = lambda x, y: (
            (x % 3 == 0)  # vertical highways
            or (y % (self.column_height + 1) == 0)  # horizontal highways
            or (y == self.grid_size[0] - 1)  # delivery row
            or (  # remove a box for queuing
                (y > self.grid_size[0] - (self.column_height + 3 + self._extra_rows))
                and ((x == self.grid_size[1] // 2 - 1) or (x == self.grid_size[1] // 2))
            )
            or y in [self.goals[0][1] - i - 1 for i in range(self._extra_rows)]
        )
        item_loc_index = 1
        self.item_loc_dict = {}
        for y, x in self.goals:
            self.item_loc_dict[item_loc_index] = (x, y)
            item_loc_index+=1
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = highway_func(x, y)
                if not highway_func(x, y) and (x, y) not in self.goals:
                    self.item_loc_dict[item_loc_index] = (y, x)
                    item_loc_index+=1
    
    def _make_layout_from_str(self, layout):
        layout = layout.strip()
        layout = layout.replace(" ", "")
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])
        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                assert char.lower() in "gx."
                if char.lower() == "g":
                    self.goals.append((x, y))
                    self.highways[y, x] = 1
                elif char.lower() == ".":
                    self.highways[y, x] = 1

        assert len(self.goals) >= 1, "At least one goal is required"

    def _use_image_obs(self, image_observation_layers, directional=True):
        """
        Set image observation space
        :param image_observation_layers (List[ImageLayer]): list of layers to use as image channels
        :param directional (bool): flag whether observations should be directional (pointing in
            direction of agent or north-wise)
        """
        self.image_obs = True
        self.fast_obs = False
        self.image_observation_directional = directional
        self.image_observation_layers = image_observation_layers

        observation_shape = (1 + 2 * self.sensor_range, 1 + 2 * self.sensor_range)

        layers_min = []
        layers_max = []
        for layer in image_observation_layers:
            if layer == ImageLayer.AGENT_DIRECTION:
                # directions as int
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32) * max([d.value + 1 for d in Direction])
            else:
                # binary layer
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32)
            layers_min.append(layer_min)
            layers_max.append(layer_max)

        # total observation
        min_obs = np.stack(layers_min)
        max_obs = np.stack(layers_max)
        self.observation_space_ = spaces.Tuple(
            tuple([spaces.Box(min_obs, max_obs, dtype=np.float32)] * self.n_agents_)
        )

    def _use_slow_obs(self):
        self.fast_obs = False

        location_space = spaces.Box(low=0.0, high=max(self.grid_size), shape=(2,), dtype=np.float32)
        agent_id_space = spaces.Box(low=0.0, high=self.n_agents_, shape=(1,), dtype=np.float32)

        self._obs_bits_for_self = spaces.flatdim(agent_id_space) + spaces.flatdim(location_space) + 1  + spaces.flatdim(location_space)
        self._obs_bits_per_agent = spaces.flatdim(agent_id_space) + spaces.flatdim(location_space)
        self._obs_bits_per_picker = spaces.flatdim(agent_id_space) + spaces.flatdim(location_space)
        self._obs_bits_per_shelf = 1
        self._obs_bits_for_requests = 1

        if self.global_observations:
            self._obs_sensor_locations = self.grid_size[0] * self.grid_size[1]
        else:
            self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2
        self._obs_length = (
            self._obs_bits_for_self * self.n_agents_
            + self._obs_sensor_locations * (self._obs_bits_per_shelf
            + self._obs_bits_for_requests)
        )

        obs = {}
        for agent_id in range(self.n_agents_):
            obs[f"agent{agent_id+1}"] = spaces.Dict(OrderedDict(
                {
                    "agent_id": agent_id_space,
                    "carrying_shelf": spaces.MultiBinary(1),
                    "location": location_space,
                    "target_location": location_space,
                }
            ))
                
        individual_location_obs = spaces.Dict(OrderedDict(
            {
                "has_shelf": spaces.MultiBinary(1),
                "shelf_requested": spaces.MultiBinary(1),
            }
        ))
        obs["sensors"] = spaces.Tuple(self._obs_sensor_locations * (individual_location_obs,))
        self.observation_space_ = spaces.Tuple(tuple([spaces.Dict(OrderedDict(obs)) for _ in range(self.n_agents_)]))

    def _use_fast_obs(self):
        if self.fast_obs:
            return

        self.fast_obs = True
        ma_spaces = []
        for _ in self.observation_space_:
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self._obs_length,),
                    dtype=np.float32,
                )
            ]

        self.observation_space_ = spaces.Tuple(tuple(ma_spaces))

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]

    def _make_obs(self, agent):
        if self.image_obs:
            # write image observations
            if agent.id == 1:
                layers = []
                # first agent's observation --> update global observation layers
                for layer_type in self.image_observation_layers:
                    if layer_type == ImageLayer.SHELVES:
                        layer = self.grid[_LAYER_SHELFS].copy().astype(np.float32)
                        # set all occupied shelf cells to 1.0 (instead of shelf ID)
                        layer[layer > 0.0] = 1.0
                        # print("SHELVES LAYER")
                    elif layer_type == ImageLayer.REQUESTS:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for requested_shelf in self.request_queue:
                            layer[requested_shelf.y, requested_shelf.x] = 1.0
                        # print("REQUESTS LAYER")
                    elif layer_type == ImageLayer.AGENTS:
                        layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                        # set all occupied agent cells to 1.0 (instead of agent ID)
                        layer[layer > 0.0] = 1.0
                        # print("AGENTS LAYER")
                    elif layer_type == ImageLayer.AGENT_DIRECTION:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            if ag.can_carry:
                                agent_direction = ag.dir.value + 1
                                layer[ag.x, ag.y] = float(agent_direction)
                    elif layer_type == ImageLayer.PICKERS:
                        layer = self.grid[_LAYER_PICKERS].copy().astype(np.float32)
                        # set all occupied agent cells to 1.0 (instead of agent ID)
                        layer[layer > 0.0] = 1.0
                        # print("AGENTS LAYER")
                    elif layer_type == ImageLayer.PICKERS_DIRECTION:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            if ag.can_load and not ag.can_carry:
                                agent_direction = ag.dir.value + 1
                                layer[ag.x, ag.y] = float(agent_direction)
                        # print("AGENT DIRECTIONS LAYER")
                    elif layer_type == ImageLayer.AGENT_LOAD:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            if ag.carrying_shelf is not None:
                                layer[ag.x, ag.y] = 1.0
                        # print("AGENT LOAD LAYER")
                    elif layer_type == ImageLayer.GOALS:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for goal_y, goal_x in self.goals:
                            layer[goal_x, goal_y] = 1.0
                        # print("GOALS LAYER")
                    elif layer_type == ImageLayer.ACCESSIBLE:
                        layer = np.ones(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            layer[ag.y, ag.x] = 0.0
                        # print("ACCESSIBLE LAYER")
                    # print(layer)
                    # print()
                    # pad with 0s for out-of-map cells
                    layer = np.pad(layer, self.sensor_range, mode="constant")
                    layers.append(layer)
                self.global_layers = np.stack(layers)

            # global information was generated --> get information for agent
            start_x = agent.y
            end_x = agent.y + 2 * self.sensor_range + 1
            start_y = agent.x
            end_y = agent.x + 2 * self.sensor_range + 1
            obs = self.global_layers[:, start_x:end_x, start_y:end_y]

            if self.image_observation_directional:
                # rotate image to be in direction of agent
                if agent.dir == Direction.DOWN:
                    # rotate by 180 degrees (clockwise)
                    obs = np.rot90(obs, k=2, axes=(1,2))
                elif agent.dir == Direction.LEFT:
                    # rotate by 90 degrees (clockwise)
                    obs = np.rot90(obs, k=3, axes=(1,2))
                elif agent.dir == Direction.RIGHT:
                    # rotate by 270 degrees (clockwise)
                    obs = np.rot90(obs, k=1, axes=(1,2))
                # no rotation needed for UP direction
            return obs
        if self.global_observations:
            min_x = 0
            max_x = self.grid_size[1]
            min_y = 0
            max_y = self.grid_size[0]
        else:
            min_x = agent.x - self.sensor_range
            max_x = agent.x + self.sensor_range + 1
            min_y = agent.y - self.sensor_range
            max_y = agent.y + self.sensor_range + 1

        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > self.grid_size[1])
            or (max_y > self.grid_size[0])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_pickers = np.pad(
                self.grid[_LAYER_PICKERS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_SHELFS], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELFS]
            padded_pickers = self.grid[_LAYER_PICKERS]

        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)
        pickers = padded_pickers[min_y:max_y, min_x:max_x].reshape(-1)
        type_mapping = {type_:i for i, type_ in enumerate(set(self._agent_types))}
        if self.fast_obs:
            # write flattened observations
            obs = _VectorWriter(self.observation_space_[agent.id - 1].shape[0])

            if self.normalised_coordinates:
                agent_x = agent.x / (self.grid_size[1] - 1)
                agent_y = agent.y / (self.grid_size[0] - 1)
            else:
                agent_x = agent.x
                agent_y = agent.y
            obs.write([agent.id, int(agent.carrying_shelf is not None), agent_x, agent_y ])
            if self._targets[agent.id - 1] != 0:
                obs.write(self.item_loc_dict[self._targets[agent.id - 1]])
            else:
                obs.write(np.zeros(2))

            for i in range(self.n_agents_):
                if i != agent.id - 1:
                    agent_ = self.agents[i]
                    obs.write([agent_.id, int(agent_.carrying_shelf is not None), agent_.x, agent_.y ])
                    if self._targets[agent_.id - 1] != 0:
                        obs.write(self.item_loc_dict[self._targets[agent_.id - 1]])
                    else:
                        obs.write(np.zeros(2))

            for i, id_shelf in enumerate(shelfs):
                if id_shelf == 0:
                    obs.skip(2)
                else:
                    obs.write(
                        [1.0, int(self.shelfs[id_shelf - 1] in self.request_queue)]
                    )
            return obs.vector
 
        # write dictionary observations
        obs = {}
        if self.normalised_coordinates:
            agent_x = agent.x / (self.grid_size[1] - 1)
            agent_y = agent.y / (self.grid_size[0] - 1)
        else:
            agent_x = agent.x
            agent_y = agent.y
        # --- self data
        obs["self"] = {
            "location": np.array([agent_x, agent_y]),
            "carrying_shelf": [int(agent.carrying_shelf is not None)],
            "direction": agent.dir.value,
            "on_highway": [int(self._is_highway(agent.x, agent.y))],
        }
        # --- sensor data
        obs["sensors"] = tuple({} for _ in range(self._obs_sensor_locations))

        # find neighboring agents
        for i, id_ in enumerate(agents):
            if id_ == 0:
                obs["sensors"][i]["has_agent"] = [0]
                obs["sensors"][i]["direction"] = 0
                obs["sensors"][i]["local_message"] = self.msg_bits * [0]
            else:
                obs["sensors"][i]["has_agent"] = [1]
                obs["sensors"][i]["direction"] = self.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message"] = self.agents[id_ - 1].message
        # find neighboring pickers
        for i, id_ in enumerate(pickers):
            if id_ == 0:
                obs["sensors"][i]["has_picker"] = [0]
                obs["sensors"][i]["direction_picker"] = 0
                obs["sensors"][i]["local_message_picker"] = self.msg_bits * [0]
            else:
                obs["sensors"][i]["has_picker"] = [1]
                obs["sensors"][i]["direction_picker"] = self.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message_picker"] = self.agents[id_ - 1].message
        # find neighboring shelfs:
        for i, id_ in enumerate(shelfs):
            if id_ == 0:
                obs["sensors"][i]["has_shelf"] = [0]
                obs["sensors"][i]["shelf_requested"] = [0]
            else:
                obs["sensors"][i]["has_shelf"] = [1]
                obs["sensors"][i]["shelf_requested"] = [
                    int(self.shelfs[id_ - 1] in self.request_queue)
                ]

        return obs
    
    def find_path(self, start, goal, agent):
        grid = copy.deepcopy(self.grid[_LAYER_AGENTS])
        # for goal_ in self.goals:
        #     if not (goal[0]==goal_[1] and goal[1]==goal_[0]):
        #         grid[goal_[1], goal_[0]] = 1
        
        #grid[goal[0], goal[1]] = 0
        
        if agent.carrying_shelf:
            grid += self.grid[_LAYER_SHELFS]
        # Pickers can move everywhere withough collisions
        if agent.type == AgentType.PICKER:
            grid[grid>0] = 0
        # Goal location is available even if currently occupied
        grid = [list(map(int, l)) for l in (grid!=0)]
        astar_path = AStar(grid).search(start, goal)
        if astar_path:
            return [(x, y) for y, x in astar_path[1:]]
        else:
            return []

    def _recalc_grid(self):
        self.grid[:] = 0
        for s in self.shelfs:
            self.grid[_LAYER_SHELFS, s.y, s.x] = s.id

            for agent in self.agents:
                if agent.type == AgentType.PICKER:
                    self.grid[_LAYER_PICKERS, agent.y, agent.x] = agent.id
                else:
                    self.grid[_LAYER_AGENTS, agent.y, agent.x] = agent.id
    
    def get_shelf_request_information(self, print_=True):
        request_item_map = np.zeros(len(self.item_loc_dict) - len(self.goals))
        requested_shelf_coords = [(shelf.y, shelf.x) for shelf in self.request_queue]
        for id_, coords in self.item_loc_dict.items():
            if (coords[1], coords[0]) not in self.goals:
                if self.grid[_LAYER_SHELFS, coords[0], coords[1]]!=0:
                    request_item_map[id_ - len(self.goals) - 1] = int((coords[0], coords[1]) in requested_shelf_coords)
        return request_item_map
    
    def get_empty_shelf_informatlocationsion(self):
        empty_item_map = np.zeros(len(self.item_loc_dict) - len(self.goals))
        for id_, coords in self.item_loc_dict.items():
            if (coords[1], coords[0]) not in self.goals:
                if self.grid[_LAYER_SHELFS, coords[0], coords[1]] == 0:
                    empty_item_map[id_ - len(self.goals) - 1] = 1
        return empty_item_map
    
    def get_shelf_dispatch_information(self):
        dispatch_item_map = np.zeros(len(self.item_loc_dict) - len(self.goals))
        for id_, coords in self.item_loc_dict.items():
            if (coords[1], coords[0]) not in self.goals:
                if self.grid[_LAYER_SHELFS, coords[0], coords[1]]!=0 and self.grid[_LAYER_AGENTS, coords[0], coords[1]]!=0:
                    if self.agents[self.grid[_LAYER_AGENTS, coords[0], coords[1]] - 1].req_action == Action.TOGGLE_LOAD:
                        dispatch_item_map[id_ - len(self.goals) - 1] = 1
        return dispatch_item_map
    
    def reset(self):
        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # n_xshelf = (self.grid_size[1] - 1) // 3
        # n_yshelf = (self.grid_size[0] - 2) // 9

        # make the shelfs
        self.shelfs = [
            Shelf(x, y)
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(x, y)
        ]

        # spawn agents at random locations
        agent_locs = np.random.choice(
            np.arange(self.grid_size[0] * self.grid_size[1]),
            size=self.n_agents_,
            replace=False,
        )
        agent_locs = np.unravel_index(agent_locs, self.grid_size)
        # and direction
        agent_dirs = np.random.choice([d for d in Direction], size=self.n_agents_)
        self.agents = [
            Agent(x, y, dir_, self.msg_bits, agent_type = agent_type)
            for y, x, dir_, agent_type in zip(*agent_locs, agent_dirs, self._agent_types)
        ]
        self._recalc_grid()

        self.request_queue = list(
            np.random.choice(self.shelfs, size=self.request_queue_size, replace=False)
        )
        self._targets = np.zeros(len(self.agents), dtype=int)
        return tuple([self._make_obs(agent) for agent in self.agents])
        # for s in self.shelfs:
        #     self.grid[0, s.y, s.x] = 1
        # print(self.grid[0])

    def resolve_move_conflict(self, agent_list):
        # # stationary agents will certainly stay where they are
        # stationary_agents = [agent for agent in self.agents if agent.action != Action.FORWARD]

        # forward agents will move only if they avoid collisions
        # forward_agents = [agent for agent in self.agents if agent.action == Action.FORWARD]
        
        commited_agents = set()

        G = nx.DiGraph()
        
        for agent in agent_list:
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)

            if (
                agent.carrying_shelf
                and start != target
                and self.grid[_LAYER_SHELFS, target[1], target[0]]
                and not (
                    self.grid[_LAYER_AGENTS, target[1], target[0]]
                    and self.agents[
                        self.grid[_LAYER_AGENTS, target[1], target[0]] - 1
                    ].carrying_shelf
                )
            ):
                # there's a standing shelf at the target location
                # our agent is carrying a shelf so there's no way
                # this movement can succeed. Cancel it.
                agent.req_action = Action.NOOP
                G.add_edge(start, start)
            else:
                G.add_edge(start, target)

        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        for comp in wcomps:
            try:
                # if we find a cycle in this component we have to
                # commit all nodes in that cycle, and nothing else
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    # we have a situation like this: [A] <-> [B]
                    # which is physically impossible. so skip
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[_LAYER_AGENTS, start_node[1], start_node[0]]
                    # action = self.agents[agent_id - 1].req_action
                    # print(f"{agent_id}: C {cycle} {action}")
                    if agent_id > 0:
                        commited_agents.add(agent_id)
            except nx.NetworkXNoCycle:

                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)
        
        for agent in agent_list:
            for other in agent_list:
                if agent.id != other.id:
                    new_x, new_y = agent.req_location(self.grid_size)
                    if new_x == other.x and new_y == other.y:
                        agent.req_action = Action.NOOP
                        if other.fixing_clash==0:
                            if agent.path and agent.path[-1] == (other.x, other.y):
                                agent.busy = False
                            if other.req_action == Action.FORWARD or other.req_action == Action.NOOP or other.req_action == Action.TOGGLE_LOAD:
                                agent.fixing_clash = self.fixing_clash_time
                                new_path = self.find_path((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), agent)
                                if new_path:
                                    agent.path = new_path
                                else:
                                    agent.fixing_clash = 0

        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(agent_list) - commited_agents
        for agent in failed_agents:
            agent.req_action = Action.NOOP

    def step(
        self, macro_actions: List[Action]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        # Logic for Macro Actions
        for agent, macro_action in zip(self.agents, macro_actions):
            # Initialize action for step
            agent.req_action = Action.NOOP
            # Collision avoidance logic
            if agent.fixing_clash > 0:
                agent.fixing_clash -= 1
                # continue
            if not agent.busy:
                if macro_action != 0:
                    agent.path = self.find_path((agent.y, agent.x), self.item_loc_dict[macro_action], agent)
                    # If not path was found refuse location
                    if agent.path == []:
                        agent.busy = False
                    else:
                        agent.busy = True
                        agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                        self._targets[agent.id-1] = macro_action
                        self.same_action_count[agent.id - 1] = 1
            else:
                # Check agent finished the give path if not continue the path
                if agent.path == []:
                    if agent.type != AgentType.PICKER:
                        agent.req_action = Action.TOGGLE_LOAD
                    if agent.type != AgentType.AGV:
                        agent.busy = False
                else:
                    agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                    self.same_action_count[agent.id - 1] += 1
        # Unfreeze agents if stuck following an "impossible" action
        for agent, count in zip(self.agents, self.same_action_count):
            if count == self._same_action_threshold:
                agent.busy = False
        #  agents that can_carry should not collide
        carry_agents = [agent for agent in self.agents if agent.type != AgentType.PICKER]
        self.resolve_move_conflict(carry_agents)
        
        rewards = np.zeros(self.n_agents_)
        # Add step penalty
        # rewards -= 0.01
        for agent in self.agents:
            agent.prev_x, agent.prev_y = agent.x, agent.y
            if agent.req_action == Action.FORWARD:
                agent.x, agent.y = agent.req_location(self.grid_size)
                agent.path = agent.path[1:]
                if agent.carrying_shelf:
                    agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                agent.dir = agent.req_direction()
            elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_shelf and agent.type != AgentType.PICKER:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                if shelf_id:
                    if agent.type == AgentType.AGV:
                        picker_id = self.grid[_LAYER_PICKERS, agent.y, agent.x]
                        if picker_id:
                            agent.carrying_shelf = self.shelfs[shelf_id - 1]
                            agent.busy = False
                            # Reward Pickers for loading shelf
                            if self.reward_type == RewardType.GLOBAL:
                                rewards += 0.5
                            elif self.reward_type == RewardType.INDIVIDUAL:
                                rewards[picker_id - 1] += 0.5
                                # rewards[agent.id - 1] += 0.25
                    elif agent.type == AgentType.AGENT:
                        agent.carrying_shelf = self.shelfs[shelf_id - 1]
                        agent.busy = False
                else:
                    agent.busy = False
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_shelf:
                picker_id = self.grid[_LAYER_PICKERS, agent.y, agent.x]
                if (agent.x, agent.y) in self.goals:
                    agent.busy=False
                    continue
                if not self._is_highway(agent.x, agent.y):
                    if agent.type == AgentType.AGENT:
                        agent.carrying_shelf = None
                        agent.busy=False
                    if agent.type == AgentType.AGV and picker_id:
                        agent.carrying_shelf = None
                        agent.busy=False
                        # Reward Pickers for un-loading shelf
                        if self.reward_type == RewardType.GLOBAL:
                            rewards += 0.5
                        elif self.reward_type == RewardType.INDIVIDUAL:
                            rewards[picker_id - 1] += 0.5
                    if agent.has_delivered and self.reward_type == RewardType.TWO_STAGE:
                        # rewards[agent.id - 1] += 0.5
                        raise NotImplementedError('TWO_STAGE reward not implemenred for diverse rware')
                    agent.has_delivered = False
        shelf_delivered = False
        shelf_deliveries = 0
        for y, x in self.goals:
            shelf_id = self.grid[_LAYER_SHELFS, x, y]
            if not shelf_id:
                continue
            shelf = self.shelfs[shelf_id - 1]

            if shelf not in self.request_queue:
                continue
            # a shelf was successfully delived.
            shelf_delivered = True
            shelf_deliveries += 1
            # remove from queue and replace it
            new_request = np.random.choice(
                list(set(self.shelfs) - set(self.request_queue))
            )
            self.request_queue[self.request_queue.index(shelf)] = new_request

            if self.no_need_return_item:
                agent.carrying_shelf = None
                for sx, sy in zip(
                    np.indices(self.grid_size)[0].reshape(-1),
                    np.indices(self.grid_size)[1].reshape(-1),
                ): 
                    if not self._is_highway(sy, sx) and not self.grid[_LAYER_SHELFS, sy, sx]:
                        print(f"{sx}-{sy}")
                        self.shelfs[shelf_id - 1].x = sx
                        self.shelfs[shelf_id - 1].y = sy
                        self.grid[_LAYER_SHELFS, sy, sx] = shelf_id
                        break
            # also reward the agents
            if self.reward_type == RewardType.GLOBAL:
                rewards += 1
            elif self.reward_type == RewardType.INDIVIDUAL:
                agent_id = self.grid[_LAYER_AGENTS, x, y]
                rewards[agent_id - 1] += 1
            elif self.reward_type == RewardType.TWO_STAGE:
                agent_id = self.grid[_LAYER_AGENTS, x, y]
                self.agents[agent_id - 1].has_delivered = True
                rewards[agent_id - 1] += 1
        self._recalc_grid()

        if shelf_delivered:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1
        self._cur_steps += 1

        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps):
            dones = self.n_agents_ * [True]
        else:
            dones = self.n_agents_ * [False]

        new_obs = tuple([self._make_obs(agent) for agent in self.agents])
        info = {}
        info["vehicles_busy"] = [agent.busy for agent in self.agents]
        info["shelf_deliveries"] = shelf_deliveries
        return new_obs, list(rewards), dones, info

    def render(self, mode="human"):
        if not self.renderer:
            from rware.rendering import Viewer

            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        ...
    

if __name__ == "__main__":
    env = Warehouse(9, 8, 3, 10, 3, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    import time
    from tqdm import tqdm

    time.sleep(2)
    # env.render()
    # env.step(18 * [Action.LOAD] + 2 * [Action.NOOP])

    for _ in tqdm(range(1000000)):
        # time.sleep(2)
        # env.render()
        actions = env.action_space.sample()
        env.step(actions)
