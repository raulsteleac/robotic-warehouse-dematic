import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tarware.definitions import Action, AgentType, CollisionLayers
from tarware.spaces.MultiAgentBaseObservationSpace import (
    MultiAgentBaseObservationSpace, _VectorWriter)


class MultiAgentPartialObservationSpace(MultiAgentBaseObservationSpace):
    def __init__(self, num_agvs, num_pickers, grid_size, shelf_locations, normalised_coordinates=False):
        super(MultiAgentPartialObservationSpace, self).__init__(num_agvs, num_pickers, grid_size, shelf_locations, normalised_coordinates)

        self._define_obs_length_agvs()
        self._define_obs_length_pickers()
        self.agv_obs_lengths = [self._obs_length_agvs for _ in range(self.num_agvs)]
        self.picker_obs_lengths = [self._obs_length_pickers for _ in range(self.num_pickers)]

        ma_spaces = []
        for obs_length in self.agv_obs_lengths + self.picker_obs_lengths:
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(obs_length,),
                    dtype=np.float32,
                )
            ]

        self.ma_spaces = spaces.Tuple(tuple(ma_spaces))

    def _define_obs_length_agvs(self):
        location_space = spaces.Box(low=0.0, high=max(self.grid_size), shape=(2,), dtype=np.float32)

        self.agvs_obs_bits_for_agvs = (spaces.flatdim(location_space)  + spaces.flatdim(location_space)) * self.num_agvs
        self.agvs_obs_bits_for_pickers = (spaces.flatdim(location_space)  + spaces.flatdim(location_space)) * self.num_pickers
        self.agvs_obs_bits_per_shelf = 1 * self.shelf_locations
        self.agvs_obs_bits_for_requests = 1 * self.shelf_locations

        self._obs_length_agvs = (
            self.agvs_obs_bits_for_agvs
            + self.agvs_obs_bits_for_pickers
            + self.agvs_obs_bits_per_shelf
            + self.agvs_obs_bits_for_requests
        )

    def _define_obs_length_pickers(self):
        location_space = spaces.Box(low=0.0, high=max(self.grid_size), shape=(2,), dtype=np.float32)

        self.pickers_obs_bits_for_agvs = (3 + spaces.flatdim(location_space)  + spaces.flatdim(location_space)) * self.num_agvs
        self.pickers_obs_bits_for_pickers = (spaces.flatdim(location_space)  + spaces.flatdim(location_space)) * self.num_pickers

        self._obs_length_pickers = (
            self.pickers_obs_bits_for_agvs
            + self.pickers_obs_bits_for_pickers
        )

    def observation(self, agent, environment):        
        # write flattened observations
        obs = _VectorWriter(self.ma_spaces[agent.id - 1].shape[0])
        # Agent self observation
        obs.write(self.process_coordinates((agent.y, agent.x), environment))
        if agent.target:
            obs.write(self.process_coordinates(environment.action_id_to_coords_map[agent.target], environment))
        else:
            obs.skip(2)

        # Other agents observation
        for agent_ in environment.agents:
            if agent_.id != agent.id:
                if agent.type == AgentType.PICKER and agent_.type == AgentType.AGV:
                    if agent_.carrying_shelf:
                        obs.write([1, int(agent_.carrying_shelf in environment.request_queue)])
                    else:
                        obs.skip(2)
                    obs.write([agent_.req_action == Action.TOGGLE_LOAD])
                obs.write(self.process_coordinates((agent_.y, agent_.x), environment))
                if agent_.target:
                    obs.write(self.process_coordinates(environment.action_id_to_coords_map[agent_.target], environment))
                else:
                    obs.skip(2)

        # Shelves observation
        if agent.type == AgentType.AGV:
            for group in environment.rack_groups:
                for (x, y) in group:
                    id_shelf = environment.grid[CollisionLayers.SHELFS, x, y]
                    if id_shelf!=0:
                        obs.write([1.0 , int(environment.shelfs[id_shelf - 1] in environment.request_queue)])
                    else:
                        obs.skip(2)
        return obs.vector