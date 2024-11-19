from __future__ import division, print_function, absolute_import

import torch
import pickle

from NGSIM_env.envs.common.abstract import AbstractEnv
from NGSIM_env.road.road import Road, RoadNetwork
from NGSIM_env.road.lane import LineType, StraightLane
from NGSIM_env.utils import *
from NGSIM_env.data.data_process import build_trajecotry


class NGSIMEnv(AbstractEnv):
    """
    A highway driving environment with NGSIM data.
    """

    def __init__(self, scene):
        # self.scene_id = scene_id
        self.scene = scene
        self.trajectory_set = build_trajecotry()
        super(NGSIMEnv, self).__init__()

    def reset(self, scene_id=0):
        '''
        Reset the environment at a given time (scene) and specify whether use human target
        '''

        self.lcv_length = self.trajectory_set['lcv'][scene_id]['width'].values[0]
        self.lcv_width = self.trajectory_set['lcv'][scene_id]['height'].values[0]
        self.lcv_trajectory = self.trajectory_set['lcv'][scene_id]

        self.fv_length = self.trajectory_set['fv'][scene_id]['width'].values[0]
        self.fv_width = self.trajectory_set['fv'][scene_id]['height'].values[0]
        self.fv_trajectory = self.trajectory_set['fv'][scene_id]

        self.nlv_length = self.trajectory_set['nlv'][scene_id]['width'].values[0]
        self.nlv_width = self.trajectory_set['nlv'][scene_id]['height'].values[0]
        self.nlv_trajectory = self.trajectory_set['nlv'][scene_id]

        self.olv_length = self.trajectory_set['olv'][scene_id]['width'].values[0]
        self.olv_width = self.trajectory_set['olv'][scene_id]['height'].values[0]
        self.olv_trajectory = self.trajectory_set['olv'][scene_id]

        self.duration = len(self.lcv_trajectory)
        self._create_road()
        state = self._create_vehicles()
        self.steps = 0

        return state

    def init_vehicle(self, trajectory):
        init_v = {'x': trajectory[0], 'y': trajectory[1], 'vx': trajectory[3],
                  'vy': trajectory[4], 'ax': trajectory[5], 'ay': trajectory[6]}

        return init_v

    def normalize_obs(self, df):
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        self.features_range = [[0, 25], [0, 500], [-2*20, 2*20], [-2*20, 2*20], [-5, 5], [-2, 2]]

        for i, f_range in enumerate(self.features_range):
            df[i] = (df[i] - f_range[0]) / (f_range[1] - f_range[0])
            df[i] = np.clip(df[i], -1, 1)

        return df

    def _create_road(self):
        """
        Create a road composed of NGSIM road network
        """
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        if self.scene == 'us-101':
            length = 2150 / 3.281  # m
            width = 12 / 3.281  # m
            ends = [0, 560 / 3.281, (698 + 578 + 150) / 3.281, length]

            # first section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(5):
                origin = [ends[0], lane * width]
                end = [ends[1], lane * width]
                net.add_lane('s1', 's2', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('merge_in', 's2',
                         StraightLane([480 / 3.281, 5.5 * width], [ends[1], 5 * width], width=width, line_types=[c, c],
                                      forbidden=True))

            # second section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(6):
                origin = [ends[1], lane * width]
                end = [ends[2], lane * width]
                net.add_lane('s2', 's3', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # third section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(5):
                origin = [ends[2], lane * width]
                end = [ends[3], lane * width]
                net.add_lane('s3', 's4', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_out lanes
            net.add_lane('s3', 'merge_out',
                         StraightLane([ends[2], 5 * width], [1550 / 3.281, 7 * width], width=width, line_types=[c, c],
                                      forbidden=True))

            self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

        elif self.scene == 'i-80':
            length = 1700 / 3.281
            lanes = 6
            width = 12 / 3.281
            ends = [0, 600 / 3.281, 700 / 3.281, 900 / 3.281, length]

            # first section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(lanes):
                origin = [ends[0], lane * width]
                end = [ends[1], lane * width]
                net.add_lane('s1', 's2', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('s1', 's2',
                         StraightLane([380 / 3.281, 7.1 * width], [ends[1], 6 * width], width=width, line_types=[c, c],
                                      forbidden=True))

            # second section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, n]]
            for lane in range(lanes):
                origin = [ends[1], lane * width]
                end = [ends[2], lane * width]
                net.add_lane('s2', 's3', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('s2', 's3',
                         StraightLane([ends[1], 6 * width], [ends[2], 6 * width], width=width, line_types=[s, c]))

            # third section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, n]]
            for lane in range(lanes):
                origin = [ends[2], lane * width]
                end = [ends[3], lane * width]
                net.add_lane('s3', 's4', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lane
            net.add_lane('s3', 's4',
                         StraightLane([ends[2], 6 * width], [ends[3], 5 * width], width=width, line_types=[n, c]))

            # forth section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(lanes):
                origin = [ends[3], lane * width]
                end = [ends[4], lane * width]
                net.add_lane('s4', 's5', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self):
        """
        Create ego vehicle and NGSIM vehicles and add them on the road.
        """
        lcv_trajectory = self.lcv_trajectory[['y', 'x', 'laneId', 'v_x', 'v_y', 'a_x', 'a_y']]
        lcv_trajectory = lcv_trajectory.values[0]
        self.lcv_veh = self.init_vehicle(lcv_trajectory)
        # self.road.vehicles.append(self.lcv_veh)

        fv_trajectory = self.fv_trajectory[['y', 'x', 'laneId', 'v_x', 'v_y', 'a_x', 'a_y']]
        fv_trajectory = fv_trajectory.values[0]
        self.fv_veh = self.init_vehicle(fv_trajectory)
        # self.road.vehicles.append(self.fv_veh)

        nlv_trajectory = self.nlv_trajectory[['y', 'x', 'laneId', 'v_x', 'v_y', 'a_x', 'a_y']]
        nlv_trajectory = nlv_trajectory.values[0]
        self.nlv_veh = self.init_vehicle(nlv_trajectory)

        olv_trajectory = self.olv_trajectory[['y', 'x', 'laneId', 'v_x', 'v_y', 'a_x', 'a_y']]
        olv_trajectory = olv_trajectory.values[0]
        self.olv_veh = self.init_vehicle(olv_trajectory)

        lcv_veh = self.normalize_obs(np.array(list(self.lcv_veh.values())))[:-2]
        fv_veh = self.normalize_obs(np.array(list(self.fv_veh.values())))[:-2]
        nlv_veh = self.normalize_obs(np.array(list(self.nlv_veh.values())))[:-2]
        olv_veh = self.normalize_obs(np.array(list(self.olv_veh.values())))[:-2]

        fv_veh = lcv_veh - fv_veh
        nlv_veh = nlv_veh - lcv_veh
        olv_veh = olv_veh - lcv_veh

        state = np.concatenate([lcv_veh, fv_veh, nlv_veh, olv_veh]).reshape(-1)

        return state

    def step(self, action):
        """
        Perform a MDP step
        """
        if self.road is None:
            raise NotImplementedError("The road must be initialized in the environment implementation")
        self.steps += 1

        # lcv
        lcv_ax, lcv_ay = action[0], action[1]
        self.lcv_veh['vx'] += lcv_ax * 0.1  # x
        self.lcv_veh['vy'] += lcv_ay * 0.1  # y
        self.lcv_veh['x'] += self.lcv_veh['vy'] * 0.1  # y
        self.lcv_veh['y'] += self.lcv_veh['vx'] * 0.1  # x
        self.lcv_veh['ax'] = lcv_ax
        self.lcv_veh['ay'] = lcv_ay

        # fv
        fv_ax, fv_ay = action[2], action[3]
        self.fv_veh['vx'] += fv_ax * 0.1  # x
        self.fv_veh['vy'] += fv_ay * 0.1  # y
        self.fv_veh['x'] += self.fv_veh['vy'] * 0.1  # y
        self.fv_veh['y'] += self.fv_veh['vx'] * 0.1  # x
        self.fv_veh['ax'] = fv_ax
        self.fv_veh['ay'] = fv_ay
        # nlv and olv
        nlv_trajectory = self.nlv_trajectory[['y', 'x', 'laneId', 'v_x', 'v_y', 'a_x', 'a_y']]
        nlv_trajectory = nlv_trajectory.values[self.steps]
        self.nlv_veh = self.init_vehicle(nlv_trajectory)
        olv_trajectory = self.olv_trajectory[['y', 'x', 'laneId', 'v_x', 'v_y', 'a_x', 'a_y']]
        olv_trajectory = olv_trajectory.values[self.steps]
        self.olv_veh = self.init_vehicle(olv_trajectory)

        lcv_veh = self.normalize_obs(np.array(list(self.lcv_veh.values())))[:-2]
        fv_veh = self.normalize_obs(np.array(list(self.fv_veh.values())))[:-2]
        nlv_veh = self.normalize_obs(np.array(list(self.nlv_veh.values())))[:-2]
        olv_veh = self.normalize_obs(np.array(list(self.olv_veh.values())))[:-2]

        fv_veh = lcv_veh - fv_veh
        nlv_veh = nlv_veh - lcv_veh
        olv_veh = olv_veh - lcv_veh

        state = np.concatenate([lcv_veh, fv_veh, nlv_veh, olv_veh]).reshape(-1)
        terminal = self._is_terminal()
        info = {}

        norm_xy = np.array([500, 25])
        lcv_label_pos = self.lcv_trajectory[['x', 'y']].values[self.steps] / norm_xy
        fv_label_pos = self.fv_trajectory[['x', 'y']].values[self.steps] / norm_xy
        lcv_pred_pos = np.array([self.lcv_veh['y'], self.lcv_veh['x']]) / norm_xy
        fv_pred_pos = np.array([self.fv_veh['y'], self.fv_veh['x']]) / norm_xy
        reward = np.sqrt(((lcv_pred_pos - lcv_label_pos) ** 2).mean()) + \
                 np.sqrt(((fv_label_pos - fv_pred_pos) ** 2).mean())

        return state, -reward, terminal, info

    def _is_terminal(self):
        """
        The episode is over if the ego vehicle crashed or go off road or the time is out.
        """
        return self.steps > self.duration - 2

    def generate_experts(self):
        expert_dict = {'state': [], 'action': [], 'rewards': [], 'dones': [], 'next_states': []}

        for scene_id in range(150):
            state = self.reset(scene_id=scene_id)
            lcv_trajectory = self.lcv_trajectory[['y', 'x', 'laneId', 'v_x', 'v_y', 'a_x', 'a_y']]
            lcv_trajectory = lcv_trajectory.values
            fv_trajectory = self.fv_trajectory[['y', 'x', 'laneId', 'v_x', 'v_y', 'a_x', 'a_y']]
            fv_trajectory = fv_trajectory.values

            for t in range(self.duration-1):
                action_t = [lcv_trajectory[t, -2], lcv_trajectory[t, -1],
                            fv_trajectory[t, -2], fv_trajectory[t, -1]]
                next_state, reward, terminal, info = self.step(action_t)
                expert_dict['state'].append(state.reshape(1, -1))
                expert_dict['action'].append(np.array(action_t).reshape(1, -1))
                expert_dict['rewards'].append(np.array([0]))
                expert_dict['dones'].append(np.array([terminal]))
                expert_dict['next_states'].append(next_state.reshape(1, -1))

                state = next_state

        expert_dict['state'] = torch.from_numpy(np.concatenate(expert_dict['state']))
        expert_dict['action'] = torch.from_numpy(np.concatenate(expert_dict['action']))
        expert_dict['rewards'] = torch.from_numpy(np.concatenate(expert_dict['rewards'])).reshape(-1, 1)
        expert_dict['dones'] = torch.from_numpy(np.concatenate(expert_dict['dones'])).reshape(-1, 1)
        expert_dict['next_states'] = torch.from_numpy(np.concatenate(expert_dict['next_states']))

        with open('expert_data.pkl', 'wb') as path:
            pickle.dump(expert_dict, path)


