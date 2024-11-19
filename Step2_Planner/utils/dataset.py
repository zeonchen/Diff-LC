import numpy as np
import torch
import pickle
from collections import namedtuple

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


def trajectory_process(trajectory, decision_time):
    trajectory.columns = ['frame', 'id', 'y', 'x', 'width', 'height', 'laneId']
    trajectory['v_x'] = trajectory['x'].diff() / 0.1
    trajectory['v_y'] = trajectory['y'].diff() / 0.1
    trajectory['a_x'] = trajectory['v_x'].diff() / 0.1
    trajectory['a_y'] = trajectory['v_y'].diff() / 0.1

    trajectory = trajectory[trajectory['frame'] >= decision_time - 20]

    return trajectory


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, horizon=32, use_padding=False, if_test=False):
        self.horizon = horizon
        self.use_padding = use_padding
        self.max_path_length = 151

        with open(data_path, "rb") as input_file:
            data = pickle.load(input_file)

        self.record_trajectory = []
        self.record_path_length = []
        self.if_test = False

        if if_test:
            data = data[150:]
            self.if_test = True
        else:
            data = data[:150]

        for pair in data:
            end_frame = pair['end_frame'].values[0]
            decision_frame = pair['decision_frame'].values[0]
            lcv = pair[['frame', 'id_x', 'y_x', 'x_x', 'width_x', 'height_x', 'laneId_x']].iloc[20:]
            lcv = trajectory_process(lcv, decision_frame)
            lcv['y'] -= lcv['y'].values[0]
            lcv['x'] -= lcv['x'].values[0]
            self.record_trajectory.append(lcv)
            self.record_path_length.append(len(lcv))

        self.indices = self.make_indices(self.record_path_length, horizon)
        self.observation_dim = 4
        self.action_dim = 2

    def normalize(self):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        pass

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            # max_start = path_length - 1
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))

                if self.if_test:
                    break
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.record_trajectory[path_ind][['y', 'x', 'v_x', 'v_y']].iloc[start:end].values
        actions = self.record_trajectory[path_ind][['a_x', 'a_y']].iloc[start:end].values

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        batch = Batch(trajectories, conditions)
        return batch





