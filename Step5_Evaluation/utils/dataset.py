import numpy as np
import torch
import pickle
from collections import namedtuple
import copy

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
        self.record_goal_traj = []

        if if_test:
            data = data[150:]
        else:
            data = data[:150]

        for pair in data:
            end_frame = pair['end_frame'].values[0]
            decision_frame = pair['decision_frame'].values[0]
            lcv = pair[['frame', 'id_x', 'y_x', 'x_x', 'width_x', 'height_x', 'laneId_x']].iloc[20:]
            fv = pair[['frame', 'id_x', 'y_y', 'x_y', 'width_y', 'height_y', 'laneId_y']].iloc[20:]
            nlv = pair[['frame', 'id', 'y', 'x', 'width', 'height', 'laneId']].iloc[20:]
            olv = pair[['frame', 'id_z', 'y_z', 'x_z', 'width_z', 'height_z', 'laneId_z']].iloc[20:]

            lcv = trajectory_process(lcv, decision_frame)
            fv = trajectory_process(fv, decision_frame)
            nlv = trajectory_process(nlv, decision_frame)
            olv = trajectory_process(olv, decision_frame)

            base_y = copy.deepcopy(lcv['y'].values[0])
            base_x = copy.deepcopy(lcv['x'].values[0])

            self.record_trajectory.append(copy.deepcopy(lcv))
            self.record_path_length.append(len(lcv))

            lcv['y'] -= base_y
            lcv['x'] -= base_x
            fv['y'] -= base_y
            fv['x'] -= base_x
            nlv['y'] -= base_y
            nlv['x'] -= base_x
            olv['y'] -= base_y
            olv['x'] -= base_x

            self.record_goal_traj.append([lcv.iloc[:20], fv.iloc[:20], nlv.iloc[:20], olv.iloc[:20]])

        self.indices = self.make_indices(self.record_path_length, horizon)
        self.observation_dim = 4
        self.action_dim = 2

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
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
                break
            # break
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {
            0: observations[:, 0],
            99: observations[:, 99],
            # self.horizon - 1: observations[-1],
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.record_trajectory[path_ind][['y', 'x', 'v_x', 'v_y']].iloc[start:end].values
        actions = self.record_trajectory[path_ind][['a_x', 'a_y']].iloc[start:end].values
        lcv = self.record_goal_traj[path_ind][0][['y', 'x', 'v_x', 'v_y']].values
        fv = self.record_goal_traj[path_ind][1][['y', 'x', 'v_x', 'v_y']].values
        nlv = self.record_goal_traj[path_ind][2][['y', 'x', 'v_x', 'v_y']].values
        olv = self.record_goal_traj[path_ind][3][['y', 'x', 'v_x', 'v_y']].values
        lcv[:, 0] /= 50
        lcv[:, 1] /= 500
        lcv[:, 2] = (lcv[:, 2] + 40) / 80
        lcv[:, 3] = (lcv[:, 3] + 40) / 80

        fv[:, 0] /= 50
        fv[:, 1] /= 500
        fv[:, 2] = (fv[:, 2] + 40) / 80
        fv[:, 3] = (fv[:, 3] + 40) / 80

        nlv[:, 0] /= 50
        nlv[:, 1] /= 500
        nlv[:, 2] = (nlv[:, 2] + 40) / 80
        nlv[:, 3] = (nlv[:, 3] + 40) / 80

        olv[:, 0] /= 50
        olv[:, 1] /= 500
        olv[:, 2] = (olv[:, 2] + 40) / 80
        olv[:, 3] = (olv[:, 3] + 40) / 80

        goal_state = [torch.from_numpy(lcv).float(), torch.from_numpy(fv).float(),
                      torch.from_numpy(nlv).float(), torch.from_numpy(olv).float()]
        conditions = self.get_conditions(torch.from_numpy(observations[None, :, :]))
        trajectories = torch.from_numpy(np.concatenate([actions, observations], axis=-1))
        batch = (trajectories[None, :, :], conditions, goal_state)

        return batch





