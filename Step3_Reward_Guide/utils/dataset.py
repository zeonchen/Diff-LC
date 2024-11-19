import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import random


def data_loader(obs_len, pred_len, skip, batch_size, path, shuffle=True, save_path='', if_test='train'):
    dset = TrajectoryDataset(
        path,
        obs_len=obs_len,
        pred_len=pred_len,
        skip=skip, save_dir=save_path, if_test=if_test)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle)
    return loader


def trajectory_process(trajectory, decision_time):
    trajectory.columns = ['frame', 'id', 'y', 'x', 'width', 'height', 'laneId']
    trajectory['v_x'] = trajectory['x'].diff() / 0.1
    trajectory['v_y'] = trajectory['y'].diff() / 0.1
    trajectory['a_x'] = trajectory['v_x'].diff() / 0.1
    trajectory['a_y'] = trajectory['v_y'].diff() / 0.1

    trajectory = trajectory[trajectory['frame'] >= decision_time - 20]

    return trajectory


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=3,
              save_dir='train_data.pkl', if_test='train'):
        super(TrajectoryDataset, self).__init__()
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len

        file = open(save_dir, 'rb')
        all_data = pickle.load(file)
        file.close()

        lcv_data = []
        fv_data = []
        nlv_data = []
        olv_data = []

        assert if_test in ['train', 'val', 'test']
        if if_test == 'train':
            all_data = all_data[:150]
        else:
            all_data = all_data[150:]

        for pair in all_data:
            nlv = pair[['frame', 'id', 'y', 'x', 'width', 'height', 'laneId']].iloc[20:]
            olv = pair[['frame', 'id_z', 'y_z', 'x_z', 'width_z', 'height_z', 'laneId_z']].iloc[20:]
            lcv = pair[['frame', 'id_x', 'y_x', 'x_x', 'width_x', 'height_x', 'laneId_x']].iloc[20:]
            fv = pair[['frame', 'id_y', 'y_y', 'x_y', 'width_y', 'height_y', 'laneId_y']].iloc[20:]
            decision_frame = pair['decision_frame'].values[0]

            nlv = trajectory_process(nlv, decision_frame).values[:100, [2, 3, 7, 8]]
            olv = trajectory_process(olv, decision_frame).values[:100, [2, 3, 7, 8]]
            lcv = trajectory_process(lcv, decision_frame).values[:100, [2, 3, 7, 8]]
            fv = trajectory_process(fv, decision_frame).values[:100, [2, 3, 7, 8]]

            lcv_data.append(lcv[None, :, :])
            fv_data.append(fv[None, :, :])
            nlv_data.append(nlv[None, :, :])
            olv_data.append(olv[None, :, :])

        self.lcv_traj = torch.from_numpy(np.concatenate(lcv_data)).type(torch.float)
        self.fv_traj = torch.from_numpy(np.concatenate(fv_data)).type(torch.float)
        self.nlv_traj = torch.from_numpy(np.concatenate(nlv_data)).type(torch.float)
        self.olv_traj = torch.from_numpy(np.concatenate(olv_data)).type(torch.float)

    def __len__(self):
        return len(self.lcv_traj)

    def __getitem__(self, index):

        out = [
            self.lcv_traj[index, :], self.fv_traj[index, :],
            self.nlv_traj[index, :], self.olv_traj[index, :]
        ]
        return out