import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import copy


def normalize_obs(df):
    """
    Normalize the observation values.

    For now, assume that the road is straight along the x axis.
    :param Dataframe df: observation data
    """
    features_range = [[0, 25], [0, 500], [-2*20, 2*20], [-2*20, 2*20]]

    for i, f_range in enumerate(features_range):
        df[:, i] = (df[:, i] - f_range[0]) / (f_range[1] - f_range[0])
        df[:, i] = np.clip(df[:, i], -1, 1)

    return df


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
        self, data_dir, obs_len=8, pred_len=12, skip=1,
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

        hist_data = []
        pred_data = []
        cond_data = []
        state_data = []

        assert if_test in ['train', 'val', 'test']
        if if_test == 'train':
            all_data = all_data[:150]
        else:
            all_data = all_data[150:]

        for pair in all_data:
            nlv = pair[['frame', 'id', 'y', 'x', 'width', 'height', 'laneId']]
            olv = pair[['frame', 'id_z', 'y_z', 'x_z', 'width_z', 'height_z', 'laneId_z']]
            lcv = pair[['frame', 'id_x', 'y_x', 'x_x', 'width_x', 'height_x', 'laneId_x']]
            fv = pair[['frame', 'id_y', 'y_y', 'x_y', 'width_y', 'height_y', 'laneId_y']]
            decision_frame = pair['decision_frame'].values[0]

            nlv = trajectory_process(nlv, decision_frame)
            olv = trajectory_process(olv, decision_frame)
            lcv = trajectory_process(lcv, decision_frame)
            fv = trajectory_process(fv, decision_frame)

            hist = fv[['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[:20, :]
            pred = fv[['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[20:120, :]
            cond = lcv[['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[20:120, :]
            nlv = nlv[['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[20:120, :]
            olv = olv[['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[20:120, :]

            n_lcv = normalize_obs(copy.deepcopy(cond))[:, :4]
            n_fv = normalize_obs(copy.deepcopy(pred))[:, :4]
            n_nlv = normalize_obs(copy.deepcopy(nlv))[:, :4]
            n_olv = normalize_obs(copy.deepcopy(olv))[:, :4]
            dfv = n_lcv - n_fv
            dnlv = n_nlv - n_lcv
            dolv = n_olv - n_lcv
            state = torch.from_numpy(np.concatenate([n_lcv, dfv, dnlv, dolv], axis=-1)).float()

            cond[:, :2] -= hist[0, :2]

            hist_data.append(hist[None, :, :])
            cond_data.append(cond[None, :, :])
            pred_data.append(pred[None, :, :])
            state_data.append(state[None, :, :])

        self.hist_traj = torch.from_numpy(np.concatenate(hist_data)).type(torch.float)
        self.cond_traj = torch.from_numpy(np.concatenate(cond_data)).type(torch.float)
        self.pred_traj = torch.from_numpy(np.concatenate(pred_data)).type(torch.float)
        self.state_data = torch.from_numpy(np.concatenate(state_data)).type(torch.float)

    def __len__(self):
        return len(self.hist_traj)

    def __getitem__(self, index):

        out = [
            self.hist_traj[index, :], self.cond_traj[index, :],
            self.pred_traj[index, :], self.state_data[index, :]
        ]
        return out