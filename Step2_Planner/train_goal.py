import copy
import torch.nn as nn
from utils.arrays import *
import warnings
warnings.filterwarnings('ignore')

import pickle
from collections import namedtuple

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


def cycle(dl):
    while True:
        for data in dl:
            yield data


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
        self.record_goal = []
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
            fv = pair[['frame', 'id_x', 'y_y', 'x_y', 'width_y', 'height_y', 'laneId_y']].iloc[20:]
            nlv = pair[['frame', 'id', 'y', 'x', 'width', 'height', 'laneId']].iloc[20:]
            olv = pair[['frame', 'id_z', 'y_z', 'x_z', 'width_z', 'height_z', 'laneId_z']].iloc[20:]

            lcv = trajectory_process(lcv, decision_frame)
            fv = trajectory_process(fv, decision_frame)
            nlv = trajectory_process(nlv, decision_frame)
            olv = trajectory_process(olv, decision_frame)

            base_y = copy.deepcopy(lcv['y'].values[0])
            base_x = copy.deepcopy(lcv['x'].values[0])

            lcv['y'] -= base_y
            lcv['x'] -= base_x
            fv['y'] -= base_y
            fv['x'] -= base_x
            nlv['y'] -= base_y
            nlv['x'] -= base_x
            olv['y'] -= base_y
            olv['x'] -= base_x
            self.record_trajectory.append([lcv.iloc[:20], fv.iloc[:20], nlv.iloc[:20], olv.iloc[:20]])
            self.record_goal.append(lcv.iloc[100] - lcv.iloc[0])

        if not if_test:
            all_x = []
            all_v = []
            for idx in range(150):
                sub = self.record_goal[idx][['x', 'v_x']].values
                all_x.append(sub[0])
                all_v.append(sub[1])

            self.max_x, self.min_x = np.max(all_x), np.min(all_x)
            self.max_v, self.min_v = np.max(all_v), np.min(all_v)

    def __len__(self):
        return len(self.record_trajectory)

    def __getitem__(self, idx, eps=1e-4):
        lcv = self.record_trajectory[idx][0][['y', 'x', 'v_x', 'v_y']].values
        fv = self.record_trajectory[idx][1][['y', 'x', 'v_x', 'v_y']].values
        nlv = self.record_trajectory[idx][2][['y', 'x', 'v_x', 'v_y']].values
        olv = self.record_trajectory[idx][3][['y', 'x', 'v_x', 'v_y']].values
        goal = self.record_goal[idx][['x', 'v_x']].values  # delta x and delta v

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

        # goal[0] /= 500
        # goal[1] = (goal[1] + 40) / 80

        return [lcv, fv, nlv, olv, goal]


class GoalPredictor(torch.nn.Module):
    def __init__(self, context_dim, input_size=2):
        super(GoalPredictor, self).__init__()
        self.context_dim = context_dim

        self.lcv_gru = nn.GRU(input_size=16, hidden_size=context_dim, num_layers=1, batch_first=True)
        self.fv_gru = nn.GRU(input_size=4, hidden_size=context_dim, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(context_dim, context_dim),
                                 nn.ReLU(),
                                 nn.Linear(context_dim, 2),
                                 nn.Sigmoid())

    def forward(self, lcv, fv, nlv, olv):
        x = torch.cat([lcv, fv, nlv, olv], dim=-1)
        _, hidden_lcv = self.lcv_gru(x)

        pred = self.mlp(hidden_lcv[0])
        return pred


def evaluation(model, loader, bound_min, bound_max):
    model.eval()
    for batch in loader:
        b_lcv = batch[0].float().to(bound_min.device)
        b_fv = batch[1].float().to(bound_min.device)
        b_nlv = batch[2].float().to(bound_min.device)
        b_olv = batch[3].float().to(bound_min.device)
        b_goal = batch[4].float().to(bound_min.device)

        pred = model(b_lcv, b_fv, b_nlv, b_olv)

        pred = pred * (bound_max - bound_min) + bound_min

        x_rmse = (nn.functional.mse_loss(pred[:, 0], b_goal[:, 0], reduction='mean')).sqrt()
        v_rmse = (nn.functional.mse_loss(pred[:, 1], b_goal[:, 1], reduction='mean')).sqrt()

    return x_rmse.item(), v_rmse.item()


def main():
    device = 'cuda'
    data_path = 'data/sample_data.pkl'
    train_dataset = SequenceDataset(data_path, horizon=100)
    test_dataset = SequenceDataset(data_path, horizon=100, if_test=True)

    bound_min = torch.tensor([train_dataset.min_x, train_dataset.min_v]).to(device).float()
    bound_max = torch.tensor([train_dataset.max_x, train_dataset.max_v]).to(device).float()

    model = GoalPredictor(128).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                             num_workers=0, shuffle=True, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000,
                                              num_workers=0, shuffle=False, pin_memory=True)

    best_rmse = 1e6
    for epoch in range(100000):
        epoch_loss = []
        for batch in dataloader:
            model.train()
            b_lcv = batch[0].float().to(device)
            b_fv = batch[1].float().to(device)
            b_nlv = batch[2].float().to(device)
            b_olv = batch[3].float().to(device)
            b_goal = batch[4].float().to(device)

            b_goal = (b_goal - bound_min) / (bound_max - bound_min)
            pred = model(b_lcv, b_fv, b_nlv, b_olv)

            loss = nn.functional.mse_loss(pred, b_goal)
            loss.backward()
            optim.step()

            epoch_loss.append(loss.item())

        test_x, test_v = evaluation(model, test_loader, bound_min, bound_max)

        print('Epoch {}, Training loss {:.4}, test x {:.4}, test v {:.4}'.format(epoch,
                                                                                 np.mean(epoch_loss), test_x, test_v))

        if test_x < best_rmse:
            best_rmse = test_x
            torch.save(model, 'goal_model.pth')
            print('Best model saved!')


if __name__ == '__main__':
    main()
