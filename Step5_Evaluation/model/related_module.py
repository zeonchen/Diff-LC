import torch
import torch.nn as nn
import pickle
import numpy as np


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

def trajectory_process(trajectory, decision_time):
    trajectory.columns = ['frame', 'id', 'y', 'x', 'width', 'height', 'laneId']
    trajectory['v_x'] = trajectory['x'].diff() / 0.1
    trajectory['v_y'] = trajectory['y'].diff() / 0.1
    trajectory['a_x'] = trajectory['v_x'].diff() / 0.1
    trajectory['a_y'] = trajectory['v_y'].diff() / 0.1

    trajectory = trajectory[trajectory['frame'] >= decision_time - 20]

    return trajectory


def build_trajecotry():
    with open('sample_data.pkl', "rb") as input_file:
        data = pickle.load(input_file)

    record_trajectory = {'lcv': [], 'fv': [], 'nlv': [], 'olv': []}

    for pair in data[150:]:
        end_frame = pair['end_frame'].values[0]
        decision_frame = pair['decision_frame'].values[0]
        lcv = pair[['frame', 'id_x', 'y_x', 'x_x', 'width_x', 'height_x', 'laneId_x']].iloc[20:]
        fv = pair[['frame', 'id_y', 'y_y', 'x_y', 'width_y', 'height_y', 'laneId_y']].iloc[20:]
        nlv = pair[['frame', 'id', 'y', 'x', 'width', 'height', 'laneId']].iloc[20:]
        olv = pair[['frame', 'id_z', 'y_z', 'x_z', 'width_z', 'height_z', 'laneId_z']].iloc[20:]

        lcv = trajectory_process(lcv, decision_frame)
        fv = trajectory_process(fv, decision_frame)
        nlv = trajectory_process(nlv, decision_frame)
        olv = trajectory_process(olv, decision_frame)

        if lcv.isnull().values.any() or fv.isnull().values.any() or nlv.isnull().values.any() or olv.isnull().values.any():
            continue

        record_trajectory['lcv'].append(lcv)
        record_trajectory['fv'].append(fv)
        record_trajectory['nlv'].append(nlv)
        record_trajectory['olv'].append(olv)

    return record_trajectory


def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class AIRLDiscrim(nn.Module):
    def __init__(self, state_shape, gamma,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.g = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)

        return rs + self.gamma * (1 - dones.unsqueeze(-1)) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        # return self.f(states, dones, next_states) - log_pis.unsqueeze(-1)
        exp_f = torch.exp(self.f(states, dones, next_states))
        return (exp_f / (exp_f + torch.exp(log_pis.unsqueeze(-1))))

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)

            return (torch.log(logits + 1e-3) - torch.log((1-logits)+1e-3))#-F.logsigmoid(-logits)


