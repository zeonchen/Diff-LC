import torch

from model.temporal import TemporalUnet
from model.diffusion import GaussianDiffusion
from model.policies import Policy
from utils.dataset import SequenceDataset
from utils.arrays import *
import warnings
from tqdm import tqdm
from itertools import product
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


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


def simulation(cond, planned_traj):
    start_vx = cond[0].detach().cpu()[0, 2]
    start_vy = cond[0].detach().cpu()[0, 3]
    plan_ax = planned_traj[0][0, :, 0]
    plan_ay = planned_traj[0][0, :, 1]
    plan_vx = [start_vx]
    plan_vy = [start_vy]

    for i in range(99):
        plan_vx.append(plan_vx[-1] + plan_ax[i].item() * 0.1)
        plan_vy.append(plan_vy[-1] + plan_ay[i].item() * 0.1)

    start_x = cond[0].detach().cpu()[0, 1]
    start_y = cond[0].detach().cpu()[0, 0]
    plan_x = [start_x]
    plan_y = [start_y]

    for i in range(99):
        plan_x.append(plan_x[-1] + plan_vx[i].item() * 0.1)
        plan_y.append(plan_y[-1] + plan_vy[i].item() * 0.1)

    plan_x = torch.tensor(plan_x[:100]).unsqueeze(-1)
    plan_y = torch.tensor(plan_y[:100]).unsqueeze(-1)
    plan_vx = torch.tensor(plan_vx[:100]).unsqueeze(-1)
    plan_vy = torch.tensor(plan_vy[:100]).unsqueeze(-1)
    plan_ax = torch.from_numpy(plan_ax[:100]).unsqueeze(-1)
    plan_ay = torch.from_numpy(plan_ay[:100]).unsqueeze(-1)

    return torch.cat([plan_y, plan_x, plan_vx, plan_vy, plan_ax, plan_ay], dim=-1)


device = 'cuda'
model = TemporalUnet(horizon=100,
                     transition_dim=4+2,
                     dim=64,
                     cond_dim=4,
                     dim_mults=(1, 2, 4, 8),
                     attention=False).to(device)

diffusion = GaussianDiffusion(model,
                              horizon=100,
                              observation_dim=4,
                              action_dim=2,
                              n_timesteps=20,
                              loss_type='l2',
                              clip_denoised=False,
                              predict_epsilon=False,
                              ## loss weighting
                              action_weight=10,
                              loss_weights=None,
                              loss_discount=1).to(device)

path = 'trained_model/planner.pt'
data_path = 'data/sample_data.pkl'
goal_path = 'trained_model/goal_model.pth'
dataset = SequenceDataset(data_path, horizon=100, if_test=True)
diffusion_parm = torch.load(path)
diffusion.load_state_dict(diffusion_parm['ema'])
policy = Policy(diffusion, None)
goal_predictor = torch.load(goal_path).to(device)

all_data = {}

for traj_id in tqdm(range(53)):
    all_data[traj_id] = {}

    # set targets
    bound_max = torch.tensor([168.4894,  12.1933], device=device)
    bound_min = torch.tensor([31.9857, -7.8002], device=device)
    batch = dataset[traj_id]
    true_x = batch[0][0, :, 3].detach().cpu().unsqueeze(-1)
    true_y = batch[0][0, :, 2].detach().cpu().unsqueeze(-1)
    goal_state = goal_predictor(batch[2][0].to(device), batch[2][1].to(device),
                                batch[2][2].to(device), batch[2][3].to(device))
    goal_state = goal_state * (bound_max - bound_min) + bound_min
    goal_x = goal_state[0].detach().cpu()
    goal_v = goal_state[1].detach().cpu()

    # Generate conditions
    lane_center = [(3.6576+7.3152)/2, (7.3152+10.9728)/2, (10.9728+14.6304)/2, (14.6304+18.288)/2, (18.288+21.9456)/2]
    cond = batch[1]
    noise_x = goal_x
    noise_vx = goal_v + cond[0][:, 2]

    potent_y = [lane_center[np.abs(np.subtract.outer(lane_center, true_y[-1])).argmin(0)[0]] - true_y[0]]
    potent_x = [noise_x - 2*i for i in range(1, 10)] + [noise_x + 2*i for i in range(10)]
    potent_vy = [0]
    potent_vx = [noise_vx]

    potent_cond = list(product(list(product(potent_y, potent_x)), list(product(potent_vx, potent_vy))))

    cond[0] = cond[0].to(device)
    cond[99] = cond[99].to(device)
    cond[0][:, 0] -= true_y[0, :].to(device)
    cond[0][:, 1] -= true_x[0, :].to(device)
    cond[99][:, 0] -= true_y[0, :].to(device)
    cond[99][:, 1] -= true_x[0, :].to(device)

    all_data[traj_id]['true_x'] = true_x
    all_data[traj_id]['true_y'] = true_y
    all_data[traj_id]['planned'] = []

    for final_pos in potent_cond:
        cond[99] = torch.tensor(list(final_pos[0] + final_pos[1])).to(device).unsqueeze(0)
        _, planned_traj = policy(cond, batch_size=1)
        plan_res = simulation(cond, planned_traj)
        plan_res[:, 0] += true_y[0, :]
        plan_res[:, 1] += true_x[0, :]
        all_data[traj_id]['planned'].append(plan_res)

with open('plan_data.pkl', 'wb') as path:
    pickle.dump(all_data, path)













