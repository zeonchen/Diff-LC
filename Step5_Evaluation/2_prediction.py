import torch
import matplotlib.pyplot as plt
from model.ddpm import GaussianDiffusionSampler
import copy
from tqdm import tqdm
from utils.ngsim_dataset import data_loader, TrajectoryDataset
import pickle
import warnings
warnings.filterwarnings('ignore')


device = 'cuda'
model = torch.load('trained_model/predictor.pth')
ema_model = copy.deepcopy(model)
guide = torch.load('trained_model/guide.pth').to(device)
model.eval()
ema_sampler = GaussianDiffusionSampler(ema_model, beta_1=1e-4, beta_T=0.05, T=100).to(device)

test_dataset = TrajectoryDataset('',
                                 obs_len=20,
                                 pred_len=100,
                                 skip=5, save_dir='data/sample_data.pkl', if_test='test')
# Prediction
file = open('plan_data.pkl', 'rb')
plan_data = pickle.load(file)

for traj_id in tqdm(range(53)):
    traj_id = 34
    plan_data[traj_id]['predicted'] = []
    hist_traj = test_dataset.hist_traj[traj_id:traj_id+1].to(device)
    hist_init = copy.deepcopy(hist_traj[:, 0:1, :2]).to(device)
    state_data = test_dataset.state_data[traj_id:traj_id+1].to(device)
    cond_x_init = plan_data[traj_id]['true_x'][0]
    cond_y_init = plan_data[traj_id]['true_y'][0]

    with torch.no_grad():
        for p in range(len(plan_data[traj_id]['planned'])):
            cond_traj = plan_data[traj_id]['planned'][p].unsqueeze(0).float()
            cond_traj[:, :, :2] -= hist_init.cpu()

            diff_pred_traj = torch.randn(1, 100, 4).to(device)
            hist_traj[:, :, :2] -= hist_init

            hist_traj = hist_traj.to(device)
            cond_traj = cond_traj.to(device)
            context = model.context(hist_traj, cond_traj)

            # Sampling
            pred_diff, _ = ema_sampler(node_loc=diff_pred_traj,
                                       context=context, state=state_data, guide=guide)
            hist_traj[:, :, :2] += hist_init
            preds = torch.cat([hist_traj[:, -1:, :2], pred_diff[:, :, :2]], dim=1)
            preds = torch.cumsum(preds, dim=1)[:, 1:, :]
            velocity = pred_diff[:, :, 2:]

            plan_data[traj_id]['predicted'].append(torch.cat([preds, velocity], dim=-1))
            # break

            # av_traj = (cond_traj[:, :, :2] + hist_init).detach().cpu()
            # hv_traj = preds.detach().cpu()
            # plt.plot(hv_traj[0, :, 1], hv_traj[0, :, 0], color='orange')
            # plt.plot(av_traj[0, :, 1], av_traj[0, :, 0], color='blue')
            # plt.axhline(3.6576, linestyle='--', color='red')
            # plt.axhline(7.3152, linestyle='--', color='red')
            # plt.axhline(10.9728, linestyle='--', color='red')
            # plt.axhline(14.6304, linestyle='--', color='red')
            # plt.axhline(18.288, linestyle='--', color='red')
            # plt.axhline(21.9456, linestyle='--', color='red')
            # plt.show()
            # pass

with open('pred_data.pkl', 'wb') as path:
    pickle.dump(plan_data, path)



















