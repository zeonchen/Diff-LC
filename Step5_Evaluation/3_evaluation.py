import copy
import time

from model.related_module import *
import pickle
import warnings
warnings.filterwarnings('ignore')

# Evaluation Module
disc_lcv = torch.load('trained_model/disc_lcv.pt')
reward_lcv = disc_lcv.g
device = 'cuda'

# load data
pred_data = pickle.load(open('pred_data.pkl', 'rb'))
plan_data = pickle.load(open('plan_data.pkl', 'rb'))
trajectory_set = build_trajecotry()
best_res = {}
all_ade = []
all_fde = []
collide_count = 0
for scene_id in range(53):
    print(scene_id)
    nlv = trajectory_set['nlv'][scene_id][['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[:100, :4]
    olv = trajectory_set['olv'][scene_id][['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[:100, :4]
    true_lcv = trajectory_set['lcv'][scene_id][['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[:100, :4]
    true_fv = trajectory_set['fv'][scene_id][['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[:100, :4]

    best_reward = -1e5
    all_reward = []
    all_rmse = []
    all_final_rmse = []
    all_lcv = []

    for p in range(len(plan_data[scene_id]['planned'])):
        lcv = plan_data[scene_id]['planned'][p][:, :4].detach().cpu().numpy()
        fv = pred_data[scene_id]['predicted'][p][0].detach().cpu().numpy()

        n_lcv = normalize_obs(copy.deepcopy(lcv))
        n_fv = normalize_obs(copy.deepcopy(fv))
        n_nlv = normalize_obs(copy.deepcopy(nlv))
        n_olv = normalize_obs(copy.deepcopy(olv))

        dfv = n_lcv - n_fv
        dnlv = n_nlv - n_lcv
        dolv = n_olv - n_lcv

        start = time.time()
        state = torch.from_numpy(np.concatenate([n_lcv, dfv, dnlv, dolv], axis=1)).to(device).float()

        reward = reward_lcv(state).mean()

        # Calculate rmse
        loss = torch.cat([torch.from_numpy(true_lcv[:, 1:2]), torch.from_numpy(true_lcv[:, 0:1])], dim=-1) - \
               torch.cat([torch.from_numpy(lcv[:, 1:2]), torch.from_numpy(lcv[:, 0:1])], dim=-1)
        loss = (loss ** 2).sum(dim=1).sqrt()
        rmse = loss.mean()
        final_rmse = loss[-1]

        all_reward.append(reward.item())
        all_rmse.append(rmse.item())
        all_final_rmse.append(final_rmse.item())
        all_lcv.append(plan_data[scene_id]['planned'][p].detach().cpu().numpy())


    _, idx = torch.topk(torch.from_numpy(np.array(all_reward)), k=3)
    min_idx = idx[np.array(all_final_rmse)[idx.numpy()].argmin()]
    best_reward = all_reward[min_idx]
    best_lcv = all_lcv[min_idx]
    best_rmse = all_rmse[min_idx]
    best_final_rmse = all_final_rmse[min_idx]
    best_res[scene_id] = [best_lcv,
                          trajectory_set['lcv'][scene_id][['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[:100],
                          best_rmse, best_final_rmse]
    all_ade.append(best_rmse)
    all_fde.append(best_final_rmse)
    print('Best reward {:.4}, best rmse {:.4}, best final {:.4}'.format(best_reward, best_rmse, best_final_rmse))

with open('final_data.pkl', 'wb') as path:
    pickle.dump(best_res, path)































