import torch
import numpy as np
import pandas as pd
import argparse
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import TransformerConcatLinear, NoisyReward
from model.ddpm import GaussianDiffusionTrainer
from model.ddpm import GaussianDiffusionSampler
import copy
import time
# from utils.dataset import data_loader
from utils.dataset import data_loader
import random
import warnings
warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def displacement_error(pred_traj, pred_traj_gt):
    select_loss = []
    pred_traj_gt = pred_traj_gt[:, :, :2]
    pred_traj = pred_traj[:, :, :2]
    batch_size, seq_len, _ = pred_traj.size()
    loss = pred_traj_gt.view(-1, seq_len, 2) - pred_traj.view(-1, seq_len, 2)
    loss = loss**2

    select_idx = [19, 39, 59, 79, 99]
    for idx in select_idx:
        temp_loss = torch.sqrt(loss[:, idx, :].sum(dim=1)).sum(dim=0) / batch_size
        select_loss.append(temp_loss)

    all_loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    all_loss = torch.sum(all_loss) / batch_size / seq_len

    return all_loss, select_loss


def ema(source, target, decay=0.9999):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def eval(args, loader, sampler, model, guide):
    model.eval()
    device = args.device

    ade_list = []
    ade_list_1s = []
    ade_list_2s = []
    ade_list_3s = []
    ade_list_4s = []
    ade_list_5s = []

    for batch in loader:
        start = time.time()
        hist_traj = batch[0].to(device)
        cond_traj = batch[1].to(device)
        pred_traj = batch[2].to(device)
        state_data = batch[3].to(device)

        hist_init = copy.deepcopy(hist_traj[:, 0:1, :2])
        with torch.no_grad():
            pre_pred_traj = torch.cat([hist_traj[:, -1:, :], pred_traj[:, :-1, :]], dim=1)
            diff_pred_traj = pred_traj - pre_pred_traj
            diff_pred_traj = diff_pred_traj[:, :, :4]
            hist_traj[:, :, :2] -= hist_init
            context = model.context(hist_traj, cond_traj)

            # Sampling
            pred_diff, _ = sampler(node_loc=diff_pred_traj,
                                   context=context, state=state_data, guide=guide)
            hist_traj[:, :, :2] += hist_init
            preds = torch.cat([hist_traj[:, -1:, :2], pred_diff[:, :, :2]], dim=1)
            preds = torch.cumsum(preds, dim=1)[:, 1:, :]

        print(time.time() - start)
        all_dist, select_dist = displacement_error(preds, pred_traj)

        ade_list.append(all_dist.item())
        ade_list_1s.append(select_dist[0].item())
        ade_list_2s.append(select_dist[1].item())
        ade_list_3s.append(select_dist[2].item())
        ade_list_4s.append(select_dist[3].item())
        ade_list_5s.append(select_dist[4].item())

        # break

    print('Evaluation dist {:.3f}, 1s {:.3f}, 2s {:.3f}, 3s {:.3f}, 4s {:.3f}, 5s {:.3f}'.format(np.mean(ade_list),
                                                                                                 np.mean(ade_list_1s),
                                                                                                 np.mean(ade_list_2s),
                                                                                                 np.mean(ade_list_3s),
                                                                                                 np.mean(ade_list_4s),
                                                                                                 np.mean(ade_list_5s)))

    model.train()

    return np.mean(ade_list)


def train(args):
    device = args.device

    model = TransformerConcatLinear(args.context_dim, args.T)
    ema_model = copy.deepcopy(model)
    guide = torch.load('utils/guide.pth').to(device)

    # show model size
    model_size = 0
    for param in ema_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    trainer = GaussianDiffusionTrainer(model, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T).to(device)
    sampler = GaussianDiffusionSampler(model, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T).to(device)
    ema_sampler = GaussianDiffusionSampler(ema_model, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # dataset
    path = ''
    val_path = ''

    train_loader = data_loader(args.obs_len, args.pred_len, args.skip, args.batch_size, path,
                               save_path='data/sample_data.pkl', if_test='train')
    test_loader = data_loader(args.obs_len, args.pred_len_val, args.skip, args.batch_size * 4, val_path,
                              shuffle=False, save_path='data/sample_data.pkl', if_test='test')

    best_ade = 1e5

    for epoch in range(args.total_epoch):
        epoch_loss = []

        for batch in train_loader:
            hist_traj = batch[0].to(device)
            cond_traj = batch[1].to(device)
            pred_traj = batch[2].to(device)
            state_data = batch[3].to(device)

            pre_pred_traj = torch.cat([hist_traj[:, -1:, :], pred_traj[:, :-1, :]], dim=1)
            diff_pred_traj = pred_traj - pre_pred_traj
            diff_pred_traj = diff_pred_traj[:, :, :4]
            hist_traj[:, :, :2] -= hist_traj[:, 0:1, :2]
            context = model.context(hist_traj, cond_traj)
            optim.zero_grad()
            cur_lr = optim.state_dict()['param_groups'][0]['lr']

            loss = trainer(node_feat=None, node_loc=diff_pred_traj, node_v=None,
                           context=context)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            ema(model, ema_model, decay=args.ema_decay)

            epoch_loss.append(loss.item())

            # break

        if epoch % args.print_step == 0:
            print('Epoch {}, loss {:.6f}, lr {}'.format(epoch, np.mean(epoch_loss), cur_lr))

        if epoch % args.sample_step == 0:
            val_ade = eval(args, test_loader, ema_sampler, ema_model, guide)

            if val_ade < best_ade:
                best_ade = val_ade
                print('Best model updated, testing ...')
                eval(args, test_loader, ema_sampler, ema_model, guide)

                torch.save(ema_model, 'best_model.pth')


def main(args):
    train(args)


if __name__ == "__main__":
    setup_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')

    # Dataset
    parser.add_argument('--obs_len', type=int, default=20)
    parser.add_argument('--pred_len', type=int, default=100)
    parser.add_argument('--pred_len_val', type=int, default=100)
    parser.add_argument('--skip', type=int, default=5)

    # Backbone
    parser.add_argument('--node_feat_dim', type=int, default=0)
    parser.add_argument('--time_dim', type=int, default=64)
    parser.add_argument('--context_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)

    # Diffusion
    parser.add_argument('--beta_1', type=int, default=1e-4)
    parser.add_argument('--beta_T', type=int, default=0.05)
    parser.add_argument('--T', type=int, default=100)

    # Training
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--total_epoch', type=int, default=5000)
    parser.add_argument('--warmup', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--sample_step', type=int, default=10)
    parser.add_argument('--print_step', type=int, default=5)

    args = parser.parse_args()
    main(args)












