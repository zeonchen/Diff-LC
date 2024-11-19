import argparse
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import NoisyReward
from model.ddpm import GaussianDiffusionTrainer
import copy
from tqdm import tqdm
from model.related_module import *
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


def eval(args, loader, trainer, model, fv_reward):
    model.eval()
    device = args.device
    loss_list = []
    for batch in loader:
        lcv_traj = batch[0].cpu().numpy()
        fv_traj = batch[1].cpu().numpy()
        nlv_traj = batch[2].cpu().numpy()
        olv_traj = batch[3].cpu().numpy()

        # Generate true reward
        n_lcv = normalize_obs(copy.deepcopy(lcv_traj))
        n_fv = normalize_obs(copy.deepcopy(fv_traj))
        n_nlv = normalize_obs(copy.deepcopy(nlv_traj))
        n_olv = normalize_obs(copy.deepcopy(olv_traj))

        dfv = n_lcv - n_fv
        dnlv = n_nlv - n_lcv
        dolv = n_olv - n_lcv

        state = torch.from_numpy(np.concatenate([n_lcv, dfv, dnlv, dolv], axis=-1)).to(device).float()
        true_reward = fv_reward(state)

        # Noisy data
        noisy_fv, t = trainer(torch.from_numpy(n_fv).to(device))
        noisy_dfv = n_lcv - noisy_fv.cpu().numpy()
        noisy_state = torch.from_numpy(np.concatenate([n_lcv, noisy_dfv, dnlv, dolv], axis=-1)).to(device).float()
        pred_reward = model(noisy_state, t)

        loss = F.mse_loss(pred_reward, true_reward)
        loss_list.append(loss.item())

    model.train()

    return np.mean(loss_list)


def train(args):
    device = args.device

    model = NoisyReward(args.context_dim, args.T)
    ema_model = copy.deepcopy(model)
    guide = None

    # show model size
    model_size = 0
    for param in ema_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    trainer = GaussianDiffusionTrainer(model, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # dataset
    path = ''
    val_path = ''

    train_loader = data_loader(args.obs_len, args.pred_len, args.skip, args.batch_size, path,
                               save_path='data/sample_data.pkl', if_test='train')
    test_loader = data_loader(args.obs_len, args.pred_len_val, args.skip, args.batch_size * 4, val_path,
                              shuffle=False, save_path='data/sample_data.pkl', if_test='test')
    fv_reward = torch.load('disc_fv.pt').g
    best_ade = 1e5

    for epoch in range(args.total_epoch):
        epoch_loss = []

        for batch in train_loader:
            lcv_traj = batch[0].cpu().numpy()
            fv_traj = batch[1].cpu().numpy()
            nlv_traj = batch[2].cpu().numpy()
            olv_traj = batch[3].cpu().numpy()

            optim.zero_grad()
            cur_lr = optim.state_dict()['param_groups'][0]['lr']

            # Generate true reward
            n_lcv = normalize_obs(copy.deepcopy(lcv_traj))
            n_fv = normalize_obs(copy.deepcopy(fv_traj))
            n_nlv = normalize_obs(copy.deepcopy(nlv_traj))
            n_olv = normalize_obs(copy.deepcopy(olv_traj))

            dfv = n_lcv - n_fv
            dnlv = n_nlv - n_lcv
            dolv = n_olv - n_lcv

            state = torch.from_numpy(np.concatenate([n_lcv, dfv, dnlv, dolv], axis=-1)).to(device).float()
            true_reward = fv_reward(state)

            # Noisy data
            noisy_fv, t = trainer(torch.from_numpy(n_fv).to(device))
            noisy_dfv = n_lcv - noisy_fv.cpu().numpy()
            noisy_state = torch.from_numpy(np.concatenate([n_lcv, noisy_dfv, dnlv, dolv], axis=-1)).to(device).float()
            pred_reward = model(noisy_state, t)

            loss = F.mse_loss(pred_reward, true_reward)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            epoch_loss.append(loss.item())

            # break

        if epoch % args.print_step == 0:
            print('Epoch {}, loss {:.6f}, lr {}'.format(epoch, np.mean(epoch_loss), cur_lr))

        if epoch % args.sample_step == 0:
            val_ade = eval(args, test_loader, trainer, model, fv_reward)

            if val_ade < best_ade:
                best_ade = val_ade
                print('Best model updated, testing ...')
                eval(args, test_loader, trainer, model, fv_reward)

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
    parser.add_argument('--total_epoch', type=int, default=100000)
    parser.add_argument('--warmup', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--sample_step', type=int, default=10)
    parser.add_argument('--print_step', type=int, default=5)

    args = parser.parse_args()
    main(args)












