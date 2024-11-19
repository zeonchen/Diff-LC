import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, node_feat=False):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        self.node_feat = node_feat

    def add_noise(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0).float()
        x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        return x_t, t

    def forward(self, node_feat, node_loc, node_v=None, context=None, node_mask=None):
        """
        Algorithm 1.
        """
        x_0 = node_loc
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0).float()

        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        pred_noise = self.model(t, x_t, node_v, context, node_mask)
        loss = F.mse_loss(pred_noise, noise, reduction='none')

        return loss.mean()


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


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        self.node_feat = False

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        # x_t = x_t.permute(0, 2, 1, 3)
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, context, guide=None, y=None, origin_x=None, state=None):
        # below: only log_variance is used in the KL computations
        model_log_var = torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]]))
        model_log_var = extract(model_log_var, t, x_t.shape)

        eps = self.model(t, x_t, origin_x, context, None)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        if guide:
            gradient = self.guide_fn(xt_prev_mean, t, guide, state)
            new_mean = (
                    xt_prev_mean.float() + torch.exp(model_log_var) * gradient.float()
            )

            return new_mean, model_log_var

        else:
            return xt_prev_mean, model_log_var

    def guide_fn(self, all_x, t, guide, state, guide_scale=10.0):
        with torch.enable_grad():
            guide.train()
            n_fv = state[:, :, :4] - state[:, :, 4:8]
            n_nlv = state[:, :, :4] + state[:, :, 8:12]
            n_olv = state[:, :, :4] + state[:, :, 12:]
            new_n_lcv = normalize_obs(all_x.detach().cpu().numpy())
            new_n_lcv = torch.from_numpy(new_n_lcv).to(all_x.device).float()
            dfv = new_n_lcv - n_fv
            dnlv = n_nlv - new_n_lcv
            dolv = n_olv - new_n_lcv

            state = torch.concatenate([new_n_lcv, dfv, dnlv, dolv], dim=-1).requires_grad_(True)

            pred_reward_loss = torch.abs(guide(state, t) - 0.5)
            grad = torch.autograd.grad(outputs=pred_reward_loss.sum(), inputs=state)[0].mean() * guide_scale

            return grad

    def forward(self, node_loc, context, state=None, guide=None):
        """
        Algorithm 2.
        """
        origin_x = node_loc
        x_0 = node_loc
        x_t = torch.randn_like(x_0)

        all_x = []
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t, origin_x=origin_x,
                                                 context=context, state=state,
                                                 guide=guide)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise

            all_x.append(x_t.unsqueeze(1))
            if bool(torch.isnan(x_t).sum()) > 0:
                break

        return x_t, torch.cat(all_x, dim=1)

