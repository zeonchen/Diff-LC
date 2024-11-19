from collections import namedtuple
# import numpy as np
import torch
import einops
import pdb

from utils.arrays import *
# from diffusion.datasets.preprocessing import get_policy_preprocess_fn

Trajectories = namedtuple('Trajectories', 'actions observations')
# GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations value')


class Policy:
    def __init__(self, diffusion_model, normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = 2

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        # conditions = apply_dict(
        #     self.normalizer.normalize,
        #     conditions,
        #     'observations',
        # )
        conditions = to_torch(conditions, dtype=torch.float32, device='cuda:0')
        # conditions = apply_dict(
        #     einops.repeat,
        #     conditions,
        #     'd -> repeat d', repeat=batch_size,
        # )
        return conditions

    def __call__(self, conditions, measurement_cond_fn=None, debug=False, batch_size=1):
        conditions = self._format_conditions(conditions, batch_size)

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        sample = self.diffusion_model(conditions, measurement_cond_fn)
        sample = to_np(sample.trajectories)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        # actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        observations = sample[:, :, self.action_dim:]
        # observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations)
        return action, trajectories
