import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from ppo_model.PPO import PPO
# from gail_airl_ppo.network import AIRLDiscrim


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
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)


class AIRL(PPO):
    def __init__(self, env, buffer_exp, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, device, action_std_init=0.6, lr_disc=3e-4):
        super().__init__(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                         has_continuous_action_space, action_std_init=action_std_init)
        self.env = env
        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = AIRLDiscrim(
            state_shape=state_dim,
            gamma=gamma,
            hidden_units_r=(64, 64),
            hidden_units_v=(64, 64),
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)
        self.device = device
        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = 64
        self.epoch_disc = 10

    def train(self):
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset(scene_id=0)

        for step in range(1, 100):
            # Pass to the algorithm to update state and episode timestep.
            action = self.select_action(state) * 5
            next_state, reward, done, info = self.env.step(action)

            self.buffer.rewards.append(reward)
            self.buffer.is_terminals.append(done)
            self.buffer.next_states.append(torch.from_numpy(next_state.reshape(1, -1)).to(self.device))

            # Update the algorithm whenever ready.
            if step > 20:
                self.update()
            state = next_state
            # # Evaluate regularly.
            # if step % self.eval_interval == 0:
            #     self.evaluate(step)

    def update(self):
        epoch_disc_loss = []
        for _ in range(self.epoch_disc):
            # Samples from current policy's trajectories.
            states, _, _, dones, log_pis, next_states = self.buffer.sample(self.batch_size)
            # Samples from expert's demonstrations.
            states_exp, actions_exp, _, dones_exp, next_states_exp = self.buffer_exp.sample(self.batch_size)
            states_exp = states_exp.float()
            actions_exp = actions_exp.float()
            next_states_exp = next_states_exp.float()
            states = states.float()
            next_states = next_states.float()
            # Calculate log probabilities of expert actions.
            with torch.no_grad():
                log_pis_exp, _, _ = self.policy.evaluate(states_exp, actions_exp)
            # Update discriminator.
            sub_loss = self.update_disc(
                                        states, dones, log_pis, next_states, states_exp,
                                        dones_exp, log_pis_exp, next_states_exp
                                    )
            epoch_disc_loss.append(sub_loss)

        print('Epoch disc loss {}'.format(np.mean(epoch_disc_loss)))
        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()
        dones = dones.int().to(self.device)
        # Calculate rewards.
        rewards = self.disc.calculate_reward(states, dones, log_pis, next_states)

        # Update PPO using estimated rewards.
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states)

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp):
        # Output of discriminator is (-inf, inf), not [0, 1].
        dones = dones.int().to(self.device)
        logits_pi = self.disc(states, dones, log_pis, next_states)
        dones_exp = dones_exp.int().to(self.device)
        logits_exp = self.disc(states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        # if self.learning_steps_disc % self.epoch_disc == 0:
        #     print('loss disc {}'.format(loss_disc.item()))

        return loss_disc.item()