import torch
from torch.optim import Adam
from torch import nn
from ppo_model.PPO import PPO, RolloutBuffer
from NGSIM_env.envs.env import NGSIMEnv
import numpy as np
import warnings

warnings.filterwarnings('ignore')


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

            return (torch.log(logits + 1e-3) - torch.log((1-logits)+1e-3))


def run():
    env = NGSIMEnv(scene='us-101')
    env.generate_experts()

    buffer_exp = RolloutBuffer()
    buffer_exp.add_exp(path='expert_data.pkl')

    model_lcv = PPO(state_dim=16, action_dim=2, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2,
                    has_continuous_action_space=True, action_std_init=0.2)
    model_fv = PPO(state_dim=16, action_dim=2, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2,
                   has_continuous_action_space=True, action_std_init=0.2)

    # Discriminator
    disc_lcv = AIRLDiscrim(state_shape=16,
                           gamma=0.99,
                           hidden_units_r=(64, 64),
                           hidden_units_v=(64, 64),
                           hidden_activation_r=nn.ReLU(inplace=True),
                           hidden_activation_v=nn.ReLU(inplace=True)).to('cuda')
    disc_fv = AIRLDiscrim(state_shape=16,
                          gamma=0.99,
                          hidden_units_r=(64, 64),
                          hidden_units_v=(64, 64),
                          hidden_activation_r=nn.ReLU(inplace=True),
                          hidden_activation_v=nn.ReLU(inplace=True)).to('cuda')

    optim_disc = Adam([{'params': disc_lcv.parameters()},
                       {'params': disc_fv.parameters()}], lr=3e-4)
    disc_criterion = nn.BCELoss()
    # printing and logging variables
    time_step = 0
    i_episode = 0
    scene_id = 0
    # expert_dict = {'state': [], 'action': [], 'rewards': [], 'dones': [], 'next_states': []}

    # training loop
    while i_episode <= 110000:
        state = env.reset(scene_id=scene_id)
        current_ep_reward = 0

        while True:
            # select action with policy
            action_lcv = model_lcv.select_action(state) * 5
            action_fv = model_fv.select_action(state) * 5
            action = np.concatenate([action_lcv, action_fv])
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            model_lcv.buffer.rewards.append(reward)
            model_lcv.buffer.is_terminals.append(done)
            model_lcv.buffer.next_states.append(torch.from_numpy(state.reshape(1, -1)).to('cuda'))

            model_fv.buffer.rewards.append(reward)
            model_fv.buffer.is_terminals.append(done)
            model_fv.buffer.next_states.append(torch.from_numpy(state.reshape(1, -1)).to('cuda'))

            time_step += 1
            current_ep_reward += reward

            # if continuous action space; then decay action std of ouput action distribution
            if time_step % int(1e5) == 0:
                model_lcv.decay_action_std(0.05, 0.1)
                model_fv.decay_action_std(0.05, 0.1)

            if done:
                break

        # IRL updating
        if i_episode % 150 == 0:
            epoch_disc_loss = []
            acc_exp_lcv = []
            acc_exp_fv = []
            acc_pi_lcv = []
            acc_pi_fv = []
            batch_size = 64
            for _ in range(10):
                # Samples from current policy's trajectories.
                states_lcv, actions_lcv, _, dones_lcv, log_pis_lcv, next_states_lcv = model_lcv.buffer.sample(batch_size)
                states_fv, actions_fv, _, dones_fv, log_pis_fv, next_states_fv = model_fv.buffer.sample(batch_size)
                # Samples from expert's demonstrations.
                states_exp, actions_exp, _, dones_exp, next_states_exp = buffer_exp.sample(batch_size, if_exp=True)
                states_exp = states_exp.float()
                actions_exp = actions_exp.float() / 5
                next_states_exp = next_states_exp.float()
                states_lcv = states_lcv.float()
                states_fv = states_fv.float()
                next_states_lcv = next_states_lcv.float()
                next_states_fv = next_states_fv.float()
                # Calculate log probabilities of expert actions.
                with torch.no_grad():
                    log_pis_exp_lcv, _, _ = model_lcv.policy.evaluate(states_exp, actions_exp[:, :2])
                    log_pis_exp_fv, _, _ = model_fv.policy.evaluate(states_exp, actions_exp[:, 2:])

                # Update discriminator.
                dones_lcv = dones_lcv.int().to('cuda')
                dones_fv = dones_fv.int().to('cuda')
                dones_exp = dones_exp.int().to('cuda')

                prob_pi_lcv = disc_lcv(states_lcv, dones_lcv, log_pis_lcv, next_states_lcv)
                prob_exp_lcv = disc_lcv(states_exp, dones_exp.squeeze(-1), log_pis_exp_lcv, next_states_exp)

                prob_pi_fv = disc_fv(states_fv, dones_fv, log_pis_fv, next_states_fv)
                prob_exp_fv = disc_fv(states_exp, dones_exp.squeeze(-1), log_pis_exp_fv, next_states_exp)

                # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
                expert_loss = disc_criterion(prob_exp_lcv, torch.ones(prob_exp_lcv.shape[0], 1).to('cuda')) + \
                              disc_criterion(prob_exp_fv, torch.ones(prob_exp_fv.shape[0], 1).to('cuda'))
                agent_loss = disc_criterion(prob_pi_lcv, torch.zeros(prob_pi_lcv.shape[0], 1).to('cuda')) + \
                             disc_criterion(prob_pi_fv, torch.zeros(prob_pi_fv.shape[0], 1).to('cuda'))
                loss_disc = (expert_loss + agent_loss)

                expert_acc_lcv = ((prob_exp_lcv > 0.5).float()).mean().detach().cpu()
                expert_acc_fv = ((prob_exp_fv > 0.5).float()).mean().detach().cpu()
                learner_acc_lcv = ((prob_pi_lcv < 0.5).float()).mean().detach().cpu()
                learner_acc_fv = ((prob_pi_fv < 0.5).float()).mean().detach().cpu()

                if expert_acc_lcv > 0.8 and learner_acc_lcv > 0.8 and expert_acc_fv > 0.8 and learner_acc_fv > 0.8:
                    break
                optim_disc.zero_grad()
                loss_disc.backward()
                optim_disc.step()
                epoch_disc_loss.append(loss_disc.item())
                acc_exp_lcv.append(expert_acc_lcv)
                acc_exp_fv.append(expert_acc_fv)
                acc_pi_lcv.append(learner_acc_lcv)
                acc_pi_fv.append(learner_acc_fv)

            # We don't use reward signals here,
            states_lcv, actions_lcv, dones_lcv, log_pis_lcv, next_states_lcv = model_lcv.buffer.get()
            states_fv, actions_fv, dones_fv, log_pis_fv, next_states_fv = model_fv.buffer.get()
            # torch.save(model_lcv.buffer, 'lcv_buffer.pt')
            # torch.save(model_fv.buffer, 'fv_buffer.pt')
            next_states_lcv = next_states_lcv.float()
            next_states_fv = next_states_fv.float()
            dones_lcv = dones_lcv.int().to('cuda')
            dones_fv = dones_fv.int().to('cuda')

            rewards_lcv = disc_lcv.calculate_reward(states_lcv, dones_lcv, log_pis_lcv, next_states_lcv)
            rewards_fv = disc_fv.calculate_reward(states_fv, dones_fv, log_pis_fv, next_states_fv)
            disc_reward_lcv = rewards_lcv.mean()
            disc_reward_fv = rewards_fv.mean()
            model_lcv.buffer.rewards = []
            model_fv.buffer.rewards = []
            for sub_r in rewards_lcv:
                model_lcv.buffer.rewards.append(sub_r)
            for sub_r in rewards_fv:
                model_fv.buffer.rewards.append(sub_r)
            print('Epoch disc loss {:.4}, acc exp lcv {:.4}, acc pi lcv {:.4},'
                  'acc exp fv {:.4}, acc pi fv {:.4}'.format(np.mean(epoch_disc_loss),
                                                             np.mean(acc_exp_lcv), np.mean(acc_pi_lcv),
                                                             np.mean(acc_exp_fv), np.mean(acc_pi_fv)))

            # Update PPO
            model_lcv.update()
            model_fv.update()
            print('Episode {}, reward {:.4}, disc r lcv {:.4}, disc r fv {:.4}'.format(i_episode, current_ep_reward,
                                                                                       disc_reward_lcv, disc_reward_fv))

        i_episode += 1
        scene_id += 1

        if scene_id > 150:
            scene_id = 0
            # torch.save(disc_lcv, 'disc_lcv.pt')
            # torch.save(disc_fv, 'disc_fv.pt')
            # torch.save(model_lcv, 'model_lcv.pt')
            # torch.save(model_fv, 'model_fv.pt')

    env.close()


if __name__ == '__main__':
    run()









