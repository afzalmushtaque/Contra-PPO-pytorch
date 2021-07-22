"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import MultipleEnvironments
from src.model import PPO
from src.process import test
import torch.multiprocessing as _mp
from torch.distributions import Categorical
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
# import shutil
# from tensorboardX import GlobalSummaryWriter
import wandb
from sklearn.metrics import explained_variance_score
import logging



def get_args():
    parser = argparse.ArgumentParser("""Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Contra Nes""")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-6, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--tau', type=float, default=0.99, help='Variance parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--vf_loss_coeff', type=float, default=1.0, help='Critic loss coefficient')
    parser.add_argument('--actor_loss_coeff', type=float, default=1.0, help='Actor loss coefficient')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Parameter for Clipped Surrogate Objective')
    parser.add_argument('--num_batches', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=200)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--num_processes", type=int, default=6)
    parser.add_argument("--save_interval", type=int, default=1, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_contra")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False, help="Load weight from previous trained stage")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
        raise Exception('Cuda not working')
    # if os.path.isdir(opt.log_path):
    #     shutil.rmtree(opt.log_path)
    # os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    envs = MultipleEnvironments(opt.level, opt.num_processes, opt.max_episode_steps)
    model = PPO(envs.num_states, envs.num_actions)
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()

    process = mp.Process(target=test, args=(opt, model, envs.num_states, envs.num_actions, opt.max_episode_steps))
    process.start()
    wandb.watch(model, log_freq=10, log_graph=True, log="parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.stack(curr_states)).permute(0, 3, 1, 2)
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()
    # writer.add_graph(model, input_to_model=curr_states)
    
    curr_episode = 0
    while True:
        if curr_episode % opt.save_interval == 0 and curr_episode > 0:
            torch.save(model.state_dict(),
                       "{}/ppo_contra_level{}".format(opt.saved_path, opt.level))
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        total_rewards = []
        episode_lengths = []
        policies = []
        for _ in range(opt.num_local_steps):
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            policies.append(policy.detach().cpu().numpy())
            old_m = Categorical(policy)
            action = old_m.sample()
            # action = torch.randint(low=0, high=envs.num_actions, size=(opt.num_processes,)).cuda()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            if torch.cuda.is_available():
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            state = list(state)
            info = list(info)
            for i in range(opt.num_processes):
                if done[i]:
                    total_rewards.append(info[i]['total_reward'])
                    episode_lengths.append(info[i]['episode_length'])
                    envs.agent_conns[i].send(("reset", None)) 
                    state[i] = envs.agent_conns[i].recv()
                    print('Worker {0:.0f} has completed an episode with length {1:,.0f} and total reward {2:,.2f}.'.format(i, info[i]['episode_length'], info[i]['total_reward']))
                    logging.info('Worker {0:.0f} has completed an episode with length {1:,.0f} and total reward {2:,.2f}.'.format(i, info[i]['episode_length'], info[i]['total_reward']))

            state = torch.from_numpy(np.stack(state)).permute(0, 3, 1, 2)
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
            rewards.append(reward)
            dones.append(done)
            curr_states = state

        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        scalar_rewards = torch.stack(rewards).detach().cpu().numpy()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        gae = 0
        advantages = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            # R.append(gae + value)
            advantages.append(gae)
        advantages = advantages[::-1]
        advantages = torch.cat(advantages).detach()
        R = advantages + values
        actor_losses = []
        entropy_losses = []
        for i in range(opt.num_epochs):
            print('Epoch {0:,.0f}'.format(i+1))
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)
            for j in range(opt.num_batches):
                print('Processing batch number {0:,.0f}'.format(j+1))
                batch_indices = indice[
                                int(j * (opt.num_local_steps * opt.num_processes / opt.num_batches)): int((j + 1) * (
                                        opt.num_local_steps * opt.num_processes / opt.num_batches))]
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                raw_actor_loss = torch.min(
                    ratio * advantages[batch_indices],
                    torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) * advantages[batch_indices]
                )
                actor_loss = -torch.mean(raw_actor_loss)
                actor_losses.extend(raw_actor_loss.detach().cpu().numpy())
                # critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                critic_loss = F.mse_loss(R[batch_indices], value.squeeze())

                entropy_loss = torch.mean(new_m.entropy())
                entropy_losses.extend(new_m.entropy().detach().cpu().numpy())
                total_loss = actor_loss * opt.actor_loss_coeff + \
                                critic_loss * opt.vf_loss_coeff - \
                                entropy_loss * opt.beta
                optimizer.zero_grad()
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        episode_actor_loss = -np.mean(actor_losses) * opt.actor_loss_coeff
        episode_critic_loss = F.mse_loss(R, values) * opt.vf_loss_coeff
        episode_entropy_loss = np.mean(entropy_losses) * opt.beta 
        episode_total_loss = episode_actor_loss + episode_critic_loss + episode_entropy_loss

        episode_policies = np.array(policies).reshape(-1, envs.num_actions)

        print("Episode: {}. Total loss: {}".format(curr_episode, episode_total_loss))
        wandb.log(
            {
                'custom/loss/total': episode_total_loss,
                'custom/loss/actor': episode_actor_loss,
                'custom/loss/critic': episode_critic_loss,
                'custom/loss/entropy': episode_entropy_loss,
                'custom/lengths/mean': np.mean(episode_lengths) if len(episode_lengths) > 0 else np.NaN,
                'custom/lengths/std': np.std(episode_lengths) if len(episode_lengths) > 0 else np.NaN,
                'custom/lengths/min': np.min(episode_lengths) if len(episode_lengths) > 0 else np.NaN,
                'custom/lengths/max': np.max(episode_lengths) if len(episode_lengths) > 0 else np.NaN,
                'custom/lengths/aggregated': wandb.Histogram(episode_lengths),
                'custom/rewards/mean': np.mean(total_rewards) if len(total_rewards) > 0 else np.NaN,
                'custom/rewards/std': np.std(total_rewards) if len(total_rewards) > 0 else np.NaN,
                'custom/rewards/max': np.max(total_rewards) if len(total_rewards) > 0 else np.NaN,
                'custom/rewards/min': np.min(total_rewards) if len(total_rewards) > 0 else np.NaN,
                'custom/rewards/aggregated': wandb.Histogram(total_rewards),
                'custom/rewards/raw': wandb.Histogram(scalar_rewards.reshape(-1, )),
                'custom/values/predicted': wandb.Histogram(values.cpu()),
                'custom/values/target': wandb.Histogram(R.cpu()),
                'custom/values/exp_var': explained_variance_score(R.cpu().numpy(), values.cpu().numpy()),
                'custom/model/critic/bias': model.state_dict()['critic_linear.2.bias'][0],
                'custom/policy': wandb.Histogram(episode_policies),
            }
        )


if __name__ == "__main__":
    opt = get_args()
    wandb.init(sync_tensorboard=True)
    wandb.config.update(opt)
    train(opt)
    
