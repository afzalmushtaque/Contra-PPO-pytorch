"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import argparse
import torch
from src.env import create_train_env, ACTION_MAPPING
from src.model import PPO
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Contra Nes""")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    args = parser.parse_args()
    return args


def test(opt):
    torch.manual_seed(123)
    env = create_train_env(
        opt.level, 
        max_episode_steps=opt.max_episode_steps, 
        # output_path="{}/video_{}.mp4".format(opt.output_path, opt.level) # TODO recording showing strange artifacts
    )
    # env = create_train_env(opt.level, max_episode_steps=opt.max_episode_steps)
    model = PPO(env.observation_space.shape[-1], len(ACTION_MAPPING))
    if False: # torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/ppo_contra_level{}".format(opt.saved_path, opt.level)))
        model.cuda()
    else:
        model.load_state_dict(
            torch.load(
                "{}/ppo_contra_level{}".format(opt.saved_path, opt.level),
                map_location=lambda storage, 
                loc: storage,
            )
        )
    model.eval()
    state = torch.from_numpy(env.reset()).unsqueeze(dim=0).permute(0,3,1,2)
    total_reward = 0
    done = False
    while not done:
        # if torch.cuda.is_available():
        #     state = state.cuda()
        logits, value = model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        total_reward += reward
        state = torch.from_numpy(state).unsqueeze(dim=0).permute(0,3,1,2)
        env.render()
        if info["level"] > opt.level or done:
            print("Level {} completed".format(opt.level))


if __name__ == "__main__":
    opt = get_args()
    test(opt)
