"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

# import retro
import Contra
import gym
from gym.wrappers.time_limit import TimeLimit
from ray.rllib.env.wrappers.atari_wrappers import MaxAndSkipEnv
# from Contra.customcontra import CustomContra
from gym.spaces import Box
from gym import Wrapper
import cv2
import numpy as np
import subprocess as sp
import torch.multiprocessing as mp

ACTION_MAPPING = {
    0:  0b10000000, # Right
    1:  0b10000010, # Right Fire
    2:  0b10000001, # Right Jump
    3:  0b01000000, # Left
    4:  0b01000010, # Left Fire
    5:  0b01000001, # Left Jump
    6:  0b10010010, # Right Up Fire
    7:  0b10100010, # Right Down Fire
}


class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        # self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_pos = 0
        self.curr_score = 0
        self.curr_lives = 2
        self.total_reward = None
        self.episode_length = None
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, done, info = self.env.step(ACTION_MAPPING[action])
        # reward = np.clip(reward, -1.0, 1.0)
        if self.monitor:
            self.monitor.record(state)
        state = np.float32(state) / 255.0
        # state = process_frame(state)
        # reward = min(max((info["xscroll"] - self.curr_pos - 0.01), -3), 3)
        self.curr_pos = info["xscroll"]
        # reward += min(max((info["score"] - self.curr_score), 0), 2)
        self.curr_score = info["score"]
        # if info["lives"] < self.curr_lives:
        #     reward -= 15
        #     self.curr_lives = info["lives"]
        # reward -= 0.002
        # if done:
        #     if info["lives"] != 2:
        #         reward += 0
        #     else:
        #         reward -= 35
        self.total_reward += reward
        self.episode_length += 1
        info['total_reward'] = self.total_reward
        info['episode_length'] = self.episode_length

        return state, reward, done, info

    def reset(self):
        self.curr_pos = 0
        self.curr_score = 0
        self.curr_lives = 2
        self.total_reward = 0
        self.episode_length = 0
        state = np.float32(self.env.reset()) / 255.0
        return state
        # return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(ACTION_MAPPING[action])
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)


def create_train_env(level, max_episode_steps, output_path=None):
    env = gym.make('Contra-v0')
    
    if output_path:
        monitor = Monitor(240, 224, output_path)
    else:
        monitor = None
    env = TimeLimit(MaxAndSkipEnv(env), max_episode_steps=max_episode_steps)
    env = CustomReward(env, monitor)
    # env = CustomSkipFrame(env)
    return env


class MultipleEnvironments:
    def __init__(self, level, num_envs, max_episode_steps, output_path=None):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        env = create_train_env(level, max_episode_steps, output_path=output_path)
        self.num_states = env.observation_space.shape[-1]
        env.close()
        self.num_actions = len(ACTION_MAPPING)
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index, level, max_episode_steps, output_path))
            process.start()
            self.env_conns[index].close()

    def run(self, index, level, max_episode_steps, output_path):
        env = create_train_env(level, max_episode_steps, output_path=output_path)
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(env.step(action.item()))
            elif request == "reset":
                self.env_conns[index].send(env.reset())
            else:
                raise NotImplementedError
