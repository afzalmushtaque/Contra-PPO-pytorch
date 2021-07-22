"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch
from src.env import create_train_env
from src.model import PPO
import torch.nn.functional as F
from collections import deque
from PIL import Image
import numpy as np
from torch.distributions import Categorical


def test(opt, global_model, num_states, num_actions, max_episode_steps):
    torch.manual_seed(123)
    env = create_train_env(opt.level, max_episode_steps)
    local_model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()
    state = torch.from_numpy(env.reset())
    if torch.cuda.is_available():
        state = state.cuda()
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    record_complete = True
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        if not record_complete:
            image_array = state.cpu().numpy()
            img = Image.fromarray(np.uint8(image_array), 'RGB')
            img.save('vision/state_' + str(curr_step) + '.jpg')
        
        logits, value = local_model(state.unsqueeze(dim=0).permute(0,3,1,2))
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = Categorical(policy).sample().cpu().item()
        state, reward, done, info = env.step(action)
        if (done and info["lives"] != 0) or info["level"] == opt.level:
            torch.save(local_model.state_dict(), "{}/ppo_contra_success_{}".format(opt.saved_path, info["lives"]))

        env.render()
        actions.append(action)
        if curr_step > opt.max_episode_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
            record_complete = True
        state = torch.from_numpy(state)
        if torch.cuda.is_available():
            state = state.cuda()
