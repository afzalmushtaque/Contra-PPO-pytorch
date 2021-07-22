"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch.nn as nn
import torch.nn.functional as F


class PPO(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PPO, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.critic_linear = nn.Sequential(
            # nn.Linear(64 * 7 * 7, 512),
            nn.Linear(32 * 26 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.actor_linear = nn.Sequential(
            # nn.Linear(64 * 7 * 7, 512),
            nn.Linear(32 * 26 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.normal_(module.weight, std=0.01)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.normal_(module.bias, std=0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = x.view(x.size(0), -1)
        x = x.flatten(start_dim=1)
        return self.actor_linear(x), self.critic_linear(x)

# class PPO_RAM(nn.Module):
#     def __init__(self, num_inputs, num_actions):
#         super(PPO, self).__init__()
#         self.dense1 = nn.Linear(num_inputs, 256)
#         self.critic_linear = nn.Sequential(
#             nn.Linear(256, 1)
#         )
#         self.actor_linear = nn.Sequential(
#             nn.Linear(256, num_actions)
#         )
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for module in self.modules():
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 nn.init.trunc_normal_(module.bias)

#     def forward(self, x):
#         x = F.relu(self.dense1(x))
#         return self.actor_linear(x), self.critic_linear(x)