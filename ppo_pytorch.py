"""
Actor-Critic network for PPO using PyTorch.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from collections import OrderedDict

class ActorCritic(nn.Module):

    def __init__(self, action_space, ff_dim):
        super().__init__()

        self.network = nn.Sequential(
            OrderedDict([
                ("layer1", nn.Linear(action_space, ff_dim)),
                ("gelu1", nn.GELU()),
                ("layer2", nn.Linear(ff_dim, ff_dim)),
                ("gelu2", nn.GELU()),
                ("layer2", nn.Linear(ff_dim, 4)),
                ("gelu2", nn.GELU()),
                ("softmax", nn.Softmax(dim=-1))
            ])
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = ActorCritic(action_space=8, ff_dim=64)
    ipt = torch.randn(1, 8)
    print(f'ipt size', {ipt.shape})
    print(model(ipt))