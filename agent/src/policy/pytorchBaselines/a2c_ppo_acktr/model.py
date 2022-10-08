import torch
import torch.nn as nn

from policy.pytorchBaselines.a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from policy.pytorchBaselines.a2c_ppo_acktr.srnn_model import SRNN

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, config):
        super(Policy, self).__init__()
        self.base = SRNN(config)
        self.srnn = True
        num_outputs = config["action_size"]
        self.dist = DiagGaussian(self.base.output_size, num_outputs)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        
        dist = self.dist(actor_features)
        action = dist.mode()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):

        value, _, _ = self.base(inputs, rnn_hxs, masks)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs



