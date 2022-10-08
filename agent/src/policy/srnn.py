import numpy as np
import torch
import torch.nn as nn

from policy.pytorchBaselines.a2c_ppo_acktr.model import Policy

class SRNN():
    def __init__(self, config):
        """
        A policy for static obstacle.
        It does nothing
        """
        # initialize torch settings
        torch.manual_seed(config["torch_seed"])
        torch.cuda.manual_seed_all(config["torch_seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(1)
        self.device = torch.device(config["device"])
        # initialize srnn model
        self.actor_critic = Policy(config)
        self.actor_critic.load_state_dict(torch.load(config["ckpt_path"], map_location=self.device))
        self.actor_critic.base.nenv = 1

        # allow the usage of multiple GPUs to increase the number of examples processed simultaneously
        nn.DataParallel(self.actor_critic).to(self.device)
        
        # initialize hidden states
        node_num = 1
        edge_num = self.actor_critic.base.human_num + 1
        rnn_factor = 1 # 1 for GRU, 2 for LSTM
        self.recurrent_hidden_states = {}
        self.recurrent_hidden_states['human_node_rnn'] = torch.zeros(1, node_num, config["human_node_rnn_size"] * rnn_factor, device=self.device)
        self.recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(1, edge_num, config["human_human_edge_rnn_size"]*rnn_factor, device=self.device)
        
        self.masks = torch.zeros(1, 1, device=self.device)
        
    def predict(self, state):
        """
        vx = 0
        vy = 0
        return (vx,vy)
        """
        # convert to tensor in the shape of (len_seq*nenv, num_agent, num_feature)
        state["robot_node"] = torch.tensor(state["robot_node"], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        state["spatial_edges"] = torch.tensor(state["spatial_edges"], dtype=torch.float32, device=self.device).unsqueeze(0)
        state["temporal_edges"] = torch.tensor(state["temporal_edges"], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        # compute action value
        _, action, _, self.recurrent_hidden_states = self.actor_critic.act(state, self.recurrent_hidden_states, self.masks)

        return action[0].detach().cpu().numpy()
