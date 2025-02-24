import torch
import torch.nn as nn
from distributions_opp1217 import Agent_Actor
from torch.distributions import Categorical

import numpy as np
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)



class Policy(nn.Module):
    def __init__(self, nn1, nn2, action_space, agent_num, opp_agents,num_quant):
        super(Policy, self).__init__()
        self.agent_num = agent_num
        self.num_quant = num_quant

        assert isinstance(nn1, torch.nn.Module)
        assert isinstance(nn2, torch.nn.Module)
        self.nn_actor = nn1

        self.nn_critic = nn2
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            #self.dist = Categorical(self.nn.output_size, num_outputs)
            self.dist = Agent_Actor(self.nn_actor.output_size, num_outputs,opp_agents)
        else:
            raise NotImplementedError
        #self.agent_prob = Agent_Actor(self.nn.output_size, num_outputs, opp_agents)

        #print('self.dist',self.dist)



    @property
    def is_recurrent(self):
        return self.nn_actor.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.nn_actor.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks,deterministic=False):

        action_log_probs1 = []
        action1 = []
        dist_entropy_act = []
        opp_dist_entropy_act = []
        for agent_id in range(self.agent_num):
            input_critic = inputs.transpose(0, 1).to(inputs.device)  # torch.Size([2, 16, 1092])
            actor_features, rnn_hxs = self.nn_actor(input_critic[agent_id], rnn_hxs, masks)
            action_prob,opp_action_probs,opp_actions_entropy = self.dist(actor_features)

            dist = FixedCategorical(logits=action_prob)  # Categorical(logits: torch.Size([16, 6]))
            if deterministic:
                action_act = dist.mode()
            else:
                action_act = dist.sample()  # torch.size([16,2])
            action_log_probs_act = dist.log_probs(action_act)
            action1.append(action_act)
            action_log_probs1.append(action_log_probs_act)
            #_ = dist.entropy().mean()
            dist_entropy_act1 = dist.entropy().mean()
            dist_entropy_act.append(dist_entropy_act1)
            opp_dist_entropy_act.append(opp_actions_entropy)
        dist_entropy = sum(dist_entropy_act) / 2
        opp_dist_entropy = sum(opp_dist_entropy_act) / 2
        action = torch.cat(action1, dim=-1)  # torch.size([16,2])
        action_log_probs = torch.cat(action_log_probs1, dim=-1)  # action_log_probs torch.Size([16, 2])
        value1 = []
        for agent_id in range(self.agent_num):
            input_critic = inputs.transpose(0, 1).to(inputs.device)
            batch_size = len(input_critic[agent_id])
            ids = (torch.ones(batch_size) * agent_id).view(-1, 1).to(inputs.device)
            value0 = self.nn_critic(input_critic, ids, rnn_hxs, masks, action)  # torch.Size([16, 5])

            action_taken = action.type(torch.long)[:, agent_id].reshape(-1, 1)
            value_taken = value0[np.arange(len(value0)), action_taken.squeeze(-1)]

            value1.append(value_taken)

        value = torch.cat(value1, dim=0).reshape(self.agent_num, len(value1[0]), self.num_quant)

        return value, action, action_log_probs,dist_entropy,opp_action_probs,opp_dist_entropy, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, actions):
        value1 = []
        input_critic = inputs.transpose(0, 1).to(inputs.device)
        for agent_id in range(self.agent_num):
            batch_size = len(input_critic[agent_id])
            ids = (torch.ones(batch_size) * agent_id).view(-1, 1).to(inputs.device)
            value_act = self.nn_critic(input_critic, ids, rnn_hxs, masks, actions)

            value_taken = value_act[np.arange(len(value_act)), value_act.mean(2).max(1)[1]]

            value1.append(value_taken)

        value = torch.cat(value1, dim=0).reshape(self.agent_num, len(value1[0]), self.num_quant)

        return value

    def evaluate_actions(self, agent_id, inputs, rnn_hxs, masks, action):

        input_critic = inputs.transpose(0, 1).to(inputs.device)
        batch_size = len(input_critic[0])
        ids = (torch.ones(batch_size) * agent_id).view(-1, 1).to(inputs.device)
        value = self.nn_critic(input_critic, ids, rnn_hxs, masks, action)

        actor_features, rnn_hxs = self.nn_actor(input_critic[agent_id], rnn_hxs, masks)
        action_probs,_,_ = self.dist(actor_features)

        dist = FixedCategorical(logits=action_probs)
        action_taken = action.type(torch.long)[:, agent_id].reshape(-1, 1)
        action_log_probs = dist.log_probs(action_taken)
        dist_entropy = dist.entropy().mean()
        value_taken = value[np.arange(len(value)), action_taken.squeeze(-1)]



        return value_taken, action_log_probs, dist_entropy, rnn_hxs