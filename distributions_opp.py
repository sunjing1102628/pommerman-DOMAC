import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AddBias, init, init_normc_

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

# FixedCategorical = torch.distributions.Categorical
#
# old_sample = FixedCategorical.sample
# FixedCategorical.sample = lambda self: old_sample(self)
#
# log_prob_cat = FixedCategorical.log_prob
# FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)
#
# FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

class Opp_Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Opp_Actor, self).__init__()

        init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):

        x = self.linear(x)
        # print('x_opp is',x)
        x= F.softmax(x, dim=-1)
        # print('x2_opp is',x)
        return x


class Agent_Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs,opp_agents):
        super(Agent_Actor, self).__init__()

        init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              gain=0.01)

        self.linear = init_(nn.Linear(num_inputs+opp_agents*6, num_outputs))
        self.opp_actors = nn.ModuleList([Opp_Actor(num_inputs, num_outputs) for _ in range(opp_agents)])


    def forward(self, x):
        opp_actions0 = []
        opp_actions_probs0 = []
        evl_oppaction_prob = []
        opp_actions_entropy = []
        for opp_actor in self.opp_actors:
            opp_action_dist = opp_actor(x)
            dist_entropy = torch.distributions.Categorical(opp_action_dist).entropy().mean()
            opp_actions_entropy.append(dist_entropy)
            evl_oppaction_prob.append(opp_action_dist)

            num_sample = 80
            opp_actions = torch.zeros(num_sample, len(x),1).long()
            for i in range(num_sample):
                opp_action = torch.distributions.Categorical(opp_action_dist).sample()
                opp_actions[i].copy_(opp_action)
            opp_action_prob = torch.gather(opp_action_dist.repeat(num_sample, 1).reshape(num_sample, len(x), 6), dim=2,
                                           index=opp_actions.to(x.device)).to(x.device)
            opp_actions0.append(opp_actions.squeeze().t())
            opp_actions_probs0.append(opp_action_prob.squeeze().t())
        opp_actions_probs1 = opp_actions_probs0[0] * opp_actions_probs0[1]*opp_actions_probs0[2]
        opp_actions_probs2 = opp_actions_probs1 / torch.sum(opp_actions_probs1, 1).unsqueeze(1)
        opp_actions_probs3 = opp_actions_probs2.reshape(len(x),1,num_sample).to(x.device)
        actions2 = torch.cat(opp_actions0, dim=1).reshape(len(x), 3, num_sample).to(x.device)
        actions3 = actions2.transpose(-1, -2).to(x.device)


        opp_actions_num = torch.nn.functional.one_hot(actions3, 6).view(len(x), num_sample, 18).to(x.device)

        a = torch.cat([x.repeat(1, num_sample).reshape(len(x), num_sample, len(x[0])),
                       opp_actions_num.to(x.device)],
                      dim=2).to(x.device)

        a = self.linear(a).to(x.device)
        # print('x is',a)
        agent_action_probs= F.softmax(a, dim=-1).to(x.device)
        # print('opp_actions_probs_size',opp_actions_probs.reshape(len(x),1, 6).size())
        # print('agent_probs is',agent_action_probs.size())
        # print('action_probs1', torch.matmul(opp_actions_probs, agent_action_probs).size())
        actions_probs = torch.matmul(opp_actions_probs3, agent_action_probs).squeeze(1).to(x.device)
        
        # print('actions_probs',actions_probs.size())

        return actions_probs,evl_oppaction_prob,sum(opp_actions_entropy)/3



