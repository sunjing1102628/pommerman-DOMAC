import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

# FixedCategorical = torch.distributions.Categorical

'''old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)
log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)'''


#


class OPP_Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(OPP_Actor, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)

        x = F.softmax(x, dim=-1)

        return x


class Agent_Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, opp_agents):
        super(Agent_Actor, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)
        self.linear = init_(nn.Linear(num_inputs + opp_agents * 6, num_outputs))

        self.opp_actors = nn.ModuleList([OPP_Actor(num_inputs, num_outputs) for _ in range(opp_agents)])

        # print('self.',self.opp_actors)

    def forward(self, x):
        # print('x is',len(x))
        opp_actions0 = []
        opp_actions_probs0 = []

        for opp_actor in self.opp_actors:
            #print('!')
            opp_action_dist = opp_actor(x)
            # print('opp_action_dist',opp_action_dist)

            num_sample = 20
            opp_actions = torch.zeros(len(x), num_sample).long()
            opp_actions_prob = []
            for i in range(num_sample):
                opp_action = torch.distributions.Categorical(opp_action_dist).sample()
                opp_action_prob0 = torch.gather(opp_action_dist, dim=-1, index=opp_action)

                opp_actions_prob.append(opp_action_prob0)
                opp_actions[:, i].copy_(opp_action.squeeze(-1))
            opp_action_prob = torch.cat(opp_actions_prob, dim=-1)
            opp_actions0.append(opp_actions)

            opp_actions_probs0.append(opp_action_prob)

        opp_actions_probs1 = opp_actions_probs0[0] * opp_actions_probs0[1]

        opp_actions_probs2 = opp_actions_probs1 / torch.sum(opp_actions_probs1, 1).unsqueeze(1)

        opp_actions_probs3 = opp_actions_probs2.reshape(len(x), 1, num_sample) #opp_actions_probs3 torch.Size([16, 1, 18])

        opp_actions2 = torch.cat(opp_actions0, dim=1).reshape(len(x), 2, num_sample).transpose(-1, -2)#torch.Size([3, 5, 2])



        opp_actions_num = torch.nn.functional.one_hot(opp_actions2, 6).view(len(x), num_sample, 12).to(x.device)

        a = torch.cat([x.repeat(1, num_sample).reshape(len(x), num_sample, len(x[0])),
                       opp_actions_num.to(x.device)],
                      dim=2)
        # print('a',a)
        # print('a_size',a.size())
        a = self.linear(a)
        # print('a2 is',a)
        agent_action_probs = F.softmax(a, dim=-1).to(x.device)
        # print('agent_action_probs is', agent_action_probs)
        # print('agent_action_probs_size is', agent_action_probs.size())
        # print('opp_actions_probs',opp_actions_probs3.size())
        actions_probs = torch.matmul(opp_actions_probs3, agent_action_probs).squeeze(1).to(x.device) #torch.Size([16, 6])

        #print('actions_probs.size', actions_probs.size())

        return actions_probs


