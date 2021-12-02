import torch
import torch.nn as nn
from distributions_opp import Agent_Actor
from torch.distributions import Categorical





class Policy(nn.Module):
    def __init__(self, nn, action_space,opp_agents):
        super(Policy, self).__init__()

        assert isinstance(nn, torch.nn.Module)
        self.nn = nn
        num_outputs = action_space.n
        self.dist = Agent_Actor(self.nn.output_size, num_outputs,opp_agents)
        #self.agent_prob = Agent_Actor(self.nn.output_size, num_outputs, opp_agents)

        # print('self.dist',self.dist)



    @property
    def is_recurrent(self):
        return self.nn.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.nn.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks,deterministic=False):
        #print('act!!!!')
        #print('inputs',inputs)
        value, actor_features, rnn_hxs = self.nn(inputs, rnn_hxs, masks)
        #print('actor_features',actor_features)
        value, actor_features, rnn_hxs = self.nn(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        # print('dist',dist)
        if deterministic:
            action = dist.argmax(dim=1, keepdim=True)
        else:
            action = Categorical(dist).sample()
        #action = Categorical(dist).sample()
        # print('action_act is',action)
        action_log_probs=torch.log(torch.gather(dist, dim=1, index=action))
        #action_log_probs = dist.log_probs(action)  # action_log_probs0 tensor([[-1.7920]])
        # print('action_log_probs',action_log_probs)

        _ = Categorical(dist).entropy().mean()


        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        # print('get_value!!!')
        value, _, _ = self.nn(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        # print('eva_action!!!')
        # print('actiin is',action)

        value, actor_features, rnn_hxs = self.nn(inputs, rnn_hxs, masks)
        # print('value_eval is',value)
        actions_prob = self.dist(actor_features)
        # print('actions_prob',actions_prob)
        # print('action000 is',action)
        action_log_probs=torch.log(torch.gather(actions_prob, dim=1, index=action))
        # print('action_log_probs',action_log_probs)
        dist_entropy = Categorical(actions_prob).entropy().mean()
        # print('dist_entropy',dist_entropy)

        return value, action_log_probs, dist_entropy, rnn_hxs