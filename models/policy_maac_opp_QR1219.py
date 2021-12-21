import torch
import torch.nn as nn
from distributions_opp1217 import Agent_Actor
from torch.distributions import Categorical





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
        #print('act!!!!')
        value1 = []
        action_log_probs1 = []
        action1 = []
        for agent_id in range(self.agent_num):
            input_critic = inputs.transpose(0, 1).to(inputs.device)  # torch.Size([2, 16, 1092])


            value_act = self.nn_critic(input_critic, rnn_hxs, masks)
            value1.append(value_act)
            actor_features, rnn_hxs = self.nn_actor(input_critic[agent_id], rnn_hxs, masks)
            dist = self.dist(actor_features)
            if deterministic:
                action_act = dist.argmax(dim=1, keepdim=True)
            else:
                action_act = Categorical(dist).sample()
            action_log_probs_act = torch.log(torch.gather(dist, dim=1, index=action_act))
            _ = Categorical(dist).entropy().mean()
            action1.append(action_act)
            action_log_probs1.append(action_log_probs_act)
        action = torch.cat(action1, dim=-1)  # torch.size([16,2])

        action_log_probs = torch.cat(action_log_probs1, dim=-1)  # action_log_probs torch.Size([16, 2])

        value = torch.cat(value1, dim=-1).reshape(self.agent_num,len(value1[0]),self.num_quant)
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value1 = []
        for agent_id in range(self.agent_num):
            input_critic = inputs.transpose(0, 1).to(inputs.device)

            value_act = self.nn_critic(input_critic, rnn_hxs, masks)

            value1.append(value_act)
        value = torch.cat(value1, dim=-1).reshape(self.agent_num,len(value1[0]),self.num_quant)  # torch.Size([16, 2])

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value1 = []
        #print('input2 is', inputs.size())
        action_log_probs0 = []
        dist_entropy = []
        for agent_id in range(self.agent_num):
            input_critic = inputs.transpose(0, 1).to(inputs.device)

            value_act = self.nn_critic(input_critic,  rnn_hxs, masks)
            value1.append(value_act)
            actor_features, rnn_hxs = self.nn_actor(input_critic[agent_id], rnn_hxs, masks)
            #print('act_fea', actor_features.size())
            actions_prob = self.dist(actor_features)
            action_taken = action.type(torch.long)[:, agent_id].reshape(-1, 1)
            action_log_probs1 = torch.log(torch.gather(actions_prob, dim=1, index=action_taken))
            action_log_probs0.append(action_log_probs1)
            dist_entropy1 = Categorical(actions_prob).entropy().mean()
            dist_entropy.append(dist_entropy1)
        action_log_probs = torch.cat(action_log_probs0, dim=-1)
        value = torch.cat(value1, dim=-1).reshape(self.agent_num,len(value1[0]),self.num_quant)



        return value, action_log_probs, dist_entropy, rnn_hxs