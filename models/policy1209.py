import torch
import torch.nn as nn
from distributions1223 import Categorical, DiagGaussian
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

class Policy(nn.Module):
    def __init__(self, nn1,nn2, action_space,agent_num):
        super(Policy, self).__init__()
        self.agent_num = agent_num

        assert isinstance(nn1, torch.nn.Module)
        assert isinstance(nn2, torch.nn.Module)
        self.nn_actor = nn1
        self.nn_critic = nn2

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.nn_actor.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.nn_actor.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.nn_actor.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.nn_actor.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError


    def act(self, inputs, rnn_hxs, masks, deterministic=False):

        value1 =[]
        action_log_probs1=[]
        action1=[]


        for agent_id in range(self.agent_num):
            input_critic = inputs.transpose(0,1).to(inputs.device) #torch.Size([2, 16, 1092])
            actor_features, rnn_hxs = self.nn_actor(input_critic[agent_id], rnn_hxs, masks) #torch.Size([16, 512])
            action_prob = self.dist(actor_features)

            dist = FixedCategorical(logits=action_prob) #Categorical(logits: torch.Size([16, 6]))
            if deterministic:
                action_act = dist.mode()
            else:
                action_act = dist.sample() #torch.size([16,2])

            action_log_probs_act = dist.log_probs(action_act)
            action1.append(action_act)
            action_log_probs1.append(action_log_probs_act)
            _ = dist.entropy().mean()


        action= torch.cat(action1,dim= -1) # torch.size([16,2])

        action_log_probs = torch.cat(action_log_probs1,dim= -1) #action_log_probs torch.Size([16, 2])
        for agent_id in range(self.agent_num):
            input_critic = inputs.transpose(0, 1).to(inputs.device)
            batch_size = len(input_critic[agent_id])
            ids = (torch.ones(batch_size) * agent_id).view(-1, 1).to(inputs.device)

            value0 = self.nn_critic(input_critic, ids, rnn_hxs, masks, action)  # torch.Size([16, 6])

            action_taken = action.type(torch.long)[:, agent_id].reshape(-1, 1)
            value_taken = torch.gather(value0, dim=1, index=action_taken)
            value1.append(value_taken)
        value= torch.cat(value1,dim= -1) #value torch.Size([16, 2])
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, actions):

        value1 = []
        for agent_id in range(self.agent_num):
            input_critic = inputs.transpose(0, 1).to(inputs.device)
            batch_size = len(input_critic[agent_id])
            ids = (torch.ones(batch_size) * agent_id).view(-1, 1).to(inputs.device)
            value0 = self.nn_critic(input_critic, ids, rnn_hxs, masks, actions)
            action_taken = actions.type(torch.long)[:, agent_id].reshape(-1, 1)
            value_taken = torch.gather(value0, dim=1, index=action_taken)
            value1.append(value_taken)
        value = torch.cat(value1, dim=-1)  # torch.Size([16, 2])


        return value

    def evaluate_actions(self, agent_id, inputs, rnn_hxs, masks, action):

        input_critic = inputs.transpose(0, 1).to(inputs.device)
        batch_size = len(input_critic[0])
        ids = (torch.ones(batch_size) * agent_id).view(-1, 1).to(inputs.device)

        value = self.nn_critic(input_critic, ids, rnn_hxs, masks, action)


        actor_features, rnn_hxs = self.nn_actor(input_critic[agent_id], rnn_hxs, masks)
        action_probs = self.dist(actor_features)

        dist = FixedCategorical(logits=action_probs)
        action_taken = action.type(torch.long)[:, agent_id].reshape(-1, 1)
        value_taken = torch.gather(value, dim=1, index=action_taken)

        action_log_probs = dist.log_probs(action_taken)

        dist_entropy = dist.entropy().mean()

        return value, value_taken, action_probs, action_log_probs, dist_entropy, rnn_hxs
