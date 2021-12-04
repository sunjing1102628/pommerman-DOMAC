import torch
import torch.nn as nn
from distributions import Categorical, DiagGaussian


class Policy(nn.Module):
    def __init__(self, nn_actor,nn_critic, action_space):
        super(Policy, self).__init__()

        assert isinstance(nn_actor, torch.nn.Module)
        assert isinstance(nn_critic, torch.nn.Module)
        #self.nn = nn
        self.nn_actor = nn_actor
        self.nn_critic = nn_critic

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
        #value, actor_features, rnn_hxs = self.nn(inputs, rnn_hxs, masks)
        actor_features, rnn_hxs = self.nn_actor(inputs, rnn_hxs, masks)
        value = self.nn_critic(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        _ = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value= self.nn_critic(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        #actor_features, rnn_hxs = self.nn(inputs, rnn_hxs, masks)
        actor_features, rnn_hxs = self.nn_actor(inputs, rnn_hxs, masks)
        #value = self.nn_critic(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return  action_log_probs, dist_entropy, rnn_hxs
