import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .kfac import KFACOptimizer

def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

class QR_A2C_ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 lr_schedule=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
            self.scheduler = None
        else:
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)
            if lr_schedule is not None:
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, lr_schedule)
            else:
                self.scheduler = None

    def update(self, rollouts, update_index, replay=None):
        if self.scheduler is not None:
            self.scheduler.step(update_index)

        obs_shape = rollouts.obs.size()[2:]

        action_shape = rollouts.actions.size()[-1]
        num_quant = rollouts.value_preds.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        tau = torch.Tensor((2 * np.arange(num_quant) + 1) / (2.0 * num_quant)).view(1, -1).to(rollouts.actions.device)

        #print('num_steps',num_steps)

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, num_quant)
        #print('values.size',values.size())
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        #print('rollouts.returns[:-1], 2)',rollouts.returns[:-1].size())
        # Qtgt = torch.mean(rollouts.returns[:-1], 2).reshape(5, 16, 1)
        # #print('values',values.size())
        # Q = torch.mean(values, 2).reshape(5, 16, 1)
        # advantages = Qtgt-Q
        #print('advantages',advantages.size())

        advantages = rollouts.returns[:-1] - values
        theta = values.unsqueeze(3)
        Theta = rollouts.returns[:-1].unsqueeze(2)
        diff =  Theta - theta
        loss = huber(diff).to(rollouts.actions.device) * (tau - (diff.detach() < 0).float()).abs().to(
            rollouts.actions.device)
        value_loss = loss.mean()
        #value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item(), {}
