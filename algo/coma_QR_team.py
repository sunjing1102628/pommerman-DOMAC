import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from .kfac import KFACOptimizer


class comaqr_team():
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
        self.agent_num=2

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
        tau = torch.Tensor((2 * np.arange(num_quant) + 1) / (2.0 * num_quant)).view(1, -1, 1).to(
            rollouts.actions.device)
        for i in range(self.agent_num):
            values,value_taken, action_prob, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(i,
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape))

            values = values.view(num_steps, num_processes,6, num_quant).mean(-1)

            action_prob = action_prob.view(num_steps, num_processes, 6)
            action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
            baseline = torch.sum(action_prob * values, dim=-1).detach()
            value_taken1=value_taken.mean(-1).view(num_steps, num_processes,1)

            advantages1 = value_taken1 - baseline.view(num_steps, num_processes, 1)

            #advantages = rollouts.returns[:-1][:,i] - values
            value_taken = value_taken.view(num_steps, num_processes, num_quant)
            theta = value_taken.unsqueeze(3)

            Theta = rollouts.returns[:-1][:,i].unsqueeze(2)

            u = Theta - theta
            weight = torch.abs(tau - u.le(0.).float())
            loss0 = F.smooth_l1_loss(theta, Theta.detach(), reduction='none')
            loss1 = torch.mean(weight * loss0, dim=1).mean(dim=1)
            value_loss = torch.mean(loss1)
            action_loss = -(advantages1.detach() * action_log_probs).mean()
            if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
                # print('asktr')
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
