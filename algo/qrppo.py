import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

class DPPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 lr_schedule=None,
                 eps=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        if lr_schedule is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, lr_schedule)
        else:
            self.scheduler = None

    def update(self, rollouts, update_index, replay=None):
        if self.scheduler is not None:
            self.scheduler.step(update_index)

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample


                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch,
                    masks_batch, actions_batch)
                num_quant=5
                tau = torch.Tensor((2 * np.arange(num_quant) + 1) / (2.0 * num_quant)).view(1, -1).to(
                    values.device)


                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

                surr1 = ratio * adv_targ.mean(-1).unsqueeze(-1)

                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ.mean(-1).unsqueeze(-1)
                action_loss = -torch.min(surr1, surr2).mean()



                #value_loss = F.mse_loss(return_batch, values)
                #critic loss
                theta = values

                Theta = return_batch.t().unsqueeze(-1)

                u = Theta - theta

                weight = torch.abs(tau - u.le(0.).float())
                loss0 = F.smooth_l1_loss(theta, Theta.detach(), reduction='none')
                # print('loss0', loss0)
                # print('loss0.size', loss0.size())
                # (m, N_QUANT, N_QUANT)
                loss2 = torch.mean(weight * loss0, dim=1).mean(dim=1)
                # print('loss2 is', loss2)
                # print('loss2 is.size', loss2.size())
                value_loss = torch.mean(loss2)
                #diff = return_batch.t().unsqueeze(-1) - values

                #loss = huber(diff) * (tau - (diff.detach() < 0).float()).abs()

                #value_loss = loss.mean()


                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, {}
