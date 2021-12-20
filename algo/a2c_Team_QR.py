import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .kfac import KFACOptimizer
import torch.nn.functional as F

class A2C_Team():
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

        values,action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))
        #print('values11',values) #torch.Size([2, 80, 5])
        #print('values11.size',values.size())
        values = values.view(2, num_steps, num_processes, num_quant)
        #print('values.size',values)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 2)
        #print('dist_entropy',dist_entropy) # [tensor(1.7918, grad_fn=<MeanBackward0>), tensor(1.7918, grad_fn=<MeanBackward0>)]

        values_loss =[]
        action_loss =[]
        for i in range(self.agent_num):


            advantages = rollouts.returns[:-1][:,i] - values[i] #torch.Size([5, 16, 5])

            theta = values[i].unsqueeze(3) #torch.Size([5, 16, 5, 1])

            Theta = rollouts.returns[:-1][:,i].unsqueeze(2) #torch.Size([5, 16, 1, 5])

            u = Theta - theta
            #print('tau',tau) #torch.Size([1, 5, 1])

            #print('u.le(0.)',u.le(0.).size())
            weight = torch.abs(tau - u.le(0.).float())#torch.Size([5, 16, 5, 5])
            loss0 = F.smooth_l1_loss(theta, Theta.detach(), reduction='none')
            #print('loss0', loss0) #torch.Size([5, 16, 5, 5])
            loss2 = torch.mean(weight * loss0, dim=1).mean(dim=1)
           # print('loss2 is', loss2) #torch.Size([5, 5])
            value_loss0 = torch.mean(loss2) #value_loss0 tensor(0.0021, grad_fn=<MeanBackward0>)

            values_loss.append(value_loss0)
            #print('action_log_probs[:,:,i]', action_log_probs[:,:,i].unsqueeze(-1).size()) #torch.Size([5, 16, 1])
            #print('advantages',advantages.size()) #torch.Size([5, 16, 5])

            action_loss0 = -(advantages.detach() * action_log_probs[:,:,i].unsqueeze(-1)).mean()
            action_loss.append(action_loss0)


        #print('val_loss',sum(values_loss))
        #print('action_loss',sum(action_loss))
        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            #print('asktr')
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
        (sum(values_loss) * self.value_loss_coef + sum(action_loss) -
         sum(dist_entropy) * self.entropy_coef).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()
        #print('value_loss.item()', sum(values_loss).item())
        #print('action_loss.item()', sum(action_loss).item())


        return sum(values_loss).item(), sum(action_loss).item(), sum(dist_entropy), {}
