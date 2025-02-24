import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():
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
        #self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps)
        if lr_schedule is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, lr_schedule)
        else:
            self.scheduler = None

    def update(self, rollouts, update_index, replay=None):
        if self.scheduler is not None:
            self.scheduler.step(update_index)
        obs_shape = rollouts.obs.size()[2:]

        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        # print('num_steps',num_steps)
        for e in range(self.ppo_epoch):
            values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape))

            action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
            values = values.view(num_steps, num_processes, 1)

            old_action_log_probs = rollouts.action_log_probs
            # print('old_action_log_probs',old_action_log_probs)
            # print('rollouts.returns[:-1]',rollouts.returns[:-1].size())
            # print('rollouts.value_preds[:-1]',rollouts.value_preds[:-1].size())
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

            advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-5).to(rollouts.returns[:-1].device)
            # print('adv', advantages.size())

            # actor loss
            ratio = torch.exp(action_log_probs - old_action_log_probs)

            surr1 = ratio * advantages

            surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                1.0 + self.clip_param) * advantages
            action_loss = -torch.min(surr1, surr2).mean()

            #critic loss
            value_loss = F.mse_loss(rollouts.returns[:-1], values)

            self.optimizer.zero_grad()

            (value_loss * self.value_loss_coef + action_loss -
             dist_entropy * self.entropy_coef).backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)
            self.optimizer.step()
            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.item()
        value_loss_epoch /= self.ppo_epoch
        action_loss_epoch /= self.ppo_epoch
        dist_entropy_epoch /= self.ppo_epoch


        '''value_loss_epoch = 0
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


                ratio = torch.exp(action_log_probs - old_action_log_probs_batch).to(obs_batch.device)

                surr1 = ratio * adv_targ

                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ

                action_loss = -torch.min(surr1, surr2).mean().to(obs_batch.device)

                value_loss = F.mse_loss(return_batch, values).to(obs_batch.device)

                self.optimizer.zero_grad()

                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch #64

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates'''


        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, {}
