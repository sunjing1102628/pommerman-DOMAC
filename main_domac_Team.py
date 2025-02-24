import copy
import glob
import os
import time
import types
from collections import deque
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import algo
from arguments import get_args
from envs.make_env2 import make_vec_envs
from models.factory_opp_QR1219 import create_policy
from rollout_storage_QR2 import RolloutStorage
from replay_storage import ReplayStorage
#from visualize import visdom_plot

args = get_args()

assert args.algo in ['a2c','qra2c', 'ppo', 'ppo-sil', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR or SIL'

update_factor = args.num_steps * args.num_processes
num_updates = int(args.num_frames) // update_factor
lr_update_schedule = None if args.lr_schedule is None else args.lr_schedule // update_factor

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"
try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    '''if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None'''

    train_envs = make_vec_envs(
        args.env_name, args.seed, args.num_processes, args.gamma, args.no_norm, args.num_stack,
        args.log_dir, args.add_timestep, device, allow_early_resets=False)

    if args.eval_interval:
        eval_envs = make_vec_envs(
            args.env_name, args.seed + args.num_processes, args.num_processes, args.gamma,
            args.no_norm, args.num_stack, eval_log_dir, args.add_timestep, device,
            allow_early_resets=True, eval=True)

        if eval_envs.venv.__class__.__name__ == "VecNormalize":
            eval_envs.venv.ob_rms = train_envs.venv.ob_rms
    else:
        eval_envs = None

    # FIXME this is very specific to Pommerman env right now
    actor_critic = create_policy(
        train_envs.observation_space,
        train_envs.action_space,
        args.n_agents,
        args.num_quant,
        args.opp_agents,
        name='pomm',
        nn_kwargs={
            'batch_norm': False if args.algo == 'acktr' else True,
            'recurrent': args.recurrent_policy,
            'hidden_size': 512,
        },
        train=True)

    actor_critic.to(device)



    if args.algo.startswith('qra2c'):
        agent = algo.A2C_TeamQR(
            actor_critic, args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr, lr_schedule=lr_update_schedule,
            eps=args.eps, alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo.startswith('ppo'):
        agent = algo.PPO(
            actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
            args.value_loss_coef, args.entropy_coef,
            lr=args.lr, lr_schedule=lr_update_schedule,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef,
            args.entropy_coef,
            acktr=True)

    if args.algo.endswith('sil'):
        agent = algo.SIL(
            agent,
            update_ratio=args.sil_update_ratio,
            epochs=args.sil_epochs,
            batch_size=args.sil_batch_size,
            value_loss_coef=args.sil_value_loss_coef or args.value_loss_coef,
            entropy_coef=args.sil_entropy_coef or args.entropy_coef)
        replay = ReplayStorage(
            5e5,
            args.num_processes,
            args.gamma,
            0.1,
            train_envs.observation_space.shape,
            train_envs.action_space,
            actor_critic.recurrent_hidden_state_size,
            device=device)
    else:
        replay = None

    rollouts = RolloutStorage(
        args.num_steps, args.num_processes,
        train_envs.observation_space.shape,
        train_envs.action_space,
        args.num_quant,
        args.n_agents,
        actor_critic.recurrent_hidden_state_size)

    obs = train_envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    log_mean = []
    log_std = []
    log_dist_entropy = []
    log_dist_entropy_std = []
    log_acc_mean=[]
    log_acc_std=[]
    log_opp_dist_entropy=[]
    log_opp_dist_entropy_std=[]


    start = time.time()
    for j in tqdm(range(num_updates)):
        for step in range(args.num_steps):
            #print('args.num_steps',args.num_steps)
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob,_,_, _, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos,true_opp = train_envs.step(action)



            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            # print('done_main',done)
            masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], device=device)
            # print('masks_main',masks)
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)
            if replay is not None:
                replay.insert(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    action,
                    reward,
                    done)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1],
                                                rollouts.actions[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy, other_metrics = agent.update(rollouts, j, replay)
        # print('other_metrics',other_metrics)

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model.state_dict(),
                          hasattr(train_envs.venv, 'ob_rms') and train_envs.venv.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + '_' +str(j)+ ".pt"))

        total_num_steps = (j + 1) * update_factor

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()

            print('episode_rewards',episode_rewards)
            print('value_loss',value_loss)
            print('action_loss',action_loss)
            print("Updates {}, num timesteps {}, FPS {}, last {} mean/median reward {:.1f}/{:.1f}, "
                  "min / max reward {:.1f}/{:.1f}, value/action loss {:.5f}/{:.5f}".
                  format(j, total_num_steps,
                         int(total_num_steps / (end - start)),
                         len(episode_rewards),
                         np.mean(episode_rewards),
                         np.median(episode_rewards),
                         np.min(episode_rewards),
                         np.max(episode_rewards), dist_entropy,
                         value_loss, action_loss), end=', ' if other_metrics else '\n')
            if 'sil_value_loss' in other_metrics:
                print("SIL value/action loss {:.1f}/{:.1f}.".format(
                    other_metrics['sil_value_loss'],
                    other_metrics['sil_action_loss']
                ))

        if args.eval_interval and len(episode_rewards) > 1 and j > 0 and j % args.eval_interval == 0:
            eval_episode_rewards = []
            eval_dist_entropy = []
            eval_opp_dist_entropy = []
            acc = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                                       actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 200:
                with torch.no_grad():
                    _, action, _, eval_dist_entropy1,evl_opp_action_probs,eval_opp_dist_entropy1,eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)
                    eval_dist_entropy.append(eval_dist_entropy1.tolist())
                    eval_opp_dist_entropy.append(eval_opp_dist_entropy1.tolist())

                # Obser reward and next obs
                obs, reward, done, infos, evl_true_opp = eval_envs.step(action)
                CL_loss = []
                loss = nn.CrossEntropyLoss()
                for i in range(2):
                    input_opp = evl_opp_action_probs[i]
                    #print('input_opp', input_opp)
                    #print('input_opp', input_opp.size())
                    target_opp = evl_true_opp[:, i]
                    #print('target_opp', target_opp)
                    #print(target_opp.size())
                    loss0 = loss(input_opp, target_opp)

                    CL_loss.append(loss0)
                CL_loss1= sum(CL_loss)/2
                acc.append(CL_loss1.tolist())
                eval_masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], device=device)
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])
            #log eval_episode reward mean and std
            log_mean.append([j, np.mean(eval_episode_rewards)])
            eval_episode_rewards_std = np.array(eval_episode_rewards).std()
            log_std.append([j, eval_episode_rewards_std])
            #log eval_acc mean and std

            log_acc_mean.append([j, np.mean(acc)])
            eval_acc_std = np.array(acc).std()
            log_acc_std.append([j,eval_acc_std])
            #log eval_dist_entropy mean and std
            log_dist_entropy.append([j, np.mean(eval_dist_entropy)])
            eval_dist_entropy_std = np.array(eval_dist_entropy).std()
            log_dist_entropy_std.append([j, eval_dist_entropy_std])
            # log eval_opp_dist_entropy mean and std
            log_opp_dist_entropy.append([j, np.mean(eval_opp_dist_entropy)])
            eval_opp_entropy_std = np.array(eval_opp_dist_entropy).std()
            log_opp_dist_entropy_std.append([j, eval_opp_entropy_std])

            print(" using {} episodes: mean reward {:.5f}\n".
                  format(j, np.mean(eval_episode_rewards)))
            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                  format(len(eval_episode_rewards), np.mean(eval_episode_rewards)))
        np.savetxt('./results/domac_Team_seed40/train_score_seed_{}.csv'.format(40),
                   np.array(log_mean),
                   delimiter=";")
        np.savetxt('./results/domac_Team_seed40/train_scorestd_seed_{}.csv'.format(40),
                   np.array(log_std),
                   delimiter=";")
        '''np.savetxt('./results/domac_Team_partial/train_dist_entropy_seed_{}.csv'.format(42),
                   np.array(log_dist_entropy),
                   delimiter=";")
        np.savetxt('./results/domac_Team_partial/train_dist_entropystd_seed_{}.csv'.format(42),
                   np.array(log_dist_entropy_std),
                   delimiter=";")
        np.savetxt('./results/domac_Team_partial/train_acc_seed_{}.csv'.format(42),
            np.array(log_acc_mean),
            delimiter=";")
        np.savetxt('./results/domac_Team_partial/train_accstd_seed_{}.csv'.format(42),
            np.array(log_acc_std),
            delimiter=";")'''



        '''if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass'''


if __name__ == "__main__":

    main()
