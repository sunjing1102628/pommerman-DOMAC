import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='qra2c',
                        help='algorithm to use: a2c|sa2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=2.5e-5,
                        help='learning rate (default: 2.5e-4)')
    parser.add_argument("--num-quant", type=str, default=5, help="numbers of the quant")
    parser.add_argument("--state-dim", type=str, default=1092, help="numbers of the state dim")
    parser.add_argument("--action-dim", type=str, default=6, help="numbers of the action dim")
    parser.add_argument("--target_update_interval", type=int, default=20,
                        help="update target network once every time this many episodes are completed")
    #parser.add_argument("--lr-actor", type=float, default=3e-4, help="learning rate of actor")
    #parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--n-agents", type=str, default=2, help="numbers of the agents")
    parser.add_argument("--opp-agents", type=str, default=2, help="numbers of the agents")
    parser.add_argument('--lr-schedule', type=float, default=None,
                        help='learning rate step schedule (default: None)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=20,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--sil-update-ratio', type=float, default=1.0,
                        help='sil off-policy updates per on-policy updates (default: 1.0)')
    parser.add_argument('--sil-epochs', type=int, default=1,
                        help='number of sil epochs (default: 1)')
    parser.add_argument('--sil-batch-size', type=int, default=80,
                        help='sil batch size (default: 80)')
    parser.add_argument('--sil-entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.0)')
    parser.add_argument('--sil-value-loss-coef', type=float, default=0.01,
                        help='value loss coefficient (default: 0.01)')
    parser.add_argument('--num-stack', type=int, default=1,
                        help='number of frames to stack (default: 1)')
    # parser.add_argument('--update-interval', type=int, default=10,
    #                     help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=5000,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=1000,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-frames', type=int, default=5e7,
                        help='number of frames to train (default: 5e7)')
    parser.add_argument('--env-name', default='PommeTeamCompetitionFast-v01',
                        help='environment to train on (default: PommeTeamCompetitionFast-v01PommeFFACompetitionFast-v0)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')

    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument("--save-rate", type=int, default=200000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--model-dir", type=str, default="",
                        help="directory in which training state and model are loaded")

    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--no-norm', action='store_true', default=False,
                        help='disables normalization')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    return args
