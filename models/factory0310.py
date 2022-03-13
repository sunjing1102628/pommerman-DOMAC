from .model_pomm1203 import ActorNet,CriticNet
from .model_generic import CNNBase, MLPBase
from .policy0310 import Policy


def create_policy(obs_space, action_space, name='basic', nn_kwargs={}, train=True):
    nn = None
    obs_shape = obs_space.shape
    if name.lower() == 'basic':
        if len(obs_shape) == 3:
            nn = CNNBase(obs_shape[0], **nn_kwargs)
        elif len(obs_shape) == 1:
            nn = MLPBase(obs_shape[0], **nn_kwargs)
        else:
            raise NotImplementedError
    elif name.lower() == 'pomm':
        '''nn = PommNet(
            obs_shape=obs_shape,
            **nn_kwargs)'''
        nn1 = ActorNet(
            obs_shape=obs_shape,
            **nn_kwargs)
        nn2 = CriticNet(
            obs_shape=obs_shape,
            **nn_kwargs)
    else:
        assert False and "Invalid policy name"

    if train:
        nn1.train()
        nn2.train()
    else:
        nn1.eval()
        nn2.eval()

    #policy = Policy(nn, action_space=action_space)
    policy = Policy(nn1, nn2, action_space=action_space)

    return policy
