from .model_pomm_QR import PommNet
from .model_generic import CNNBase, MLPBase
from .policy_QR import Policy


def create_policy(obs_space, action_space,num_quant, name='basic', nn_kwargs={}, train=True):
    #print('@@@')
    nn = None
    obs_shape = obs_space.shape
    num_outputs = action_space.n
    if name.lower() == 'basic':
        if len(obs_shape) == 3:
            nn = CNNBase(obs_shape[0], **nn_kwargs)
        elif len(obs_shape) == 1:
            nn = MLPBase(obs_shape[0], **nn_kwargs)
        else:
            raise NotImplementedError
    elif name.lower() == 'pomm':
        nn = PommNet(
            obs_shape=obs_shape,
            action_dim=num_outputs,
            num_quant=num_quant,
            **nn_kwargs)
    else:
        assert False and "Invalid policy name"

    if train:
        nn.train()
    else:
        nn.eval()

    policy = Policy(nn, action_space=action_space)

    return policy
