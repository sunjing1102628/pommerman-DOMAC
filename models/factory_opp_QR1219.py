from .model_pomm1215 import ActorNet, CriticNet
from .model_generic import CNNBase, MLPBase
from .policy_maac_opp_QR1219 import Policy



def create_policy(obs_space, action_space,agent_num, num_quant,opp_agents, name='basic', nn_kwargs={}, train=True):
    # print('create_policy!')
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
        nn1 = ActorNet(
            obs_shape=obs_shape,
            **nn_kwargs)
        nn2 = CriticNet(
             obs_shape=obs_shape,
             num_quant =num_quant,
             **nn_kwargs)
    else:
        assert False and "Invalid policy name"

    if train:
        nn1.train()
        nn2.train()


    else:
        nn1.eval()
        nn2.eval()
    policy = Policy(nn1,nn2, action_space=action_space,agent_num=agent_num,opp_agents=opp_agents,num_quant=num_quant)

    return policy