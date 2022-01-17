import pommerman
import pommerman.characters
import pommerman.envs
import numpy as np
import gym
import random

DEFAULT_FEATURE_CONFIG = {
    'recode_agents': True,
    'compact_powerups': True,
    'compact_structure': True,
    'rescale': True,
}


def make_np_float(feature):
    return np.array(feature).astype(np.float32)


def _rescale(x):
    return (x - 0.5) * 2.0


def get_feature_channels(config):
    num_channels = 15
    if config['recode_agents']:
        num_channels -= 2
    if config['compact_powerups']:
        num_channels -= 2
    if config['compact_structure']:
        num_channels -= 2
    return num_channels


def get_unflat_obs_space(channels=15, board_size=11, rescale=True):
    min_board_obs = np.zeros((channels, board_size, board_size))
    max_board_obs = np.ones_like(min_board_obs)
    min_other_obs = np.zeros(3)
    max_other_obs = np.ones_like(min_other_obs)

    if rescale:
        min_board_obs = _rescale(min_board_obs)
        max_board_obs = _rescale(max_board_obs)
        min_other_obs = _rescale(min_other_obs)
        max_other_obs = _rescale(max_other_obs)

    return gym.spaces.Tuple([
        gym.spaces.Box(min_board_obs, max_board_obs),
        gym.spaces.Box(min_other_obs, max_other_obs)])


def featurize(obs, agent_id, config):

    max_item = pommerman.constants.Item.Agent3.value

    ob = obs["board"]
    ob_bomb_blast_strength = obs["bomb_blast_strength"].astype(np.float32) / pommerman.constants.AGENT_VIEW_SIZE
    ob_bomb_life = obs["bomb_life"].astype(np.float32) / pommerman.constants.DEFAULT_BOMB_LIFE

    # one hot encode the board items
    ob_values = max_item + 1
    ob_hot = np.eye(ob_values)[ob]

    # replace agent item channels with friend, enemy, self channels
    if config['recode_agents']:
        self_value = pommerman.constants.Item.Agent0.value + agent_id
        enemies = np.logical_and(ob >= pommerman.constants.Item.Agent0.value, ob != self_value)
        self = (ob == self_value)
        friends = (ob == pommerman.constants.Item.AgentDummy.value)
        ob_hot[:, :, 9] = friends.astype(np.float32)
        ob_hot[:, :, 10] = self.astype(np.float32)
        ob_hot[:, :, 11] = enemies.astype(np.float32)
        ob_hot = np.delete(ob_hot, np.s_[12::], axis=2)

    if config['compact_powerups']:
        # replace powerups with single channel
        powerup = ob_hot[:, :, 6] * 0.5 + ob_hot[:, :, 7] * 0.66667 + ob_hot[:, :, 8]
        ob_hot[:, :, 6] = powerup
        ob_hot = np.delete(ob_hot, [7, 8], axis=2)

    # replace bomb item channel with bomb life
    ob_hot[:, :, 3] = ob_bomb_life

    if config['compact_structure']:
        ob_hot[:, :, 0] = 0.5 * ob_hot[:, :, 0] + ob_hot[:, :, 5]  # passage + fog
        ob_hot[:, :, 1] = 0.5 * ob_hot[:, :, 2] + ob_hot[:, :, 1]  # rigid + wood walls
        ob_hot = np.delete(ob_hot, [2], axis=2)
        # replace former fog channel with bomb blast strength
        ob_hot[:, :, 5] = ob_bomb_blast_strength
    else:
        # insert bomb blast strength next to bomb life
        ob_hot = np.insert(ob_hot, 4, ob_bomb_blast_strength, axis=2)

    self_ammo = make_np_float([obs["ammo"]])
    self_blast_strength = make_np_float([obs["blast_strength"]])
    self_can_kick = make_np_float([obs["can_kick"]])

    ob_hot = ob_hot.transpose((2, 0, 1))  # PyTorch tensor layout compat

    if config['rescale']:
        ob_hot = _rescale(ob_hot)
        self_ammo = _rescale(self_ammo / 10)
        self_blast_strength = _rescale(self_blast_strength / pommerman.constants.AGENT_VIEW_SIZE)
        self_can_kick = _rescale(self_can_kick)

    return np.concatenate([
        np.reshape(ob_hot, -1), self_ammo, self_blast_strength, self_can_kick])


class PommermanEnvWrapper(gym.Wrapper):
    #print('*****')
    def __init__(self, env=None, feature_config=DEFAULT_FEATURE_CONFIG):
        super(PommermanEnvWrapper, self).__init__(env)

        self.feature_config = feature_config
        if len(feature_config):
            self._set_observation_space(channels=get_feature_channels(feature_config))

    def _set_observation_space(self, channels):
        # The observation space cannot contain multiple tensors due to compat issues
        # with the way storages, normalizers, etc work. Thus, easiest to flatten everything and restore
        # shape in NN model

        bs = self.env._board_size
        obs_unflat = get_unflat_obs_space(channels, bs, self.feature_config['rescale'])
        min_flat_obs = np.concatenate([obs_unflat.spaces[0].low.flatten(), obs_unflat.spaces[1].low])
        max_flat_obs = np.concatenate([obs_unflat.spaces[0].high.flatten(), obs_unflat.spaces[1].high])

        self.observation_space = gym.spaces.Box(min_flat_obs, max_flat_obs)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def step(self, actions):
        #print('actions0_pom2_step',actions) #[3 4]

        # print('action_step',actions)
        # print('type_a0',type(actions))
        obs = self.env.get_observations()

        all_actions = actions.tolist()
        true_opp=self.env.act(obs)

        all_actions += self.env.act(obs) #[2, 4, 5, 5]

        #print('all_actions',all_actions)


        state, reward, done, _ = self.env.step(all_actions) #len(state)=4




        #print('reward!',reward) #reward! [0, 0, 0, 0]
        #print('reward',type(reward)) # <class 'list'>


        if not self.feature_config:

            agent_state = self.env.featurize(state[self.env.training_agent])
        else:


            agent_state = []
            for agent_id in self.env.training_agent:
               # print('agent_id',agent_id) #agent_id = 0,2


                agent_state0 = featurize(
                state[agent_id],
                agent_id,
                    self.feature_config)
                agent_state.append(agent_state0) # len(agent_state)=2; <class 'list'>

        agent_reward = []
        for agent_id in self.env.training_agent:
            agent_reward0 = reward[agent_id]
            agent_reward.append(agent_reward0) #agent_reward [0, 0]; <class 'list'>
        #print('agent_state',agent_state)




        return agent_state, agent_reward[0], done, {},true_opp

    def reset(self):
        #print('test_pom2')
        obs = self.env.reset()
       # print('self.feature_config',self.feature_config)
        if not self.feature_config:
            #print('agent_obs!')
            agent_obs = self.env.featurize(obs[self.env.training_agent])
        else:
            #print('agent_obs!!!')
            agent_obs=[]
            for id in self.env.training_agent:
                agent_obs.append(featurize(
                    obs[id],
                    id,
                    self.feature_config))

        return agent_obs


class TrainingAgent(pommerman.agents.BaseAgent):

    def __init__(self, character=pommerman.characters.Bomber):
        super(TrainingAgent, self).__init__(character)

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions."""
        return None


def _ffa_partial_env():
    """Start up a FFA config with the competition settings."""
    env = pommerman.envs.v0.Pomme
    game_type = pommerman.constants.GameType.FFA
    env_entry_point = 'pommerman.envs.v0:Pomme'
    env_id = 'PommeFFACompetition-v0'
    env_kwargs = {
        'game_type': game_type,
        'board_size': pommerman.constants.BOARD_SIZE,
        'num_rigid': pommerman.constants.NUM_RIGID,
        'num_wood': pommerman.constants.NUM_WOOD,
        'num_items': pommerman.constants.NUM_ITEMS,
        'max_steps': pommerman.constants.MAX_STEPS,
        'render_fps': pommerman.constants.RENDER_FPS,
        'agent_view_size': pommerman.constants.AGENT_VIEW_SIZE,
        'is_partially_observable': True,
        'env': env_entry_point,
    }
    agent = pommerman.characters.Bomber
    return locals()


def _ffa_partial_fast_env():
    """Start up a FFA config with the competition settings."""
    env = pommerman.envs.v0.Pomme
    game_type = pommerman.constants.GameType.FFA
    env_entry_point = 'pommerman.envs.v0:Pomme'
    env_id = 'PommeFFACompetitionFast-v0'
    env_kwargs = {
        'game_type': game_type,
        'board_size': pommerman.constants.BOARD_SIZE,
        'num_rigid': pommerman.constants.NUM_RIGID,
        'num_wood': pommerman.constants.NUM_WOOD,
        'num_items': pommerman.constants.NUM_ITEMS,
        'max_steps': pommerman.constants.MAX_STEPS,
        'render_fps': 1000,
        'agent_view_size': pommerman.constants.AGENT_VIEW_SIZE,
        'is_partially_observable': True,
        'env': env_entry_point,
    }
    agent = pommerman.characters.Bomber
    return locals()
def _team_partial_fast_env():
    """Start up a FFA config with the competition settings."""
    env = pommerman.envs.v01.Pomme
    game_type = pommerman.constants.GameType.Team
    env_entry_point = 'pommerman.envs.v01:Pomme'
    env_id = 'PommeTeamCompetitionFast-v01'
    env_kwargs = {
        'game_type': game_type,
        'board_size': pommerman.constants.BOARD_SIZE,
        'num_rigid': pommerman.constants.NUM_RIGID,
        'num_wood': pommerman.constants.NUM_WOOD,
        'num_items': pommerman.constants.NUM_ITEMS,
        'max_steps': pommerman.constants.MAX_STEPS,
        'render_fps': 1000,
        'agent_view_size': pommerman.constants.AGENT_VIEW_SIZE,
        'is_partially_observable': True,
        'env': env_entry_point,
    }
    agent = pommerman.characters.Bomber
    return locals()


def make_env(config):

    #training_agent = TrainingAgent()
    training_agent1 = TrainingAgent()
    training_agent2 = TrainingAgent()
    '''agent_list = [
        training_agent,
        pommerman.agents.SimpleAgent(),
        pommerman.agents.SimpleAgent(),
        pommerman.agents.SimpleAgent(),
    ]'''
    idx = 0
    team_id =2
    agent_list = [
        training_agent1,
        pommerman.agents.SimpleAgent(),
        training_agent2,
        #pommerman.agents.SimpleAgent(),
        pommerman.agents.SimpleAgent(),
    ]
    agent_list[idx] =  training_agent1
    agent_list[team_id] = training_agent2

    if config == "PommeFFAPartialFast-v0":

        env_spec = _ffa_partial_fast_env()
        env = pommerman.envs.v0.Pomme(**env_spec['env_kwargs'])
        for id, agent in enumerate(agent_list):
            assert isinstance(agent, pommerman.agents.BaseAgent)
            # NOTE: This is IMPORTANT so that the agent character is initialized
            agent.init_agent(id, env_spec['game_type'])
        env.set_agents(agent_list)
        #env.set_init_game_state(game_state_file)
        #env.set_render_mode(render_mode)
    elif config == "PommeTeamCompetitionFast-v01":

        env_spec = _team_partial_fast_env()
        env = pommerman.envs.v01.Pomme(**env_spec['env_kwargs'])

        for id, agent in enumerate(agent_list):
            assert isinstance(agent, pommerman.agents.BaseAgent)
            # NOTE: This is IMPORTANT so that the agent character is initialized
            agent.init_agent(id, env_spec['game_type'])
        env.set_agents(agent_list)


    else:
        env = pommerman.make(config, agent_list)
    env.set_training_agent([training_agent1.agent_id,training_agent2.agent_id])
    env.set_init_game_state(None)



    return PommermanEnvWrapper(env)