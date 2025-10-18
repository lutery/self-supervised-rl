import copy
import gym
import numpy as np
from gym.spaces import Tuple, Box


class ScaledStateWrapper(gym.ObservationWrapper):
    """
    Scales the observation space to [-1,1]
    看起来是要将观察缩放到[-1,1]范围内
    """

    def __init__(self, env):
        super(ScaledStateWrapper, self).__init__(env)
        obs = env.observation_space
        self.compound = False
        self.low = None
        self.high = None
        print(type(obs))
        print(obs)
        if isinstance(obs, gym.spaces.Box):
            # 根据不同的观察空间类型，提取不同的低高值
            self.low = env.observation_space.low # 观察空间的低值
            self.high = env.observation_space.high # 观察空间的高值
            self.observation_space = gym.spaces.Box(low=-np.ones(self.low.shape), high=np.ones(self.high.shape),
                                                    dtype=np.float32) # 创建一个新的观察空间，范围为[-1,1]
        elif isinstance(obs, Tuple):
            # 如果是元组类型的观察空间，假设第一个元素是Box类型，第二个元素是Discrete类型
            # todo 元组类型的第二个元素是干啥的？
            self.low = obs.spaces[0].low
            self.high = obs.spaces[0].high
            assert len(obs.spaces) == 2 and isinstance(obs.spaces[1], gym.spaces.Discrete)
            self.observation_space = Tuple(
                (gym.spaces.Box(low=-np.ones(self.low.shape), high=np.ones(self.high.shape),
                                dtype=np.float32),
                 obs.spaces[1]))
            self.compound = True # 这个是干啥的？
        else:
            raise Exception("Unsupported observation space type: %s" % self.observation_space)

    def scale_state(self, state):
        state = 2. * (state - self.low) / (self.high - self.low) - 1.
        return state

    def _unscale_state(self, scaled_state):
        state = (self.high - self.low) * (scaled_state + 1.) / 2. + self.low
        return state

    def observation(self, obs):
        if self.compound:
            state, steps = obs
            ret = (self.scale_state(state), steps)
        else:
            ret = self.scale_state(obs)
        return ret


class TimestepWrapper(gym.Wrapper):
    """
    Adds a timestep return to an environment for compatibility reasons.
    """

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        return state, 0

    def step(self, action):
        state, reward, terminal, info = self.env.step(action)
        obs = (state, 1)
        return obs, reward, terminal, info


class ScaledParameterisedActionWrapper(gym.ActionWrapper):
    """
    Changes the scale of the continuous action parameters to [-1,1].
    Parameter space must be flattened!

    看起来这里是将动作空间中的连续参数缩放到[-1,1]范围内
    这里假设动作空间是扁平化的，即参数空间不是嵌套的Tuple，而是直接展开的Box列表

    Tuple((
        Discrete(n),
        Box(c_1),
        Box(c_2),
        ...
        Box(c_n)
        )
    """

    def __init__(self, env):
        super(ScaledParameterisedActionWrapper, self).__init__(env)
        self.old_as = env.action_space # 获取旧的动作空间
        self.num_actions = self.old_as.spaces[0].n # 离散动作的数量
        self.high = [self.old_as.spaces[i].high for i in range(1, self.num_actions + 1)] # 每个离散动作对应的参数空间的高值
        self.low = [self.old_as.spaces[i].low for i in range(1, self.num_actions + 1)] # 每个离散动作对应的参数空间的低值
        self.range = [self.old_as.spaces[i].high - self.old_as.spaces[i].low for i in range(1, self.num_actions + 1)] # 每个参数空间的范围
        # 这里创建一个新的动作空间，将连续动作的参数空间的范围被缩放到[-1,1]
        new_params = [  # parameters
            Box(-np.ones(self.old_as.spaces[i].low.shape), np.ones(self.old_as.spaces[i].high.shape), dtype=np.float32)
            for i in range(1, self.num_actions + 1)
        ]
        # 构建新的动作空间
        # self.action_space[0]是离散动作空间
        # 后面的Box是缩放后的连续参数空间
        self.action_space = Tuple((
            self.old_as.spaces[0],  # actions
            *new_params,
        ))

    def action(self, action):
        """
        Rescale from [-1,1] to original action-parameter range.

        :param action:
        :return:
        """
        action = copy.deepcopy(action)
        p = action[0]
        action[1][p] = self.range[p] * (action[1][p] + 1) / 2. + self.low[p]
        return action


class QPAMDPScaledParameterisedActionWrapper(gym.ActionWrapper):
    """
    Changes the scale of the continuous action parameters to [-1,1].
    Parameter space not flattened in this case

    Tuple((
        Discrete(n),
        Tuple((
            Box(c_1),
            Box(c_2),
            ...
            Box(c_n)
            ))
        )
    """

    def __init__(self, env):
        super(QPAMDPScaledParameterisedActionWrapper, self).__init__(env)
        self.old_as = env.action_space
        self.num_actions = self.old_as.spaces[0].n
        self.high = [self.old_as.spaces[1][i].high for i in range(self.num_actions)]
        self.low = [self.old_as.spaces[1][i].low for i in range(self.num_actions)]
        self.range = [self.old_as.spaces[1][i].high - self.old_as.spaces[1][i].low for i in range(self.num_actions)]
        new_params = [  # parameters
            gym.spaces.Box(-np.ones(self.old_as.spaces[1][i].low.shape), np.ones(self.old_as.spaces[1][i].high.shape),
                           dtype=np.float32)
            for i in range(self.num_actions)
        ]
        self.action_space = gym.spaces.Tuple((
            self.old_as.spaces[0],  # actions
            gym.spaces.Tuple(tuple(new_params)),
        ))

    def action(self, action):
        """
        Rescale from [-1,1] to original action-parameter range.

        :param action:
        :return:
        """
        action = copy.deepcopy(action)
        p = action[0]
        action[1][p] = self.range[p] * (action[1][p] + 1) / 2. + self.low[p]
        return action
