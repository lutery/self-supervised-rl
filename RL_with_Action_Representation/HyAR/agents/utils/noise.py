import numpy as np


class OrnsteinUhlenbeckActionNoise(object):
    """
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    Source: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/utils.py
    # todo 注意如果是开启了这个，则需要保存这里的探索状态，在每个episode开始时reset
    """

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2, random_machine=np.random):
        '''
        action_dim: 所有连续动作参数的总维度
        random_machine: 随机数生成器，目前是None
        mu、theta、sigma： Ornstein-Uhlenbeck过程的参数，分别表示均值、速度和噪声强度
        '''
        super(OrnsteinUhlenbeckActionNoise, self).__init__()
        self.random = random_machine
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * self.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X
