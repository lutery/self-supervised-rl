import numpy as np
import gym


class PlatformFlattenedActionWrapper(gym.ActionWrapper):
    """
    Changes the format of the parameterised action space to conform to that of Goal-v0 and Platform-v0
    将 Platform 环境的动作空间从嵌套结构转换为扁平化结构,使其更容易被强化学习算法处理
    原始动作空间：
    # 原始动作空间结构
    old_as = Tuple(
        Discrete(3),           # 离散动作空间: [0, 1, 2]
        Tuple(                 # 连续参数空间(嵌套Tuple)
            Box([0], [30]),    # 动作0的参数范围
            Box([0], [720]),   # 动作1的参数范围  
            Box([0], [430])    # 动作2的参数范围
        )
    )

    # 修改后：
    # 扁平化后的动作空间结构
    new_as = Tuple(
        Discrete(3),           # 离散动作空间
        Box([0], [30]),        # 动作0的参数
        Box([0], [720]),       # 动作1的参数
        Box([0], [430])        # 动作2的参数
    )

    这种 Tuple(Discrete, Box, Box, Box) 的扁平结构符合 Goal-v0 和 Platform-v0 的标准格式
    """
    def __init__(self, env):
        super(PlatformFlattenedActionWrapper, self).__init__(env)
        old_as = env.action_space # 已有环境的动作空间
        # old_as.spaces[0]代表离散动作空间
        # old_as.spaces[1]代表连续参数空间，是一个包含多个Box空间的Tuple
        num_actions = old_as.spaces[0].n # 离散动作的数量
        # 看来以下是非常规的gym空间
        # 以下动作空间的元素包含离散动作的索引和对应的连续参数
        self.action_space = gym.spaces.Tuple((
            old_as.spaces[0],  # actions
            *(gym.spaces.Box(old_as.spaces[1].spaces[i].low, old_as.spaces[1].spaces[i].high, dtype=np.float32)
              for i in range(0, num_actions))
        ))

    def action(self, action):
        return action
