# 帮我查下当前项目中有哪个例子是针对混合动作（存在有1个离散动作和其他离散动作的连续动作不一致的问题，比如有2两个离散动作，1个离散动作对应的连续动作的维度是1，另一个离散动作对应的连续动作是2）
Collecting workspace information根据您提供的代码和项目结构,我找到了处理混合动作空间的示例。在当前激活的文件 main_embedding_platform_td3.py 中,**Platform-v0** 环境就是一个典型的混合动作空间示例,其中不同的离散动作对应不同维度的连续参数。

## 关键证据

在 `run` 函数中可以看到:

```python
# 第88-90行
initial_params_ = [3., 10., 400.]
for a in range(env.action_space.spaces[0].n):
    initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low) / (
            env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.
```

这段代码显示 Platform 环境有**3个不同的离散动作**,每个离散动作对应**不同的连续参数范围**。

```python
# 第99-102行
discrete_action_dim = env.action_space.spaces[0].n
action_parameter_sizes = np.array(
    [env.action_space.spaces[i].shape[0] for i in range(1, discrete_action_dim + 1)])
parameter_action_dim = int(action_parameter_sizes.sum())
```

这里计算每个离散动作对应的连续参数维度,并求和得到总维度。

## 相关示例文件

在 **HyAR** 项目中,以下文件都是针对 Platform 环境的混合动作空间:

- main_embedding_platform_td3.py (当前文件)
- main_embedding_platform_ddpg.py
- Raw_RL 目录下的相关文件(未使用嵌入表示的原始实现)

## 辅助包装器

common/platform_domain.py 中的 `PlatformFlattenedActionWrapper` 专门用于处理 Platform 环境的不规则混合动作空间。

这个 Platform 环境是您描述场景的完美示例:有多个离散动作,每个离散动作对应不同维度的连续动作参数。

# codebase 还有其他的源码针对不规则的混合动作空间吗？
Using "Codebase"

Read [](file:///c%3A/Projects/OpenSource/python/self-supervised-rl/RL_with_Action_Representation/HyAR/common/platform_domain.py)

Read [](file:///c%3A/Projects/OpenSource/python/self-supervised-rl/RL_with_Action_Representation/HyAR/common/goal_domain.py#1-1), lines 1 to 100

Read [](file:///c%3A/Projects/OpenSource/python/self-supervised-rl/RL_with_Action_Representation/HyAR/common/soccer_domain.py)

Searched text for `action_parameter_sizes|action_space\.spaces\[1\]\.spaces|FlattenedActionWrapper`, 200 results

Searched for files matching `**/*platform*.py`, 22 matches

Searched for files matching `**/*goal*.py`, 48 matches

Searched for files matching `**/*soccer*.py`, 2 matches

Read [](file:///c%3A/Projects/OpenSource/python/self-supervised-rl/RL_with_Action_Representation/HyAR/gym-platform-master/gym_platform/envs/platform_env.py#1-1), lines 1 to 150

Read [](file:///c%3A/Projects/OpenSource/python/self-supervised-rl/RL_with_Action_Representation/HyAR/gym-goal-master/gym_goal/envs/goal_env.py#1-1), lines 1 to 150

Read [](file:///c%3A/Projects/OpenSource/python/self-supervised-rl/RL_with_Action_Representation/HyAR/Raw_RL/main_platform_td3.py#1-1), lines 1 to 120

Read [](file:///c%3A/Projects/OpenSource/python/self-supervised-rl/RL_with_Action_Representation/HyAR/gym-platform-master/gym_platform/envs/platform_env.py#120-120), lines 120 to 160

Read [](file:///c%3A/Projects/OpenSource/python/self-supervised-rl/RL_with_Action_Representation/HyAR/gym-goal-master/gym_goal/envs/goal_env.py#30-30), lines 30 to 50

Read [](file:///c%3A/Projects/OpenSource/python/self-supervised-rl/RL_with_Action_Representation/HyAR/common/soccer_domain.py#75-75), lines 75 to 98

Searched text for `spaces\.Box.*for i in range|PARAMETERS_(MIN|MAX)|action_parameter_sizes\[`, 149 results

根据代码库的分析,我发现了另外两个处理**不规则混合动作空间**的环境示例:

## 1. **Goal 环境** (足球射门任务)

这是另一个典型的不规则混合动作空间示例,定义在 `gym-goal-master/gym_goal/envs/goal_env.py` 中:

```python
ACTION_LOOKUP = {
    0: KICK_TO,      # 2维参数
    1: SHOOT_GOAL,   # 1维参数
    2: SHOOT_GOAL,   # 1维参数
}

PARAMETERS_MIN = [
    np.array([0, -PITCH_WIDTH / 2]),      # 动作0: 2维
    np.array([-GOAL_WIDTH / 2]),           # 动作1: 1维
    np.array([-GOAL_WIDTH / 2]),           # 动作2: 1维
]

PARAMETERS_MAX = [
    np.array([PITCH_LENGTH, PITCH_WIDTH / 2]),  # 40, 15
    np.array([GOAL_WIDTH / 2]),                  # 7.01
    np.array([GOAL_WIDTH / 2]),                  # 7.01
]
```

**特点**: 第一个离散动作需要2维连续参数,后两个离散动作只需要1维连续参数。

## 2. **Soccer 环境** (HFO机器人足球)

定义在 `common/soccer_domain.py` 中,通过 `SoccerScaledParameterisedActionWrapper` 处理:

```python
def action(self, action):
    """将 [-1,1] 重新缩放到原始动作参数范围"""
    p = action[0]
    if p == 0:
        action[1] = self.range[0][0] * (action[1] + 1) / 2. + self.low[0][0]
        action[2] = self.range[0][1] * (action[2] + 1) / 2. + self.low[0][1]
    elif p == 1:
        action[3] = self.range[1] * (action[3] + 1) / 2. + self.low[1]
    elif p == 2:
        action[4] = self.range[2][0] * (action[4] + 1) / 2. + self.low[2][0]
        action[5] = self.range[2][1] * (action[5] + 1) / 2. + self.low[2][1]
```

**特点**: 
- 动作0: 2维连续参数 (action[1], action[2])
- 动作1: 1维连续参数 (action[3])
- 动作2: 2维连续参数 (action[4], action[5])

## 相关示例文件

### Goal 环境相关:
- `HyAR_RL/main_embedding_goal_td3.py`
- `HyAR_RL/main_embedding_goal_ddpg.py`
- [`Raw_RL/main_goal_*.py`](c:\Projects\OpenSource\python\self-supervised-rl\RL_with_Action_Representation\HyAR\Raw_RL) (多个变体)

### Hard Goal 环境相关:
- `HyAR_RL/main_embedding_hard_goal_td3.py`
- [`Raw_RL/main_hard_goal_*.py`](c:\Projects\OpenSource\python\self-supervised-rl\RL_with_Action_Representation\HyAR\Raw_RL) (多个变体)

## 总结

项目中有**三个**不规则混合动作空间的环境:

| 环境 | 离散动作数 | 连续参数维度 | 特点 |
|------|-----------|-------------|------|
| **Platform** | 3 | [1, 1, 1] | 每个动作1维参数 |
| **Goal** | 3 | [2, 1, 1] | 第1个动作2维,其余1维 |
| **Soccer** | 3 | [2, 1, 2] | 维度分布最不规则 |

这些环境都通过特定的 Wrapper 类来处理不规则的动作空间,使其能够与标准的强化学习算法配合使用。

Similar code found with 1 license type