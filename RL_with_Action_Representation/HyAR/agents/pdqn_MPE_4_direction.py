# simple_move
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import Counter
from torch.autograd import Variable

from agents.agent import Agent
from agents.memory.memory import Memory
from agents.utils import soft_update_target_network, hard_update_target_network
from agents.utils.noise import OrnsteinUhlenbeckActionNoise

import matplotlib.pyplot as plt

class QActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=(100,), action_input_layer=0,
                 output_layer_init_std=None, activation="relu", **kwargs):
        '''
        todo 这里应该是输出每一个离散动作的Q值

        state_size: 环境观察的维度
        action_size: 离散动作的维度
        action_parameter_size: 连续动作的维度
        hidden_layers: 隐藏层的维度
        action_input_layer: 动作参数输入层的位置，0表示输入层，-1 todo
        output_layer_init_std: todo
        activateion: 激活函数
        '''
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size # 看来也是是输入了环境观察和连续动作维度
        lastHiddenLayerSize = inputSize # 存储最近一层的输入维度
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))  #输出为action_size维 输出离散动作维度维度？

        # initialise layer weights 初始化权重
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        # else:
        #     nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state, action_parameters):
        '''
        state: 环境的观察
        action_parameters: 预测的连续动作
        '''
        # implement forward
        negative_slope = 0.01

        x = torch.cat((state, action_parameters), dim=1)  #将两个张量按列拼接
        # 将观察和预测的连续动作拼接起来，然后输入网络预测离散动作的Q值
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        Q = self.layers[-1](x)
        return Q


class ParamActor(nn.Module):
    '''
    混合动作空间的连续部分生成
    ParamActor在本仓库的HyAR/PDQN系算法中承担“连续参数动作头”的角色：给定状态，产出归一化到[-1,1]的连续动作参数，并与离散动作选择头(QActor)协同完成混合动作空间的决策
    Collecting workspace informationParamActor在本仓库的HyAR/PDQN系算法中承担“连续参数动作头”的角色：给定状态，产出归一化到[-1,1]的连续动作参数，并与离散动作选择头(QActor)协同完成混合动作空间的决策。

    - 产出什么
    - 输入: 状态向量（部分变体还会拼接离散动作或其嵌入）
    - 输出: 连续动作参数向量 $a_c \in [-1,1]^{d}$，用于与离散动作索引组合成完整动作
    - 代码见：
        - `pdqn_MPE_4_direction.ParamActor`
        - `pdqn_MPE_direction_catch.ParamActor`
        - TD3/DDPG 变体同名类：如`pdqn_td3_MPE.ParamActor`、`pdqn.ParamActor`

    - 如何协同决策
    - ParamActor先给出参数动作 $a_c$，QActor再计算 $Q(s, a_c)$ 并为每个离散动作给出Q值，选取argmax离散动作。对应实现参考：
        - `pdqn_MPE_direction_catch.QActor.forward`（将state与action_parameters拼接评估各离散动作Q）
        - 这两者在Agent中一起实例化与优化，例如`pdqn_MPE_4_direction.PDQNAgent.__init__`

    - 重要实现细节
    - 多数实现含“直通层”(passthrough layer)：将状态线性映射并直接加到输出上以稳定训练，且权重冻结，见`pdqn_MPE_4_direction.ParamActor`与`pdqn_MPE_direction_catch.ParamActor`
    - 输出范围与包裹器一致：环境侧用包装器把参数动作缩放到[-1,1]，参见`common.wrappers.QPAMDPScaledParameterisedActionWrapper`
    - 训练时结合Q网络梯度对ParamActor做确定性策略梯度更新，必要时用“反向梯度”保持参数在边界内（参照各PDQN Agent中的`_invert_gradients`，如`pdqn_MPE_4_direction.PDQNAgent`）

    - 与HyAR嵌入的关系
    - 在HyAR的embedding训练流程中，ParamActor常作为预策略生成连续参数数据并填充经验池；而执行阶段也可能通过VAE解码得到参数动作（见`HyAR/embedding/ActionRepresentation_vae.py`的decode接口被上层调用）。总体目标一致：为离散动作提供其对应的连续参数，从而完成混合动作控制。
    '''

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers, squashing_function=False,
                 output_layer_init_std=None, init_type="kaiming", activation="relu", init_std=None):
        '''
        state_size: 环境观察
        action_size: 离散动作空间维度
        action_parameter_size: 连续动作参数维度
        hidden_layers: 隐藏层的层数
        squashing_fcuntion: 用于将网络的输出值约束到指定的范围，但是由于作者无法正确的实现，所以暂时禁用，无用的参数
        output_layer_init_std: todo
        init_type: 初始化类型 用于初始化权重 
        activation: 激活函数
        init_std: todo

        '''
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.squashing_function = squashing_function
        self.activation = activation
        if init_type == "normal":
            assert init_std is not None and init_std > 0
        assert self.squashing_function is False  # unsupported, cannot get scaling right yet

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            # 构造隐藏层
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        # 构造输出连续动作的预测层
        self.action_parameters_output_layer = nn.Linear(lastHiddenLayerSize, self.action_parameter_size)
        # 对这个做特殊处理，不参与梯度计算
        # 初始权重为0
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)

        # initialise layer weights
        # 初始化权重
        for i in range(0, len(self.layers)):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            elif init_type == "normal":
                nn.init.normal_(self.layers[i].weight, std=init_std)
            else:
                raise ValueError("Unknown init_type "+str(init_type))
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.action_parameters_output_layer.weight, std=output_layer_init_std)
        else:
            nn.init.zeros_(self.action_parameters_output_layer.weight)
        nn.init.zeros_(self.action_parameters_output_layer.bias)

        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)

        # fix passthrough layer to avoid instability, rest of network can compensate
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        '''
        state: 输入观察
        '''
        x = state
        negative_slope = 0.01 # 根据经验选择的一个leakRelu的参数
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):
            # 遍历每一层，根据不同的激活函数选择不同的参数
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        action_params = self.action_parameters_output_layer(x) # 经过特征采集后输出的连续动作

        # print("action_params",action_params)
        # 这是一个残差连接的变体，类似ResNet中的skip connection。直通层确保即使主网络梯度消失，仍有稳定的梯度路径。
        action_params += self.action_parameters_passthrough_layer(state) # 不考虑特征采集，直接传入state后，预测输出的值加入action_params
        # print("action_params",action_params)
        if self.squashing_function:
            # 没有实现，主要目标就是要实现将参数约束到指定的范围内
            assert False  # scaling not implemented yet
            action_params = action_params.tanh()
            action_params = action_params * self.action_param_lim
        # action_params = action_params / torch.norm(action_params) ## REMOVE --- normalisation layer?? for pointmass
        return action_params


class PDQNAgent(Agent):
    """
    DDPG actor-critic agent for parameterised action spaces
    [Hausknecht and Stone 2016]
    """

    NAME = "P-DQN Agent"

    def __init__(self,
                 observation_space,
                 action_space,
                 parameter_action_dim,
                 actor_class=QActor,
                 actor_kwargs={},
                 actor_param_class=ParamActor,
                 actor_param_kwargs={},
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_steps=10000,
                 batch_size=64,
                 gamma=0.99,
                 tau_actor=0.01,  # Polyak averaging factor for copying target weights
                 tau_actor_param=0.001,
                 replay_memory_size=1000000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.00001,
                 initial_memory_threshold=0,
                 use_ornstein_noise=False,  # if false, uses epsilon-greedy with uniform-random action-parameter exploration
                 loss_func=F.mse_loss, # F.mse_loss
                 clip_grad=10,
                 inverting_gradients=False,
                 zero_index_gradients=False,
                 indexed=False,
                 weighted=False,
                 average=False,
                 random_weighted=False,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None):

        super(PDQNAgent, self).__init__(observation_space, action_space)
        self.device = torch.device(device)
        self.num_actions = action_space
        # self.action_parameter_sizes = [4,0]
        self.parameter_action_dim=parameter_action_dim
        # self.action_parameter_sizes = np.array([4, 0])
        self.action_parameter_size = parameter_action_dim

        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = -self.action_max.detach()

        self.action_range = (self.action_max-self.action_min).detach()

        self.action_parameter_max_numpy = np.ones((self.parameter_action_dim,))
        # self.action_parameter_min_numpy = np.zeros((self.parameter_action_dim,))
        self.action_parameter_min_numpy = -self.action_parameter_max_numpy

        print("self.action_parameter_max_numpy",self.action_parameter_max_numpy,self.action_parameter_min_numpy)
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        self.indexed = indexed
        self.weighted = weighted
        self.average = average
        self.random_weighted = random_weighted

        assert (weighted ^ average ^ random_weighted) or not (weighted or average or random_weighted)



        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_actor_param = learning_rate_actor_param
        self.inverting_gradients = inverting_gradients
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param
        self._step = 0
        self._episode = 0
        self.updates = 0
        self.clip_grad = clip_grad
        self.zero_index_gradients = zero_index_gradients

        self.np_random = None
        self.seed = seed
        self._seed(seed)

        self.use_ornstein_noise = use_ornstein_noise
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, random_machine=self.np_random, mu=0., theta=0.15, sigma=0.0001) #, theta=0.01, sigma=0.01)

        print(self.num_actions+self.action_parameter_size)
        print(observation_space[0][0])
        self.replay_memory = Memory(replay_memory_size, observation_space[0], (1+self.action_parameter_size,), next_actions=False)
        # 构造了一个离散动作的Q值预测网络
        self.actor = actor_class(self.observation_space[0][0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        self.actor_target = actor_class(self.observation_space[0][0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        hard_update_target_network(self.actor, self.actor_target) # 同步权重
        self.actor_target.eval() # 设置为验证模式，不存储梯度

        # 构造一个连读动作的动作预测网络
        self.actor_param = actor_param_class(self.observation_space[0][0], self.num_actions, self.action_parameter_size, **actor_param_kwargs).to(device)
        self.actor_param_target = actor_param_class(self.observation_space[0][0], self.num_actions, self.action_parameter_size, **actor_param_kwargs).to(device)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval() # 设置为验证模式，不存储梯度

        self.loss_func = loss_func  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor) #, betas=(0.95, 0.999))
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param) #, betas=(0.95, 0.999)) #, weight_decay=critic_l2_reg)
        self.cost_his = []
    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
                "Inverting Gradients: {}\n".format(self.inverting_gradients) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Initial memory: {}\n".format(self.initial_memory_threshold) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_steps: {}\n".format(self.epsilon_steps) + \
                "Clip Grad: {}\n".format(self.clip_grad) + \
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) + \
                "Zero Index Grads?: {}\n".format(self.zero_index_gradients) + \
                "Seed: {}\n".format(self.seed)
        return desc

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        passthrough_layer = self.actor_param.action_parameters_passthrough_layer

        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.device)
        if initial_bias is not None:
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.device)
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update_target_network(self.actor_param, self.actor_param_target)

    def _seed(self, seed=None):
        """
        NOTE: this will not reset the randomly initialised weights; use the seed parameter in the constructor instead.

        :param seed:
        :return:
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == torch.device("cuda"):
                torch.cuda.manual_seed(seed)

    def _ornstein_uhlenbeck_noise(self, all_action_parameters):
        """ Continuous action exploration using an Ornstein–Uhlenbeck process. """
        return all_action_parameters.data.numpy() + (self.noise.sample() * self.action_parameter_range_numpy)

    def start_episode(self):
        pass

    def end_episode(self):
        self._episode += 1

        ep = self._episode
        if ep < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    ep / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def act(self, state):
        '''
        state: 环境观察
        return: action: 离散动作索引；action_parameters: 该离散动作对应的连续动作参数（但是在当前的环境中，action_parameters=all_action_parameters）；all_action_parameters: 所有离散动作对应的连续动作参数
        '''
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)

            all_action_parameters = self.actor_param.forward(state)

            # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
            # 是随机选择一个动作还是使用actor模型预测一个动作
            rnd = self.np_random.uniform()
            if rnd < 1.0:
                action = self.np_random.choice(self.num_actions)
                if not self.use_ornstein_noise:
                    all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                              self.action_parameter_max_numpy))
            else:
                # 使用模型预测，输入状态和连续动作，输出每个离散动作的Q值
                Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                action = np.argmax(Q_a)  #返回最大离散动作的索引



            # add noise only to parameters of chosen action
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            # offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
            # if self.use_ornstein_noise and self.noise is not None:
            #     all_action_parameters[0:4] += self.noise.sample()[0:4]


            # 意思就是说all_action_parameters是所有离散动作对应的连续值，action_parameters是根据action离散动作选择的连续动作的值，比如有些离散动作是互斥的，所以存在需要将all_action_parameters中的值提取出指定的部分
            action_parameters = all_action_parameters

        return action, action_parameters, all_action_parameters

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):
        assert grad.shape[0] == batch_action_indices.shape[0]
        grad = grad.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            ind = torch.zeros(self.action_parameter_size, dtype=torch.long)
            for a in range(self.num_actions):
                ind[self.action_parameter_offsets[a]:self.action_parameter_offsets[a+1]] = a
            # ind_tile = np.tile(ind, (self.batch_size, 1))
            ind_tile = ind.repeat(self.batch_size, 1).to(self.device)
            actual_index = ind_tile != batch_action_indices[:, np.newaxis]
            grad[actual_index] = 0.
        return grad
# P-DDPG
    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '"+str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def step(self, state, action, reward, next_state, next_action, terminal):
        act, all_action_parameters = action
        self._step += 1
        # a=np.concatenate(([act],all_action_parameters)).ravel()


        # self._add_sample(state, np.concatenate((all_actions.data, all_action_parameters.data)).ravel(), reward, next_state, terminal)
        self._add_sample(state, np.concatenate(([act],all_action_parameters)).ravel(), reward, next_state, np.concatenate(([next_action[0]],next_action[1])).ravel(), terminal=terminal)

        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()
            self.updates += 1

    def _add_sample(self, state, action, reward, next_state, next_action, terminal):
        assert len(action) == 1 + self.action_parameter_size
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal)

    def _optimize_td_loss(self):
        if self._step < self.batch_size or self._step < self.initial_memory_threshold:
            return
        # Sample a batch from replay memory
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)

        states = torch.from_numpy(states).to(self.device)

        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        # print("dfgdfgfgdn", terminals)
        # ---------------------- optimize Q-network ----------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()
            # Compute the TD error
            target = rewards + (1 - terminals) * self.gamma * Qprime
        # Compute current Q-values using policy network
        q_values = self.actor(states, action_parameters)

        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad()   #清空上一步残留参数值
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()   # 将参数更新值施加到 net 的 parameters 上

        # ---------------------- optimize actor ----------------------
        with torch.no_grad():
            action_params = self.actor_param(states)

        action_params.requires_grad = True
        assert (self.weighted ^ self.average ^ self.random_weighted) or \
               not (self.weighted or self.average or self.random_weighted)
        Q = self.actor(states, action_params)
        Q_val = Q
        if self.weighted:
            # approximate categorical probability density (i.e. counting)
            counts = Counter(actions.cpu().numpy())
            weights = torch.from_numpy(
                np.array([counts[a] / actions.shape[0] for a in range(self.num_actions)])).float().to(self.device)
            Q_val = weights * Q
        elif self.average:
            Q_val = Q / self.num_actions
        elif self.random_weighted:
            weights = np.random.uniform(0, 1., self.num_actions)
            weights /= np.linalg.norm(weights)
            weights = torch.from_numpy(weights).float().to(self.device)
            Q_val = weights * Q
        if self.indexed:
            Q_indexed = Q_val.gather(1, actions.unsqueeze(1))
            Q_loss = torch.mean(Q_indexed)
        else:
            Q_loss = torch.mean(torch.sum(Q_val, 1))   #求和再取均值
        # print('Q')
        # print(Q_loss)
        x=Q_loss
        x_np = x.data.cpu().numpy()
        self.cost_his.append(x_np)

        self.actor.zero_grad()
        Q_loss.backward()

        from copy import deepcopy
        # print('-------------')
        # print(action_params.grad.data)
        delta_a = deepcopy(action_params.grad.data)
        # step 2
        action_params = self.actor_param(Variable(states))

        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)

        if self.zero_index_gradients:
            delta_a[:] = self._zero_index_gradients(delta_a, batch_action_indices=actions, inplace=True)

        out = -torch.mul(delta_a, action_params)
        # print(out)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

        self.actor_param_optimiser.step()

        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)

    def save_models(self, prefix):
        """
        saves the target actor and critic models
        :param prefix: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), prefix + '_actor.pt')
        torch.save(self.actor_param.state_dict(), prefix + '_actor_param.pt')
        print('Models saved successfully')

    def load_models(self, prefix):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param prefix: the count of episodes iterated (used to find the file name)
        :param target: whether to load the target newtwork too (not necessary for evaluation)
        :return:
        """
        # also try load on CPU if no GPU available?
        self.actor.load_state_dict(torch.load(prefix + '_actor.pt', map_location='cpu'))
        self.actor_param.load_state_dict(torch.load(prefix + '_actor_param.pt', map_location='cpu'))
        print('Models loaded successfully')

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his, c='y', label='pdqn_actor_loss')
        plt.legend(loc='best')
        plt.ylabel('loss')
        plt.xlabel('Training Steps')
        plt.show()

