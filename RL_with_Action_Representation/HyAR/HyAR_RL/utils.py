import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim,all_parameter_action_dim ,discrete_emb_dim, parameter_emb_dim,
                 max_size=int(1e6)):
        '''
        state_dim: 观察的维度
        discrete_action_dim： 离散动作的
        parameter_action_dim： 存储连续动作的维度
        all_parameter_action_dim： todo
        discrete_emb_dim： todo
        parameter_emb_dim： todo
        max_size：todo
        '''
        self.max_size = max_size # 缓冲区的大小
        self.ptr = 0 # 缓冲区的位置索引
        self.size = 0 # 缓冲区目前的样本数量

        self.state = np.zeros((max_size, state_dim)) # 环境的观察
        self.discrete_action = np.zeros((max_size, discrete_action_dim)) # 离散动作
        self.parameter_action = np.zeros((max_size, parameter_action_dim)) # 离散动作对应的连续动作的值
        self.all_parameter_action = np.zeros((max_size, all_parameter_action_dim)) #  所有连续动作的值

        self.discrete_emb = np.zeros((max_size, discrete_emb_dim)) # todo
        self.parameter_emb = np.zeros((max_size, parameter_emb_dim)) # todo
        self.next_state = np.zeros((max_size, state_dim)) # 执行动作后的下一个状态
        self.state_next_state = np.zeros((max_size, state_dim)) # 新state和旧state之间的差值

        self.reward = np.zeros((max_size, 1)) # 奖励
        self.not_done = np.zeros((max_size, 1)) # 是否没有结束

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 执行的设备

    def add(self, state, discrete_action, parameter_action, all_parameter_action,discrete_emb, parameter_emb, next_state,state_next_state, reward, done):
        '''
        state: 环境的观察
        discrtte_action: 离散动作
        parameter_action： 离散动作对应的连续动作的值
        all_parameter_action： 所有连续动作的值
        discrete_emb： 离散动作的维度，有环境传入的是None / 在训练时传入离散动作的嵌入向量 
        parameter_emb：传入连续动作的嵌入向量
        next_state： 执行动作后的下一个状态
        state_next_state：新state和旧state之间的差值
        reward：奖励
        done：是否结束

        '''

        '''
        在replay_buffer_embedding中
        state: 环境的观察
        discrtte_action: 离散动作
        parameter_action： 离散动作对应的连续动作的值
        all_parameter_action： 所有连续动作的值
        discrete_emb： None
        parameter_emb：None
        next_state： 执行动作后的下一个状态
        state_next_state：新state和旧state之间的差值
        reward：奖励
        done：是否结束

        '''
        self.state[self.ptr] = state
        self.discrete_action[self.ptr] = discrete_action
        self.parameter_action[self.ptr] = parameter_action
        self.all_parameter_action[self.ptr] = all_parameter_action
        self.discrete_emb[self.ptr] = discrete_emb
        self.parameter_emb[self.ptr] = parameter_emb
        self.next_state[self.ptr] = next_state
        self.state_next_state[self.ptr] = state_next_state

        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # 随机采样不连续的batch_size个样本
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.discrete_action[ind]).to(self.device),
            torch.FloatTensor(self.parameter_action[ind]).to(self.device),
            torch.FloatTensor(self.all_parameter_action[ind]).to(self.device),
            torch.FloatTensor(self.discrete_emb[ind]).to(self.device),
            torch.FloatTensor(self.parameter_emb[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.state_next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
