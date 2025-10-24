import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    '''
    预测输出的是实际的动作值
    '''
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim, max_action):
        '''
        state_dim: 环境的状态维度
        discrete_action_dim: 离散动作的嵌入维度
        parameter_action_dim: 连续动作的嵌入维度
        max_action: 最大动作的范围
        '''
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3_1 = nn.Linear(256, discrete_action_dim)
        self.l3_2 = nn.Linear(256, parameter_action_dim)

        self.max_action = max_action

    def forward(self, state):
        '''
        state： 环境的观察

        return: 预测的离散动作(根据上下文，这里输出的应该是离散动作的嵌入向量形式)和连续动作嵌入
        '''
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        discrete_action = self.max_action * torch.tanh(self.l3_1(a)) # 预测的离散动作
        parameter_action = self.max_action * torch.tanh(self.l3_2(a)) # 预测的连续动作的值
        return discrete_action, parameter_action


class Critic(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim):
        '''
        state_dim：环境的状态维度
        discrete_action_dim：离散动作的嵌入维度
        parameter_action_dim：连续动作的嵌入维度
        '''
        super(Critic, self).__init__()

        # Q1 architecture 
        # 输入环境+离散动过+连续动作预测Q值
        self.l1 = nn.Linear(state_dim + discrete_action_dim + parameter_action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        # 输入环境+离散动过+连续动作预测Q值
        self.l4 = nn.Linear(state_dim + discrete_action_dim + parameter_action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, discrete_action, parameter_action):
        sa = torch.cat([state, discrete_action, parameter_action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, discrete_action, parameter_action):
        sa = torch.cat([state, discrete_action, parameter_action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim, # 环境的状态维度
            discrete_action_dim, # 离散动作的嵌入维度
            parameter_action_dim, # 连续动作的嵌入维度
            max_action, # 最大动作的范围
            discount=0.99, # 折扣因子
            tau=0.005, # 目标模型软更新的系数
            policy_noise=0.2, # todo
            noise_clip=0.5, # todo
            policy_freq=2 # todo
    ):
        self.discrete_action_dim = discrete_action_dim
        self.parameter_action_dim = parameter_action_dim

        # 离散动作的范围
        self.action_max = torch.from_numpy(np.ones((self.discrete_action_dim,))).float().to(device) # shape is (discrete_action_dim,) 全1
        self.action_min = -self.action_max.detach() # shape is (discrete_action_dim,) 全-1
        self.action_range = (self.action_max - self.action_min).detach() # shape is (discrete_action_dim,) 全2

        # 连续动作的范围
        self.action_parameter_max = torch.from_numpy(np.ones((self.parameter_action_dim,))).float().to(device) # shape is (parameter_action_dim,) 全1
        self.action_parameter_min = -self.action_parameter_max.detach() # shape is (parameter_action_dim,) 全-1
        # print(" self.action_parameter_max_numpy", self.action_parameter_max)
        self.action_parameter_range = (self.action_parameter_max - self.action_parameter_min) # shape is (parameter_action_dim,) 全2

        # 动作嵌入预测网络
        self.actor = Actor(state_dim, discrete_action_dim, parameter_action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)  # 默认3e-4

        # 评测网络
        self.critic = Critic(state_dim, discrete_action_dim, parameter_action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0 # 训练的次数

    def select_action(self, state):
        '''
        state: 环境的观察

        return: 返回预测的所有离散动作的嵌入向量、返回预测的所有连续动作的值
        '''
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        all_discrete_action, all_parameter_action = self.actor(state)
        return all_discrete_action.cpu().data.numpy().flatten(), all_parameter_action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, action_rep, c_rate, recon_s_rate, batch_size=256):
        '''
        replay_buffer: 训练经验池
        action_rep: todo
        c_rate: 动作每个维度的范围
        recon_s_rate：这里传入的vae训练时的重建误差，这里用来判断状态重建的好坏，从而决定是否要重标记连续动作嵌入 todo
        '''
        recon_s_rate = recon_s_rate * 5.0 # 放大5倍作为阈值
        self.total_it += 1
        # Sample replay buffer 采样 离散 不连续
        # discrete_emb：对应真实离散动作的嵌入向量表示
        # state_next_state：表示新旧状态误差
        state, discrete_action, parameter_action, all_parameter_action, discrete_emb, parameter_emb, next_state, state_next_state, reward, not_done = replay_buffer.sample(
            batch_size)
        # print("discrete_emb----------",discrete_emb)
        with torch.no_grad():
            # 获取离散动作的嵌入表示
            discrete_emb_ = action_rep.get_embedding(discrete_action.reshape(1,
                                                                             -1).squeeze().long()).to(device)
            # discrete relable need noise 计算离散动作的噪音量 因为之前约束了符合标准正态分布，可以这么采样噪音
            noise_discrete = (
                    torch.randn_like(discrete_emb_) * 0.1
            ).clamp(-self.noise_clip, self.noise_clip)
            discrete_emb_table = discrete_emb_.clamp(-self.max_action, self.max_action) # 离散动作的嵌入表示，规范了范围
            discrete_emb_table_noise = (discrete_emb_ + noise_discrete).clamp(-self.max_action, self.max_action) # 增加了噪音的离散动作的嵌入表示

            # 因为vae训练次数比td3之类的动作预测模型少，所以vae还原的discrete_action_old属于old，而td3还原的discrete_action属于new
            discrete_action_old = action_rep.select_discrete_action(discrete_emb).reshape(-1, 1) # 将嵌入动作向量转换为离散动作的值
            d_new = discrete_action.cpu().numpy() 
            d_old = discrete_action_old
            d_bing = (d_new == d_old) * 1 # 这里是在计算什么？？计算新旧值之间是否匹配的bool矩阵
            # discrete_relable_rate
            discrete_relable_rate = sum(d_bing.reshape(1, -1)[0]) / batch_size # 计算实际动作和根据嵌入转换为离散动作之间总共有多少匹配的动作
            d_bing = torch.FloatTensor(d_bing).float().to(device)
            '''
            用于选择性地替换经验池中的动作嵌入

            # 伪代码表示
            for i in range(batch_size):
                if d_bing[i] == 1:  # 动作匹配
                    discrete_emb_[i] = discrete_emb[i]  # 保持原始嵌入
                else:  # d_bing[i] == 0，动作不匹配
                    discrete_emb_[i] = discrete_emb_table_noise[i]  # 使用重标记的嵌入，表示嵌入质量差，这个是重新根据实际离散动作计算得到含有噪音的离散动作嵌入

            todo 这就有一个问题，实际的动作嵌入也是根据网络预测的，为啥会出现偏差？
            难道是根据训练不断的得到一个更加准确的动作嵌入
            '''
            discrete_emb_ = d_bing * discrete_emb + (1.0 - d_bing) * discrete_emb_table_noise
            # print("discrete_emb_final",discrete_emb_)

            # 预测新旧观察的差值
            predict_delta_state = action_rep.select_delta_state(state, parameter_emb, discrete_emb_table)

            # print("predict_delta_state",predict_delta_state)
            # print("state_next_state",state_next_state.cpu().numpy())
            delta_state = (np.square(predict_delta_state - state_next_state.cpu().numpy())).mean(axis=1).reshape(-1, 1) # 计算预测的新旧状态误差和实际的新旧状态误差之间的差值
            # delta_state=predict_delta_state-state_next_state.cpu().numpy()
            # delta_state=np.mean(delta_state, axis=1).reshape(-1, 1)
            s_bing = (abs(delta_state) < recon_s_rate) * 1 # 计算预测误差是否在允许的范围内，又是一个bool矩阵
            parameter_relable_rate = sum(s_bing.reshape(1, -1)[0]) / batch_size # 计算在允许范围内的比例，也就是计算预测比较准确的比例
            s_bing = torch.FloatTensor(s_bing).float().to(device) # 转换为tensor
            # print("s_bing",s_bing)

            # 将真实的样本数据通过vae编码重建得到动作和观察的嵌入表示，以及连续动作和观察的潜在空间均值和方差
            recon_c, recon_s, mean, std = action_rep.vae(state, discrete_emb_table, parameter_action)
            parameter_emb_ = mean + std * torch.randn_like(std) # 重参数化采样潜在空间的值
            for i in range(len(parameter_emb_[0])):
                # 这个循环是在逐维度处理参数嵌入，将每个维度的值从VAE的分布范围映射到TD3的动作范围
                # 这里对应于在采样时做的转换
                '''
                这是范围标准化的逆变换：
                    输入: VAE分布范围内的参数嵌入值
                    输出: 标准化范围[-1, 1]内的参数嵌入值
                '''
                parameter_emb_[:, i:i + 1] = self.true_parameter_emb(parameter_emb_[:, i:i + 1], c_rate, i)
            # print("parameter_emb",parameter_emb)
            # print("parameter_emb_",parameter_emb_)

            # 根据s_bing选择性地替换经验池中的连续动作嵌入，保证传入到训练中的连续动作嵌入是比较准确的
            # 同理由于模型的不断迭代，潜在空间采样的连续动作嵌入会越来越准确，所以需要确保经验池中的连续动作嵌入也是比较准确的，对于不准确的就用重新编码得到的
            parameter_emb_ = s_bing * parameter_emb + (1 - s_bing) * parameter_emb_
            # print("parameter_emb_final",parameter_emb_)

            # 确保动作嵌入在合理范围内
            discrete_emb_ = discrete_emb_.clamp(-self.max_action, self.max_action)
            parameter_emb_ = parameter_emb_.clamp(-self.max_action, self.max_action)

            # 通过以上流程，对于经验池中记录的动作嵌入向量，如果质量较差，就通过实际的动作重新编码得到更准确的嵌入表示，从而提升训练效果
            discrete_emb = discrete_emb_
            parameter_emb = parameter_emb_

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # 这里是创造噪音，因为是潜在空间表示，所以可以更加平滑、自由的采样
            noise_discrete = (
                    torch.randn_like(discrete_emb) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            noise_parameter = (
                    torch.randn_like(parameter_emb) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            # 预测下一个状态的动作，并给下一个动作增加噪音
            next_discrete_action, next_parameter_action = self.actor_target(next_state)
            next_discrete_action = (next_discrete_action + noise_discrete).clamp(-self.max_action, self.max_action)
            next_parameter_action = (next_parameter_action + noise_parameter).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            # 根据下一个状态和下一个动作计算目标Q值，并按照TD3的方式取两个Q值中的较小值计算得到bellman方程的目标Q值
            target_Q1, target_Q2 = self.critic_target(next_state, next_discrete_action, next_parameter_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates 根据当前的状态和动作计算当前的Q值
        current_Q1, current_Q2 = self.critic(state, discrete_emb, parameter_emb)

        # Compute critic loss 计算critic的损失函数
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        # 开始训练动作策略方法
        if self.total_it % self.policy_freq == 0:
            inverting_gradients = True
            # inverting_gradients = False
            # Compute actor loss
            if inverting_gradients:
                # 这里计算存在重复，看md文档
                with torch.no_grad():
                    next_discrete_action, next_parameter_action = self.actor(state)
                    action_params = torch.cat((next_discrete_action, next_parameter_action), dim=1)
                action_params.requires_grad = True
                # 这个损失就是经典的求最大Q值的损失
                actor_loss = self.critic.Q1(state, action_params[:, :self.discrete_action_dim],
                                            action_params[:, self.discrete_action_dim:]).mean()
            else:
                # 传统的计算方式
                next_discrete_action, next_parameter_action = self.actor(state)
                actor_loss = -self.critic.Q1(state, next_discrete_action, next_parameter_action).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            if inverting_gradients:
                from copy import deepcopy
                delta_a = deepcopy(action_params.grad.data)
                # 2 - apply inverting gradients and combine with gradients from actor
                actions, action_params = self.actor(Variable(state))
                # # 对连续动作参数应用梯度反转
                action_params = torch.cat((actions, action_params), dim=1)
                # 根据预测的动作值和边界值调整梯度
                delta_a[:, self.discrete_action_dim:] = self._invert_gradients(
                    delta_a[:, self.discrete_action_dim:].cpu(),
                    action_params[:, self.discrete_action_dim:].cpu(),
                    grad_type="action_parameters", inplace=True)
                # 对离散动作嵌入应用梯度反转
                delta_a[:, :self.discrete_action_dim] = self._invert_gradients(
                    delta_a[:, :self.discrete_action_dim].cpu(),
                    action_params[:, :self.discrete_action_dim].cpu(),
                    grad_type="actions", inplace=True)
                # 这段代码的作用是将调整后的梯度应用到Actor网络参数上，详细看md文档
                out = -torch.mul(delta_a, action_params)
                self.actor.zero_grad()
                out.backward(torch.ones(out.shape).to(device))
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)

            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
            self.actor_optimizer.step()

            # Update the frozen target models
            # 同步更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 这里返回的是经验池中动作嵌入被重标记的比例，比例越高说明嵌入的质量越好
        return discrete_relable_rate, parameter_relable_rate


    # def save(self, filename, directory):
    #     torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))
    #     # torch.save(self.vae.embeddings, '%s/%s_embeddings.pth' % (directory, filename))
    #
    # def load(self, filename, directory):
    #     self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=self.device))
    #     # self.vae.embeddings = torch.load('%s/%s_embeddings.pth' % (directory, filename), map_location=self.device)


    def save(self, filename,directory):
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))

    def load(self, filename,directory):
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.actor_target = copy.deepcopy(self.actor)

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        '''
        grad: 要调整的梯度张量
        vals: 与梯度对应的动作值张量
        grad_type: 指定梯度类型，是针对离散动作嵌入("actions")还是连续动作参数("action_parameters")
        inplace: 是否在原地修改梯度

        是一个梯度调整函数，用于处理有界动作空间中的梯度优化问题。它根据当前动作值在边界中的位置，智能地调整梯度强度，避免边界处的梯度消失问题
        '''

        if grad_type == "actions":
            max_p = self.action_max.cpu() # 上边界 [1, 1, 1, ...]
            min_p = self.action_min.cpu()  # 下边界 [-1, -1, -1, ...]
            rnge = self.action_range.cpu() # 范围 [2, 2, 2, ...]
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max.cpu()
            min_p = self.action_parameter_min.cpu()
            rnge = self.action_parameter_range.cpu()
        else:
            raise ValueError("Unhandled grad_type: '" + str(grad_type) + "'")

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            for n in range(grad.shape[0]): # 遍历batch中的每个样本
                # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
                '''
                原理：
                正梯度调整公式：
                adjusted_grad_positive = original_grad * (max_value - current_value) / range
                负梯度调整公式：
                adjusted_grad_negative = original_grad * (current_value - min_value) / range

                边界感知的梯度缩放
                接近边界时：梯度被大幅缩小，避免越界
                远离边界时：梯度保持较大，确保有效学习
                中间位置时：梯度适度调整，平衡探索与约束
                '''
                index = grad[n] > 0
                # # 对正梯度进行缩放
                grad[n][index] *= (index.float() * (max_p - vals[n]) / rnge)[index]
                # # 对负梯度进行缩放
                grad[n][~index] *= ((~index).float() * (vals[n] - min_p) / rnge)[~index]

        return grad

    def count_boundary(self, c_rate):
        median = (c_rate[0] - c_rate[1]) / 2
        offset = c_rate[0] - 1 * median
        return median, offset

    def true_parameter_emb(self, parameter_action, c_rate, i):
        # parameter_action_ = parameter_action.clone()
        median, offset = self.count_boundary(c_rate[i])
        # parameter_action_[i] = parameter_action_[i] * median + offset
        parameter_action = (parameter_action - offset) / median
        return parameter_action
