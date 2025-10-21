# TODO: s discrete continue ->s"
import numpy as np
import torch
from torch import float32
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from embedding.Utils.utils import NeuralNet, pairwise_distances, pairwise_hyp_distances, squash, atanh
from embedding.Utils import Basis
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as functional


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, action_embedding_dim, parameter_action_dim, latent_dim, max_action,
                 hidden_size=256):
        '''
        state_dim: 环境观察的维度
        action_dim: 离散动作的维度
        action_embedding_dim：离散动作对应的连续动作值的维度
        parameter_action_dim：离散动作对应的连续动作的维度（当前的代码中每个离散动作对应的连续动作的维度都是相同的）
        latent_dim: todo 动作潜在空间的维度
        max_action: 动作的最大值，看来必须要对称
        hidden_size: 隐藏层的尺寸
        '''
        super(VAE, self).__init__()

        # embedding table 这个是啥？todo
        # 初始化了一个shape（action_dim, action_embedding_dim) * 2 - 1 )的随机矩阵
        # 感觉类似词潜入矩阵，将离散动作转换为一个潜入向量
        init_tensor = torch.rand(action_dim,
                                 action_embedding_dim) * 2 - 1  # Don't initialize near the extremes.
        self.embeddings = torch.nn.Parameter(init_tensor.type(float32), requires_grad=True)
        print("self.embeddings", self.embeddings)
        self.e0_0 = nn.Linear(state_dim + action_embedding_dim, hidden_size)
        self.e0_1 = nn.Linear(parameter_action_dim, hidden_size)

        self.e1 = nn.Linear(hidden_size, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d0_0 = nn.Linear(state_dim + action_embedding_dim, hidden_size)
        self.d0_1 = nn.Linear(latent_dim, hidden_size)
        self.d1 = nn.Linear(hidden_size, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)

        self.parameter_action_output = nn.Linear(hidden_size, parameter_action_dim)

        self.d3 = nn.Linear(hidden_size, hidden_size)

        self.delta_state_output = nn.Linear(hidden_size, state_dim) # 还原观察

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action, action_parameter):
        '''
        state:环境观察
        action: 离散动作的嵌入向量 
        action_parameter: 离散动作对应的连续动作的值

        结合输入的参数计算得到返回重建后的离散动作对应连续动作、观察（这里的观察是新旧状态之间的差值）、连续动作对应的潜在空间的均值和方差
        '''
        
        z_0 = F.relu(self.e0_0(torch.cat([state, action], 1))) # 离散动作和环境的组合提取特征
        z_1 = F.relu(self.e0_1(action_parameter)) # 离散动作的连续动作提取特征
        z = z_0 * z_1 # 两者相乘结合

        # 进一步提取特征
        z = F.relu(self.e1(z))
        z = F.relu(self.e2(z))

        mean = self.mean(z) # 得到多模态（观察+离散动作+连续动作）下动作潜在空间的均值
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15) # 得到多模态（观察+离散动作+连续动作）下动作潜在空间的方差

        std = torch.exp(log_std) # 得到真实的方差
        z = mean + std * torch.randn_like(std) # 基于均值的采样
        u, s = self.decode(state, z, action) # 解码得到重构的动作和状态（这里的观察是新旧状态之间的差值）

        # 确认这里的返回值是啥？
        # 返回重建后的离散动作对应连续动作、观察（这里的观察是新旧状态之间的差值）、连续动作对应的潜在空间的均值和方差
        return u, s, mean, std

    def decode(self, state, z=None, action=None, clip=None, raw=False):
        '''
        state: 环境观察
        z: 经过vae特征采集后预测的动作潜在空间值
        action：离散动作的嵌入向量 
        raw： True仅返回预测的所有离散动作的连续值，False返回重建后的观察和预测的所有离散动作的连续值

        return: 返回重建后的观察（这里的观察是新旧状态之间的差值）和重建的离散动作的对应的连续动作值值
        '''
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            # 生成一个随机的z预测的动作值
            z = torch.randn((state.shape[0], self.latent_dim)).to(device)
            if clip is not None:
                z = z.clamp(-clip, clip)
        # 通过环境观察和离散动作的嵌入向量提取特征
        v_0 = F.relu(self.d0_0(torch.cat([state, action], 1)))
        v_1 = F.relu(self.d0_1(z)) # decode动作潜在空间值
        v = v_0 * v_1 # 组合两者特征
        v = F.relu(self.d1(v)) # 进一步decode
        v = F.relu(self.d2(v))

        # 预测连续动作
        parameter_action = self.parameter_action_output(v)

        v = F.relu(self.d3(v))
        s = self.delta_state_output(v) # 重建观察

        if raw: return parameter_action, s
        return self.max_action * torch.tanh(parameter_action), torch.tanh(s)


class Action_representation(NeuralNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 parameter_action_dim,
                 reduced_action_dim=2,
                 reduce_parameter_action_dim=2,
                 embed_lr=1e-4,
                 ):
        '''
        state_dim: 环境观察的维度
        action_dim: 离散动作的维度
        parameter_action_dim：离散动作对应的连续动作的维度（而不是全部连续动作的维度）
        reduced_action_dim：离散动作对应的潜在空间的维度
        reduce_parameter_action_dim：动作潜在空间的维度
        embed_lr：学习率
        '''
        super(Action_representation, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.parameter_action_dim = parameter_action_dim
        self.reduced_action_dim = reduced_action_dim
        self.reduce_parameter_action_dim = reduce_parameter_action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Action embeddings to project the predicted action into original dimensions
        # latent_dim=action_dim*2+parameter_action_dim*2
        self.latent_dim = self.reduce_parameter_action_dim
        self.embed_lr = embed_lr
        self.vae = VAE(state_dim=self.state_dim, action_dim=self.action_dim,
                       action_embedding_dim=self.reduced_action_dim, parameter_action_dim=self.parameter_action_dim,
                       latent_dim=self.latent_dim, max_action=1.0,
                       hidden_size=256).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-4)

    def discrete_embedding(self, ):
        emb = self.vae.embeddings

        return emb

    def unsupervised_loss(self, s1, a1, a2, s2, sup_batch_size, embed_lr):
        '''
        s1: 环境观察
        a1：离散动作
        a2：离散动作对应的连续动作的值
        s2：新旧状态之间的差值
        sup_batch_size：采样的样本数
        embed_lr：学习率
        '''

        # 将离散动作转换嵌入向量
        a1 = self.get_embedding(a1).to(self.device)

        s1 = s1.to(self.device)
        s2 = s2.to(self.device)
        a2 = a2.to(self.device)
        
        # 完成vae的训练，并返回：vae重建损失、观察变化损失、连续动作重建损失、KL约束散度损失 以上损失都只是标量值，估计只是为了记录
        vae_loss, recon_loss_d, recon_loss_c, KL_loss = self.train_step(s1, a1, a2, s2, sup_batch_size, embed_lr)
        return vae_loss, recon_loss_d, recon_loss_c, KL_loss

    def loss(self, state, action_d, action_c, next_state, sup_batch_size):
        '''
        state:环境观察
        action_d: 离散动作的潜入向量 
        action_c 离散动作对应的连续动作的值
        next_state, 新旧状态之间的差值
        sup_batch_size：采样的样本数

        return: vae重建损失、观察变化损失、连续动作重建损失、KL约束散度损失
        '''
        # recon_c：重建后的离散动作对应连续动作的值
        # recon_s: 重构新旧状态之间的差值观察
        recon_c, recon_s, mean, std = self.vae(state, action_d, action_c)

        #  size_average=True：计算所有元素的MSE后，再除以元素总数，返回的是平均损失值
        recon_loss_s = F.mse_loss(recon_s, next_state, size_average=True) # 计算重建观察差值之间的损失
        recon_loss_c = F.mse_loss(recon_c, action_c, size_average=True) # 计算重建离散动作对应的连续动作值之间的损失

        # 这个最大的作用感觉是即保证了网络接近重建的能力又保证了一定的随机性
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()

        # vae_loss = 0.25 * recon_loss_s + recon_loss_c + 0.5 * KL_loss
        # vae_loss = 0.25 * recon_loss_s + 2.0 * recon_loss_c + 0.5 * KL_loss  #best
        vae_loss = recon_loss_s + 2.0 * recon_loss_c + 0.5 * KL_loss # 计算总损失，其中的比例通过注释可知是测试出来的
        # print("vae loss",vae_loss)
        # return vae_loss, 0.25 * recon_loss_s, recon_loss_c, 0.5 * KL_loss
        # return vae_loss, 0.25 * recon_loss_s, 2.0 * recon_loss_c, 0.5 * KL_loss #best
        return vae_loss, recon_loss_s, 2.0 * recon_loss_c, 0.5 * KL_loss

    def train_step(self, s1, a1, a2, s2, sup_batch_size, embed_lr=1e-4):
        '''
        s1:环境观察
        a1: 离散动作的潜入向量 
        a2： 离散动作对应的连续动作的值
        s2, 新旧状态之间的差值
        sup_batch_size：采样的样本数
        embed_lr：学习率

        return vae重建损失、观察变化损失、连续动作重建损失、KL约束散度损失 以上损失都只是标量值，估计只是为了记录
        '''
        state = s1
        action_d = a1
        action_c = a2
        next_state = s2
        vae_loss, recon_loss_s, recon_loss_c, KL_loss = self.loss(state, action_d, action_c, next_state,
                                                                  sup_batch_size)

        # 更新VAE 
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=embed_lr) # todo 这里存在问题，如果仅仅只是为了手动调整学习率，可以有更好的做法，参考md
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        return vae_loss.cpu().data.numpy(), recon_loss_s.cpu().data.numpy(), recon_loss_c.cpu().data.numpy(), KL_loss.cpu().data.numpy()

    def select_parameter_action(self, state, z, action):
        '''
        state: 环境观察
        z: 预测连续动作并转换为真实动作范围的连续动作嵌入向量
        action：预测离散动作的嵌入向量

        return 返回预测的连续动作的连续动作值
        '''
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            z = torch.FloatTensor(z.reshape(1, -1)).to(self.device)
            action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
            action_c, state = self.vae.decode(state, z, action) # 返回重建后的观察（这里的观察是新旧状态之间的差值）和预测的所有离散动作的连续值
        return action_c.cpu().data.numpy().flatten()

    # def select_delta_state(self, state, z, action):
    #     with torch.no_grad():
    #         state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #         z = torch.FloatTensor(z.reshape(1, -1)).to(self.device)
    #         action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
    #         action_c, state = self.vae.decode(state, z, action)
    #     return state.cpu().data.numpy().flatten()
    def select_delta_state(self, state, z, action):
        '''
        state: 环境观察
        z：连续动作的嵌入表示
        action： 离散动作的嵌入表示

        return: 重建后的新旧观察差值
        '''
        with torch.no_grad():
            action_c, state = self.vae.decode(state, z, action)
        return state.cpu().data.numpy()

    def get_embedding(self, action):
        '''
        action：执行的离散动作

        return: 将离散动作转换为一个嵌入向量，类似LSTM的嵌入模型
        '''
        # Get the corresponding target embedding
        action_emb = self.vae.embeddings[action] # 看起来是从VAE的embeddings随机矩阵中选择一个和离散动作对应的随机连续动作值
        action_emb = torch.tanh(action_emb) # 归一化到-1～1
        return action_emb

    def get_match_scores(self, action):
        '''
        action: 输入的是离散动作的嵌入向量

        return: 输入动作嵌入与所有可能动作嵌入的相似度
        '''
        # compute similarity probability based on L2 norm
        embeddings = self.vae.embeddings
        embeddings = torch.tanh(embeddings)
        action = action.to(self.device)
        # compute similarity probability based on L2 norm 输入动作嵌入与所有可能动作嵌入的相似度
        similarity = - pairwise_distances(action, embeddings)  # Negate euclidean to convert diff into similarity score
        return similarity

        # 获得最优动作，输出于embedding最相近的action 作为最优动作.

    def select_discrete_action(self, action):
        '''
        将策略网络输出的动作嵌入转换为具体的离散动作索引
        action: 输入的是离散动作的嵌入向量

        return: 离散动作的索引  也就是选择的离散动作
        '''
    
        similarity = self.get_match_scores(action)
        val, pos = torch.max(similarity, dim=1) # 选择相似度最大的动作位置索引
        # print("pos",pos,len(pos))
        if len(pos) == 1:
            return pos.cpu().item()  # data.numpy()[0]
        else:
            # print("pos.cpu().item()", pos.cpu().numpy())
            return pos.cpu().numpy()

    def save(self, filename, directory):
        torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))
        # torch.save(self.vae.embeddings, '%s/%s_embeddings.pth' % (directory, filename))

    def load(self, filename, directory):
        self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=self.device))
        # self.vae.embeddings = torch.load('%s/%s_embeddings.pth' % (directory, filename), map_location=self.device)

    def get_c_rate(self, s1, a1, a2, s2, batch_size=100, range_rate=5):
        '''
        s1: 环境观察
        a1: 离散动作
        a2: 离散动作对应的连续动作的值
        s2: 新旧动作之间的差值
        batch_size: 采样的尺寸
        range_rate： todo

        return 离散动作潜在空间的边界范围、重建观察差值损失
        '''
        a1 = self.get_embedding(a1).to(self.device) # 将离散动作转换为嵌入向量
        s1 = s1.to(self.device)
        s2 = s2.to(self.device)
        a2 = a2.to(self.device)
        recon_c, recon_s, mean, std = self.vae(s1, a1, a2) # 利用vae计算预测的返回重建后的离散动作对应连续动作、观察（这里的观察是新旧状态之间的差值）、连续动作对应的潜在空间的均值和方差
        # print("recon_s",recon_s.shape)
        # std * torch.randn_like(std)：实现VAE的重参数化技巧(Reparameterization Trick)
        '''
        mean = self.mean(z) # 得到动作的均值
        log_std = self.log_std(z).clamp(-4, 15) # 得到动作的方差log值
        std = torch.exp(log_std) # 得到真实的方差
        z = mean + std * torch.randn_like(std) # 重参数化采样

        保证梯度可以传播又可以保证随机性，如果直接torch.normal(mean, std)计算采样得到的动作是无法传播的
        '''
        z = mean + std * torch.randn_like(std) # 采样连续动作
        z = z.cpu().data.numpy()
        # 返回连续动作的潜在空间的边界值范围
        c_rate = self.z_range(z, batch_size, range_rate)
        # print("s2",s2.shape)

        # 计算重建观察差值损失
        recon_s_loss = F.mse_loss(recon_s, s2, size_average=True)

        # recon_s = abs(np.mean(recon_s.cpu().data.numpy()))
        return c_rate, recon_s_loss.detach().cpu().numpy()

    def z_range(self, z, batch_size=100, range_rate=5):
        '''
        函数的作用是计算潜在空间采样值的动态边界范围
        z: 采样得到的连续动作
        batch_size: 训练batch
        range_rate：todo
        '''

        # todo
        self.z1, self.z2, self.z3, self.z4, self.z5, self.z6, self.z7, self.z8, self.z9,\
        self.z10,self.z11,self.z12,self.z13,self.z14,self.z15,self.z16 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        border = int(range_rate * (batch_size / 100)) # # 计算边界索引 todo 这个值可能是需要根据实际进行调整

        # print("border",border)
        if len(z[0]) == 16:
            # # 对每个潜在维度分别处理
            for i in range(len(z)):
                self.z1.append(z[i][0]) # # 收集第1维的所有值
                self.z2.append(z[i][1]) # 收集第2维的所有值
                self.z3.append(z[i][2]) # 收集第3维的所有值
                self.z4.append(z[i][3]) # 收集第4维的所有值
                self.z5.append(z[i][4]) # 收集第5维的所有值
                self.z6.append(z[i][5]) # 收集第6维的所有值
                self.z7.append(z[i][6]) # 收集第7维的所有值
                self.z8.append(z[i][7]) # 收集第8维的所有值
                self.z9.append(z[i][8]) # 收集第9维的所有值
                self.z10.append(z[i][9]) # 收集第10维的所有值
                self.z11.append(z[i][10]) # 收集第11维的所有值
                self.z12.append(z[i][11]) # 收集第12维的所有值
                self.z13.append(z[i][12]) # 收集第13维的所有值
                self.z14.append(z[i][13]) # 收集第14维的所有值
                self.z15.append(z[i][14]) # 收集第15维的所有值
                self.z16.append(z[i][15]) # 收集第16维的所有值

        if len(z[0]) == 16:
            # 对收集后的维度所有值进行排序
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort(), self.z7.sort(), self.z8.sort(), \
            self.z9.sort(), self.z10.sort(), self.z11.sort(), self.z12.sort(),self.z13.sort(), self.z14.sort(), self.z15.sort(), self.z16.sort()
            c_rate_1_up = self.z1[-border - 1] # # 上边界：排序后的第95%分位
            c_rate_1_down = self.z1[border] #  # 下边界：排序后的第5%分位
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]
            c_rate_7_up = self.z7[-border - 1]
            c_rate_7_down = self.z7[border]
            c_rate_8_up = self.z8[-border - 1]
            c_rate_8_down = self.z8[border]
            c_rate_9_up = self.z9[-border - 1]
            c_rate_9_down = self.z9[border]
            c_rate_10_up = self.z10[-border - 1]
            c_rate_10_down = self.z10[border]
            c_rate_11_up = self.z11[-border - 1]
            c_rate_11_down = self.z11[border]
            c_rate_12_up = self.z12[-border - 1]
            c_rate_12_down = self.z12[border]
            c_rate_13_up = self.z13[-border - 1]
            c_rate_13_down = self.z13[border]
            c_rate_14_up = self.z14[-border - 1]
            c_rate_14_down = self.z14[border]
            c_rate_15_up = self.z15[-border - 1]
            c_rate_15_down = self.z15[border]
            c_rate_16_up = self.z16[-border - 1]
            c_rate_16_down = self.z16[border]

            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, \
            c_rate_9, c_rate_10, c_rate_11, c_rate_12, c_rate_13, c_rate_14, c_rate_15, c_rate_16 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)
            c_rate_7.append(c_rate_7_up), c_rate_7.append(c_rate_7_down)
            c_rate_8.append(c_rate_8_up), c_rate_8.append(c_rate_8_down)
            c_rate_9.append(c_rate_9_up), c_rate_9.append(c_rate_9_down)
            c_rate_10.append(c_rate_10_up), c_rate_10.append(c_rate_10_down)
            c_rate_11.append(c_rate_11_up), c_rate_11.append(c_rate_11_down)
            c_rate_12.append(c_rate_12_up), c_rate_12.append(c_rate_12_down)
            c_rate_13.append(c_rate_13_up), c_rate_13.append(c_rate_13_down)
            c_rate_14.append(c_rate_14_up), c_rate_14.append(c_rate_14_down)
            c_rate_15.append(c_rate_15_up), c_rate_15.append(c_rate_15_down)
            c_rate_16.append(c_rate_16_up), c_rate_16.append(c_rate_16_down)

            # 返回每个维度的范围（上边界和下边界）
            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8,\
                   c_rate_9, c_rate_10, c_rate_11, c_rate_12,c_rate_13, c_rate_14, c_rate_15, c_rate_16

        if len(z[0]) == 12:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])
                self.z7.append(z[i][6])
                self.z8.append(z[i][7])
                self.z9.append(z[i][8])
                self.z10.append(z[i][9])
                self.z11.append(z[i][10])
                self.z12.append(z[i][11])

        if len(z[0]) == 12:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort(), self.z7.sort(), self.z8.sort(), \
            self.z9.sort(), self.z10.sort(), self.z11.sort(), self.z12.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]
            c_rate_7_up = self.z7[-border - 1]
            c_rate_7_down = self.z7[border]
            c_rate_8_up = self.z8[-border - 1]
            c_rate_8_down = self.z8[border]
            c_rate_9_up = self.z9[-border - 1]
            c_rate_9_down = self.z9[border]
            c_rate_10_up = self.z10[-border - 1]
            c_rate_10_down = self.z10[border]
            c_rate_11_up = self.z11[-border - 1]
            c_rate_11_down = self.z11[border]
            c_rate_12_up = self.z12[-border - 1]
            c_rate_12_down = self.z12[border]
            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, c_rate_9, c_rate_10, c_rate_11, c_rate_12 = [], [], [], [], [], [], [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)
            c_rate_7.append(c_rate_7_up), c_rate_7.append(c_rate_7_down)
            c_rate_8.append(c_rate_8_up), c_rate_8.append(c_rate_8_down)
            c_rate_9.append(c_rate_9_up), c_rate_9.append(c_rate_9_down)
            c_rate_10.append(c_rate_10_up), c_rate_10.append(c_rate_10_down)
            c_rate_11.append(c_rate_11_up), c_rate_11.append(c_rate_11_down)
            c_rate_12.append(c_rate_12_up), c_rate_12.append(c_rate_12_down)
            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, c_rate_9, c_rate_10, c_rate_11, c_rate_12

        if len(z[0]) == 10:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])
                self.z7.append(z[i][6])
                self.z8.append(z[i][7])
                self.z9.append(z[i][8])
                self.z10.append(z[i][9])

        if len(z[0]) == 10:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort(), self.z7.sort(), self.z8.sort(), self.z9.sort(), self.z10.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]
            c_rate_7_up = self.z7[-border - 1]
            c_rate_7_down = self.z7[border]
            c_rate_8_up = self.z8[-border - 1]
            c_rate_8_down = self.z8[border]
            c_rate_9_up = self.z9[-border - 1]
            c_rate_9_down = self.z9[border]
            c_rate_10_up = self.z10[-border - 1]
            c_rate_10_down = self.z10[border]
            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, c_rate_9, c_rate_10 = [], [], [], [], [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)
            c_rate_7.append(c_rate_7_up), c_rate_7.append(c_rate_7_down)
            c_rate_8.append(c_rate_8_up), c_rate_8.append(c_rate_8_down)
            c_rate_9.append(c_rate_9_up), c_rate_9.append(c_rate_9_down)
            c_rate_10.append(c_rate_10_up), c_rate_10.append(c_rate_10_down)
            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, c_rate_9, c_rate_10

        if len(z[0]) == 8:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])
                self.z7.append(z[i][6])
                self.z8.append(z[i][7])

        if len(z[0]) == 8:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort(), self.z7.sort(), self.z8.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]
            c_rate_7_up = self.z7[-border - 1]
            c_rate_7_down = self.z7[border]
            c_rate_8_up = self.z8[-border - 1]
            c_rate_8_down = self.z8[border]
            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8 = [], [], [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)
            c_rate_7.append(c_rate_7_up), c_rate_7.append(c_rate_7_down)
            c_rate_8.append(c_rate_8_up), c_rate_8.append(c_rate_8_down)
            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8

        if len(z[0]) == 6:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])

        if len(z[0]) == 6:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]

            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6 = [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)

            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6

        if len(z[0]) == 4:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])

        if len(z[0]) == 4:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort()
            # print("lenz1",len(self.z1),self.z1)
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]

            c_rate_1, c_rate_2, c_rate_3, c_rate_4 = [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)

            return c_rate_1, c_rate_2, c_rate_3, c_rate_4

        if len(z[0]) == 3:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])

        if len(z[0]) == 3:
            self.z1.sort(), self.z2.sort(), self.z3.sort()
            # print("lenz1",len(self.z1),self.z1)
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]

            c_rate_1, c_rate_2, c_rate_3 = [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)

            return c_rate_1, c_rate_2, c_rate_3
