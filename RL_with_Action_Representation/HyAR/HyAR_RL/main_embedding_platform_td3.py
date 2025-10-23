import numpy as np
import gym
import gym_platform
import argparse
import os
from  HyAR_RL import utils
from agents import P_TD3_relable
from agents import P_DDPG_relable
import copy
from common import ClickPythonLiteralOption
from common.platform_domain import PlatformFlattenedActionWrapper
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
import matplotlib.pyplot as plt
from agents.pdqn import PDQNAgent
from agents.utils import soft_update_target_network, hard_update_target_network
from embedding import ActionRepresentation_vae
import torch
from common.goal_domain import GoalFlattenedActionWrapper, GoalObservationWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_action(act, act_param):
    '''
    act: 离散动作
    act_param: 离散动作对应的连续动作参数
    结合这里，可以知道本游戏的空间的连续动作参数是3维的（每个离散动作对应1维）
    但代码貌似可以推广到不同维度的情况 todo 如何推广
    同时代码中也可以用于每个连续动作的范围是不一样的情况

    返回组合后的动作：action = (discrete_action_index, [param_0, param_1, param_2])
    '''
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def evaluate(env, policy, action_rep, c_rate, episodes=100):
    returns = []
    epioside_steps = []

    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            discrete_emb, parameter_emb = policy.select_action(state)
            true_parameter_emb = true_parameter_action(parameter_emb, c_rate)
            # select discrete action
            discrete_action_embedding = copy.deepcopy(discrete_emb)
            discrete_action_embedding = torch.from_numpy(discrete_action_embedding).float().reshape(1, -1)
            discrete_action = action_rep.select_discrete_action(discrete_action_embedding)
            discrete_emb_1 = action_rep.get_embedding(discrete_action).cpu().view(-1).data.numpy()
            all_parameter_action = action_rep.select_parameter_action(state, true_parameter_emb,
                                                                      discrete_emb_1)
            parameter_action = all_parameter_action
            action = pad_action(discrete_action, parameter_action)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        epioside_steps.append(t)
        returns.append(total_reward)
    print("---------------------------------------")
    print(
        f"Evaluation over {episodes} episodes: {np.array(returns[-100:]).mean():.3f} epioside_steps: {np.array(epioside_steps[-100:]).mean():.3f}")
    print("---------------------------------------")
    return np.array(returns[-100:]).mean(), np.array(epioside_steps[-100:]).mean()


def run(args):
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    if args.env == "Platform-v0":
        # 这里对Platform-v0环境进行了一些特殊处理
        env = gym.make(args.env) # 看来是直接注册到gym里的，所以可以直接make
        env = ScaledStateWrapper(env) # 主要构建一个-1~1的观察空间
        '''
        当前代码的事例，每个离散动作都对应着连续动作
        每个连续动作的维度都相同，但是原始范围不同
        这是为每个动作设置的初始参数值:
        这里应该是将初始动作值缩放到-1~1之间

        RUN动作: 3.0 (范围在0-30之间)
        HOP动作: 10.0 (范围在0-720之间)
        LEAP动作: 400.0 (范围在0-430之间)
        '''
        initial_params_ = [3., 10., 400.] # todo 这个值是用来做啥的？
        # 这段代码的作用是将Platform环境的初始动作参数从原始范围缩放到[-1, 1]区间
        # 公式：scaled_value = 2 * (value - min) / (max - min) - 1
        for a in range(env.action_space.spaces[0].n):
            initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low) / (
                    env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.
        # 定义了一个初始权重矩阵，形状为(动作数量, 状态维度)，并初始化为零 这是什么？
        # 不用看了，这个值后面没用，代码混乱
        initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
        env = PlatformFlattenedActionWrapper(env) # 展平
        env = ScaledParameterisedActionWrapper(env) # 将动作空间缩放到-1~1

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # 是在获取状态空间的维度数（状态特征的数量）。
    '''
    env.observation_space → 整个观察空间（Tuple类型）
    .spaces[0] → Tuple中的第一个元素（Box空间，包含实际状态向量）
    .shape[0] → Box空间的第一个维度（状态向量的长度）
    '''
    state_dim = env.observation_space.spaces[0].shape[0]

    discrete_action_dim = env.action_space.spaces[0].n # 离散动作的数量
    # 构建每个离散动作对应的连续动作空间的维度
    action_parameter_sizes = np.array(
        [env.action_space.spaces[i].shape[0] for i in range(1, discrete_action_dim + 1)])
    parameter_action_dim = int(action_parameter_sizes.sum()) # 统计所有连续动作的维度和
    # 这个应该是离散动作的嵌入维度和连续动作的嵌入维度
    discrete_emb_dim = discrete_action_dim * 2 # todo 这里为什么是乘以2？
    parameter_emb_dim = parameter_action_dim * 2 # todo 这里为什么是乘以2？
    max_action = 1.0 # todo 为什么定义最大的动作是1.0，是因为之前ScaledParameterisedActionWrapper？如果连续动作的原始范围是0.0~1.0呢？
    print("state_dim", state_dim)
    print("discrete_action_dim", discrete_action_dim)
    print("parameter_action_dim", parameter_action_dim)

    kwargs = {
        "state_dim": state_dim,
        "discrete_action_dim": discrete_emb_dim,
        "parameter_action_dim": parameter_emb_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "P-TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = P_TD3_relable.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    # 加载预训练模型
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # embedding初始部分
    # 从这里可以看出，这里的每个离散动作对应的连续动作的维度依旧是1，但是每个连续动作的原始范围不同
    action_rep = ActionRepresentation_vae.Action_representation(state_dim=state_dim,
                                                                  action_dim=discrete_action_dim,
                                                                  parameter_action_dim=1,
                                                                  reduced_action_dim=discrete_emb_dim,
                                                                  reduce_parameter_action_dim=parameter_emb_dim
                                                                  )

    replay_buffer = utils.ReplayBuffer(state_dim, discrete_action_dim=1,
                                       parameter_action_dim=1,
                                       all_parameter_action_dim=parameter_action_dim,
                                       discrete_emb_dim=discrete_emb_dim,
                                       parameter_emb_dim=parameter_emb_dim,
                                       max_size=int(1e5))

    replay_buffer_embedding = utils.ReplayBuffer(state_dim, discrete_action_dim=1,
                                                 parameter_action_dim=1,
                                                 all_parameter_action_dim=parameter_action_dim,
                                                 discrete_emb_dim=discrete_emb_dim,
                                                 parameter_emb_dim=parameter_emb_dim,
                                                 # max_size=int(2e7)
                                                 max_size=int(1e6)
                                                 )

    # 这里主要用于样本采集，虽然里面有定义优化器
    # 但是实际上在本代码中并未调用
    agent_pre = PDQNAgent(
        env.observation_space.spaces[0], env.action_space,
        batch_size=128,
        learning_rate_actor=0.001,
        learning_rate_actor_param=0.0001,
        epsilon_steps=1000,
        gamma=0.9,
        tau_actor=0.1,
        tau_actor_param=0.01,
        clip_grad=10.,
        indexed=False,
        weighted=False,
        average=False,
        random_weighted=False,
        initial_memory_threshold=500,
        use_ornstein_noise=False,
        replay_memory_size=10000,
        epsilon_final=0.01,
        inverting_gradients=True,
        zero_index_gradients=False,
        seed=args.seed)

    # ------Use random strategies to collect experience------

    max_steps = 250 # todo 为啥步数会这么大
    total_reward = 0. # 所有游戏回合的总奖励
    returns = [] # 记录每轮游戏的总奖励
    for i in range(5000):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        act, act_param, all_action_parameters = agent_pre.act(state)
        # 将预测到的离散动作act和离散动作对应的连续动作act_param组合起来
        action = pad_action(act, act_param)
        episode_reward = 0. # 一轮游戏内的奖励综合
        agent_pre.start_episode() # 该代码中暂无实际用处
        for j in range(max_steps):
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret # 看来返回内容有优点不一样了，没关系
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            next_act, next_act_param, next_all_action_parameters = agent_pre.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            state_next_state = next_state - state # 保存状态差值
            # 记录采集的样本数据到缓冲区中
            replay_buffer_embedding.add(state, act, act_param, all_action_parameters, discrete_emb=None,
                                        parameter_emb=None,
                                        next_state=next_state,
                                        state_next_state=state_next_state,
                                        reward=reward, done=terminal)
            # # 下面好像都没必要？因为后续会直接被覆盖
            # 因为这里采用的采集代码流程结构，所以这里对next_state计算的act实际上就是提前
            # 对下一轮的进行预测，然后方便后续传入step计算
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state
            episode_reward += reward
            if terminal:
                break
        # agent_pre.end_episode()
        returns.append(episode_reward)
        total_reward += episode_reward
        if i % 100 == 0:
            print('per-train-{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1),
                                                                   np.array(returns[-100:]).mean()))
    save_dir = "result/platform_model/mix/1.0/0526"

    save_dir = os.path.join(save_dir, "{}".format(str(66)))
    print("save_dir", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # ------VAE训练------

    initial_losses = []
    VAE_batch_size = 64
    vae_load_model = False
    vae_save_model = True
    # vae_load_model = True
    # vae_save_model = False
    if vae_load_model:
        print("load model")
        title = "vae" + "{}".format(str(40000))
        action_rep.load(title, save_dir)
        print("load discrete embedding", action_rep.discrete_embedding())
    print("pre VAE training phase started...")
    recon_s_loss = []
    c_rate, recon_s = vae_train(action_rep=action_rep, train_step=5000, replay_buffer=replay_buffer_embedding,
                                batch_size=VAE_batch_size,
                                save_dir=save_dir, vae_save_model=vae_save_model, embed_lr=1e-4)

    print("c_rate,recon_s", c_rate, recon_s)
    print("discrete embedding", action_rep.discrete_embedding())

    # -------TD3训练------
    print("TD3 train")
    state, done = env.reset(), False
    total_reward = 0. # 所有训练期间的总奖励
    returns = [] # 每轮游戏的奖励
    Reward = [] # 总平均奖励
    Reward_100 = [] # 近100轮游戏的奖励
    Test_Reward_100 = [] # 近100轮测试的奖励
    Test_epioside_step_100 = []
    max_steps = 250 # 每次游戏的最大步数
    cur_step = 0
    internal = 10
    total_timesteps = 0
    t = 0
    discrete_relable_rate, parameter_relable_rate = 0, 0
    # for t in range(int(args.max_episodes)):
    while total_timesteps < args.max_timesteps: # 游戏训练的总轮数
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        # 这里预测的动作都是嵌入潜在空间的动作
        # 这里是TD3网络预测的动作潜在空间的动作向量
        discrete_emb, parameter_emb = policy.select_action(state)
        # 探索
        if t < args.epsilon_steps:
            epsilon = args.expl_noise_initial - (args.expl_noise_initial - args.expl_noise) * (
                    t / args.epsilon_steps)
        else:
            epsilon = args.expl_noise

        # re-lable rate todo 这里代码无用
        if t < args.relable_steps:
            relable_rate = args.relable_initial - (args.relable_initial - args.relable_final) * (
                    t / args.relable_steps)
        else:
            relable_rate = args.relable_final

        # 选中的代码确实是在给动作嵌入增加噪音，这是HyAR算法中的探索机制
        discrete_emb = (
                discrete_emb + np.random.normal(0, max_action * epsilon, size=discrete_emb_dim)
        ).clip(-max_action, max_action)
        parameter_emb = (
                parameter_emb + np.random.normal(0, max_action * epsilon, size=parameter_emb_dim)
        ).clip(-max_action, max_action)
        # parameter_emb = parameter_emb * c_rate
        #  这里将预测的连续动作的嵌入范围转换为VAE动作的嵌入范围
        # 从模型构建那边可以指导，VAE的潜入空间是没有范围限制的
        # 所以这里也可以说成是将TD3的动作转换未VAE的动作
        true_parameter_emb = true_parameter_action(parameter_emb, c_rate)

        # select discrete action
        discrete_action_embedding = copy.deepcopy(discrete_emb)
        discrete_action_embedding = torch.from_numpy(discrete_action_embedding).float().reshape(1, -1)
        discrete_action = action_rep.select_discrete_action(discrete_action_embedding) # 将动作嵌入转换为具体的离散动作
        discrete_emb_1 = action_rep.get_embedding(discrete_action).cpu().view(-1).data.numpy() # 将预测的离散动作转换为实际的动作嵌入
        # 这个应该只是单个离散动作对应的连续动作吧，对比simple_move_td3，可以得知，这里命名有问题
        all_parameter_action = action_rep.select_parameter_action(state, true_parameter_emb,
                                                                  discrete_emb_1)

        parameter_action = all_parameter_action
        action = pad_action(discrete_action, parameter_action) # 组合成最终动作
        episode_reward = 0.

        if cur_step >= args.start_timesteps: # 只有在达到预设的开始训练步数后，才进行策略网络的训练
            # 返回值无用，可以直接删除
            discrete_relable_rate, parameter_relable_rate = policy.train(replay_buffer, action_rep, c_rate,
                                                                         recon_s, args.batch_size)
        for i in range(max_steps):
            total_timesteps += 1
            cur_step = cur_step + 1
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            # print("terminal",terminal,1-terminal)
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            state_next_state = next_state - state
            replay_buffer.add(state, discrete_action=discrete_action, parameter_action=parameter_action,
                              all_parameter_action=None,
                              discrete_emb=discrete_emb,
                              parameter_emb=parameter_emb,
                              next_state=next_state,
                              state_next_state=state_next_state,
                              reward=reward, done=terminal)
            replay_buffer_embedding.add(state, discrete_action=discrete_action, parameter_action=parameter_action,
                                        all_parameter_action=None,
                                        discrete_emb=None,
                                        parameter_emb=None,
                                        next_state=next_state,
                                        state_next_state=state_next_state,
                                        reward=reward, done=done)
            # 预测下一个状态的动作，用于后续的动作执行
            next_discrete_emb, next_parameter_emb = policy.select_action(next_state)
            # if t % 100 == 0:
            #     print("策略输出", next_discrete_emb, next_parameter_emb)
            next_discrete_emb = (
                    next_discrete_emb + np.random.normal(0, max_action * epsilon, size=discrete_emb_dim)
            ).clip(-max_action, max_action)
            next_parameter_emb = (
                    next_parameter_emb + np.random.normal(0, max_action * epsilon, size=parameter_emb_dim)
            ).clip(-max_action, max_action)
            # next_parameter_emb = next_parameter_emb * c_rate
            true_next_parameter_emb = true_parameter_action(next_parameter_emb, c_rate)
            # select discrete action
            # 利用训练而来的vae 模型，将动作嵌入转换为具体的离散动作
            next_discrete_action_embedding = copy.deepcopy(next_discrete_emb)
            next_discrete_action_embedding = torch.from_numpy(next_discrete_action_embedding).float().reshape(1, -1)
            next_discrete_action = action_rep.select_discrete_action(next_discrete_action_embedding)
            next_discrete_emb_1 = action_rep.get_embedding(next_discrete_action).cpu().view(-1).data.numpy()
            # select parameter action 将连续动作的嵌入转换为真实的连续动作
            next_all_parameter_action = action_rep.select_parameter_action(next_state, true_next_parameter_emb,
                                                                           next_discrete_emb_1)
            # if t % 100 == 0:
            #     print("真实动作", next_discrete_action, next_all_parameter_action)
            # env.render()

            next_parameter_action = next_all_parameter_action
            next_action = pad_action(next_discrete_action, next_parameter_action) # 将真实离散和连续动作拼接
            # discrete_emb=next_discrete_emb： 这里存储的是离散动作嵌入
            # parameter_emb=next_parameter_emb：这里存储的是连续动作嵌入
            # action=next_action：这里存储的是具体的动作
            # discrete_action=next_discrete_action：这里存储的是具体的离散动作
            # parameter_action=next_parameter_action：这里存储的是具体的连续动作
            # 冗余代码
            discrete_emb, parameter_emb, action, discrete_action, parameter_action = next_discrete_emb, next_parameter_emb, next_action, next_discrete_action, next_parameter_action
            state = next_state
            if cur_step >= args.start_timesteps:
                # 在每轮游戏内部也进行策略网络的训练 这里面的训练就可TD3没啥不同了，唯一的区别就是利用VAE重建嵌入，防止嵌入质量差
                discrete_relable_rate, parameter_relable_rate = policy.train(replay_buffer, action_rep, c_rate,
                                                                             recon_s, args.batch_size)
            # if t % 100 == 0:
            #     print("discrete_relable_rate,parameter_relable_rate", discrete_relable_rate, parameter_relable_rate)
            episode_reward += reward

            if total_timesteps % args.eval_freq == 0:
                # 验证、打印调试信息、保存模型
                # todo
                print(
                    '{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(total_timesteps), total_reward / (t + 1),
                                                           np.array(returns[-100:]).mean()))
                while not terminal:
                    state = np.array(state, dtype=np.float32, copy=False)
                    discrete_emb, parameter_emb = policy.select_action(state)
                    true_parameter_emb = true_parameter_action(parameter_emb, c_rate)
                    # select discrete action
                    discrete_action_embedding = copy.deepcopy(discrete_emb)
                    discrete_action_embedding = torch.from_numpy(discrete_action_embedding).float().reshape(1, -1)
                    discrete_action = action_rep.select_discrete_action(discrete_action_embedding)
                    discrete_emb_1 = action_rep.get_embedding(discrete_action).cpu().view(-1).data.numpy()
                    all_parameter_action = action_rep.select_parameter_action(state, true_parameter_emb,
                                                                              discrete_emb_1)
                    parameter_action = all_parameter_action
                    action = pad_action(discrete_action, parameter_action)
                    (state, _), reward, terminal, _ = env.step(action)

                Reward.append(total_reward / (t + 1))
                Reward_100.append(np.array(returns[-100:]).mean())
                Test_Reward, Test_epioside_step = evaluate(env, policy, action_rep, c_rate, episodes=100)
                Test_Reward_100.append(Test_Reward)
                Test_epioside_step_100.append(Test_epioside_step)

            if terminal:
                break
        t = t + 1
        returns.append(episode_reward)
        total_reward += episode_reward


        # vae 训练
        if t % internal == 0 and t >= 1000:
            # 这里还要继续训练vae，防止灾难性遗忘
            # print("表征调整")
            # print("vae train")
            c_rate, recon_s = vae_train(action_rep=action_rep, train_step=1, replay_buffer=replay_buffer_embedding,
                                        batch_size=VAE_batch_size, save_dir=save_dir, vae_save_model=vae_save_model,
                                        embed_lr=1e-4)

            recon_s_loss.append(recon_s)
            # print("discrete embedding", action_rep.discrete_embedding())
            # print("c_rate", c_rate)
            # print("recon_s", recon_s)

    print("save txt")
    dir = "result/TD3/platform"
    data = "0704"
    redir = os.path.join(dir, data)
    if not os.path.exists(redir):
        os.mkdir(redir)
    print("redir", redir)
    title1 = "Reward_td3_platform_embedding_nopre_relable_"
    title2 = "Reward_100_td3_platform_embedding_nopre_relable_"
    title3 = "Test_Reward_100_td3_platform_embedding_nopre_relable_"
    title4 = "Test_epioside_step_100_td3_platform_embedding_nopre_relable_"

    np.savetxt(os.path.join(redir, title1 + "{}".format(str(args.seed) + ".csv")), Reward, delimiter=',')
    np.savetxt(os.path.join(redir, title2 + "{}".format(str(args.seed) + ".csv")), Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title3 + "{}".format(str(args.seed) + ".csv")), Test_Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title4 + "{}".format(str(args.seed) + ".csv")), Test_epioside_step_100,
               delimiter=',')


def vae_train(action_rep, train_step, replay_buffer, batch_size, save_dir, vae_save_model, embed_lr):
    '''
    action_rep: action representation model todo
    train_step: 采样的总轮数，控制训练的轮数
    replay_buffer: 缓冲区
    batch_size: 训练batch
    save_dir: 存储的目录
    vae_save_model: 是否保存vae模型
    embed_lr: embed学习率
    '''
    initial_losses = [] # 保存每次训练的损失均值
    for counter in range(int(train_step) + 10):
        losses = [] # 这里估计是为了方便后续能够计算loss mean平均值
        state, discrete_action, parameter_action, all_parameter_action, discrete_emb, parameter_emb, next_state, state_next_state, reward, not_done = replay_buffer.sample(
            batch_size)
        # 完成VAE重建模型的训练，vae重建损失、观察变化损失、连续动作重建损失、KL约束散度损失 以上损失都只是标量值，估计只是为了记录 
        vae_loss, recon_loss_s, recon_loss_c, KL_loss = action_rep.unsupervised_loss(state,
                                                                                     discrete_action.reshape(1,
                                                                                                             -1).squeeze().long(),
                                                                                     parameter_action,
                                                                                     state_next_state,
                                                                                     batch_size, embed_lr)
        losses.append(vae_loss)
        initial_losses.append(np.mean(losses))

        if counter % 100 == 0 and counter >= 100:
            # print("load discrete embedding", action_rep.discrete_embedding())
            print("vae_loss, recon_loss_s, recon_loss_c, KL_loss", vae_loss, recon_loss_s, recon_loss_c, KL_loss)
            print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-50:])))
            print("discrete embedding", action_rep.discrete_embedding())

        # Terminate initial phase once action representations have converged.
        # len(initial_losses) >= train_step：确保至少训练了train_step轮（在代码中通常是5000轮）
        # np.mean(initial_losses[-5:]) + 1e-5 >= np.mean(initial_losses[-10:])：最近5次的平均损失 vs 最近10次的平均损失、添加小的容忍度 1e-5 避免数值精度问题
        # 如果损失还在下降那么最近5次的损失一定小于最近10次的损失，说明模型还在学习
        # 如果损失稳定了，那么最近5次加上一个小值则肯定大于最近10次的损失，则退出训练
        if len(initial_losses) >= train_step and np.mean(initial_losses[-5:]) + 1e-5 >= np.mean(initial_losses[-10:]):
            # print("vae_loss, recon_loss_s, recon_loss_c, KL_loss", vae_loss, recon_loss_s, recon_loss_c, KL_loss)
            # print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-50:])))
            # print("Converged...", len(initial_losses))
            break
        if vae_save_model:
            if counter % 1000 == 0 and counter >= 1000:
                title = "vae" + "{}".format(str(counter))
                action_rep.save(title, save_dir)
                print("vae save model")

    state_, discrete_action_, parameter_action_, all_parameter_action, discrete_emb, parameter_emb, next_state, state_next_state_, reward, not_done = replay_buffer.sample(
        batch_size=5000)
    # 离散动作潜在空间的边界范围、重建观察差值损失
    c_rate, recon_s = action_rep.get_c_rate(state_, discrete_action_.reshape(1, -1).squeeze().long(), parameter_action_,
                                            state_next_state_, batch_size=5000, range_rate=2)
    return c_rate, recon_s


def count_boundary(c_rate):
    median = (c_rate[0] - c_rate[1]) / 2 # 计算中值
    offset = c_rate[0] - 1 * median # 计算距离中值的偏移量
    return median, offset


def true_parameter_action(parameter_action, c_rate):
    # 参数TD3动作潜入空间的值映射到VAE真实的动作参数范围
    # 转换公式：
    # true_param = normalized_param * scale + offset
    # todo 后续回头看看vae部分c_rate是如何对应的
    parameter_action_ = copy.deepcopy(parameter_action)
    for i in range(len(parameter_action)):
        median, offset = count_boundary(c_rate[i])
        parameter_action_[i] = parameter_action_[i] * median + offset
    return parameter_action_


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="P-TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default='Platform-v0')  # platform goal HFO
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=128, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=500, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_episodes", default=50000, type=int)  # Max time steps to run environment
    parser.add_argument("--max_embedding_episodes", default=1e5, type=int)  # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=200000, type=float)  # Max time steps to run environment for

    parser.add_argument("--epsilon_steps", default=1000, type=int)  # Max time steps to epsilon environment
    parser.add_argument("--expl_noise_initial", default=1.0)  # Std of Gaussian exploration noise 1.0
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise 0.1

    parser.add_argument("--relable_steps", default=1000, type=int)  # Max time steps relable
    parser.add_argument("--relable_initial", default=1.0)  #
    parser.add_argument("--relable_final", default=0.0)  #

    parser.add_argument("--batch_size", default=128, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.1)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    for i in range(0, 5):
        args.seed = i
        # 运行5次
        run(args)
