import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from stable_baselines.common import set_global_seeds, make_vec_env



from stable_baselines.common.callbacks import CheckpointCallback

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec


# from stable_baselines.deepq.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy

from stable_baselines import DQN, DDPG, HER

from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper

import gym
from gym import spaces
from scipy.stats import truncnorm
import pandas as pd
import numpy as np


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


data = 0
load_type = "csv" if False else "npy"

if load_type == 'csv':
    if os.path.exists('cleaned_data2020_v2.csv'):
        data = pd.read_csv('cleaned_data2020_v2.csv')
        data = data.reset_index().drop(['index'], axis=1)
    else:
       # data2018 = pd.read_csv('2018_node_set1.csv')
       # data2019 = pd.read_csv('2019_node_set1.csv')
        data2020 = pd.read_csv('2020_6_nodes.csv')
        data2020a = pd.read_csv('2020_3_nodes.csv')
        data2020b = pd.read_csv('2020_node_set4.csv')
        data2020c = pd.read_csv('2020_node_set5.csv')
        data = data2020.append(data2020a)
        data = data.append(data2020b)
        print(len(data))
        data = data.reset_index().drop(['index'], axis=1)
        outliers = data[data['Value'] > 200]
        throwaway_nodes = pd.unique(outliers['Node'])
        for node in throwaway_nodes:
            bad_days_for_node = pd.unique(outliers[outliers['Node']==node]['Date'])
            for day in bad_days_for_node:
                data = data[((data['Node'] != node) | (data['Date'] != day))]
            print("NextNode")

        print(len(data))
        data.to_csv('cleaned_data2020_v2.csv')
elif load_type == 'npy':
    data = np.load(r'E:\Projects\ProjectX\large_data_cleaned.npy')


def fixed_time(obs):
    obs = obs[-1]
    if obs[-1] >= 181 and obs[-1] < 181+36:
        return [2], obs
    elif obs[-1] >= 252 and obs[-1] < 252+36:
        return [0], obs
    else:
        return [1], obs


if __name__ == '__main__':
    env_id = 'gym_custom:fooCont-v0'
    num_cpu = 8  # Number of processes to use

    np.random.shuffle(data)
    separator = int(len(data)*.8)

    train_data = data[:separator]
    test_data = data[separator:]

    #env = make_vec_env(env_id, n_envs=8, seed=0)
    env = gym.make(env_id, data=data)
    env = DummyVecEnv([lambda: env])


    chkpt = CheckpointCallback(save_freq=288*500, save_path="./checkpoints", name_prefix="DDPG_Large")

    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    model = DDPG(LnMlpPolicy, env,
                 verbose=1,
                 gamma=0.995,
                 buffer_size=100000,
                 param_noise=param_noise,
                 action_noise= action_noise,
                 enable_popart=True,
    policy_kwargs={
       'layers': [32, 32, 32, 32, 32, 32]
    })# higher weight on future planning

    # model = DQN.load('checkpoints/DQN_LNLMLP_hist_PER_v1_cln_1296000_steps.zip', env=env)
    model.learn(total_timesteps=int(700*300*30*3), callback=chkpt)
    model.save("DDPG_Large")

    #del model # not needed, but good for saving / loading

    model = DQN.load("DDPG_Large.zip")  # PPO2(MlpLstmPolicy, env, verbose=1, nminibatches=1)
    obs = env.reset()
    p_dqn = []
    p_fixed = []

    p_solar = []
    p_solar2 = []
    d = []
    d2 = []
    for ep in range(1000):
        info = {}
        for i in range(288):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)

        p_dqn.append(info[0]['p'])
        obs = env.reset()



    env.close()
