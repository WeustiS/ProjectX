import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from stable_baselines.common import set_global_seeds, make_vec_env



from stable_baselines.common.callbacks import CheckpointCallback

from stable_baselines.common.policies import MlpLstmPolicy, FeedForwardPolicy, MlpLnLstmPolicy
from stable_baselines.deepq.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv

from stable_baselines import PPO2, A2C, DQN, DDPG, ACER, ACKTR

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
    data = np.load('superclean.npy')

print(data)

def fixed_time(obs):
    obs = obs[-1]
    if obs[-1] >= 181 and obs[-1] < 181+36:
        return [2], obs
    elif obs[-1] >= 252 and obs[-1] < 252+36:
        return [0], obs
    else:
        return [1], obs


if __name__ == '__main__':
    env_id = 'gym_custom:foo-v0'
    num_cpu = 8  # Number of processes to use


    #env = make_vec_env(env_id, n_envs=8, seed=0)
    env = gym.make('gym_custom:foo-v0', data=data)
    env = DummyVecEnv([lambda: env])

    env2 = gym.make('gym_custom:foo-v0', data=data)
    env2 = DummyVecEnv([lambda: env2])


   # # Stable Baselines provides you with make_vec_env() helper
    # which do es exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
    chkpt = CheckpointCallback(save_freq=288*500, save_path="./checkpoints", name_prefix="DQN_LNLMLP_PER_v4_cln")


    #model = PPO2(MlpLnLstmPolicy, env, verbose=1, nminibatches=8, learning_rate=.0025     )
    model = DQN(LnMlpPolicy, env, verbose=1,
                gamma=0.99,
                exploration_fraction=.15,
                buffer_size=75000,
                learning_starts=288*50,
                prioritized_replay=True,
                policy_kwargs={
       'layers': [256, 128, 64, 32]
    })# higher weight on future planning

    # model = DQN.load('checkpoints/DQN_LNLMLP_hist_PER_v1_cln_1296000_steps.zip', env=env)
    #model.learn(total_timesteps=int(700*300*30), callback=chkpt)
    #model.save("DQN_LNLMLP_PER_v4_cln")

    #del model # not needed, but good for saving / loading

    model = DQN.load("DQN_LNLMLP_hist_PER.zip")  # PPO2(MlpLstmPolicy, env, verbose=1, nminibatches=1)
    obs = env.reset()
    obs2 = env2.reset()
    p_dqn = []
    p_fixed = []

    p_solar = []
    p_solar2 = []
    d = []
    d2 = []
    for ep in range(1000):
        for i in range(288):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)

            action2, _states2 = fixed_time(obs2) # model.predict(obs2)
            obs2, rewards2, dones2, info2 = env2.step(action2)


        p_dqn.append(info[0]['p'])
        p_fixed.append(info2[0]['p'])
        p_solar.append(info[0]['s'])
        p_solar2.append(info2[0]['s'])
        d.append(info[0]['d'])
        d2.append(info2[0]['d'])
        obs = env.reset()
        obs2 = env2.reset()



    env.close()
