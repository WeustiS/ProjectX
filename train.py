import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env



from stable_baselines.common.callbacks import CheckpointCallback

from stable_baselines.common.policies import MlpLstmPolicy, FeedForwardPolicy, MlpLnLstmPolicy
from stable_baselines.deepq.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv

from stable_baselines import PPO2, A2C, DQN, DDPG, ACER, ACKTR

import pandas as pd
import gym
from gym import spaces
import numpy as np
from scipy.stats import truncnorm
import pandas as pd



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

print(data)

days = pd.unique(data['Date'])



if __name__ == '__main__':
    env_id = 'gym_custom:foo-v0'
    num_cpu = 8  # Number of processes to use


    #env = make_vec_env(env_id, n_envs=8, seed=0)
    env = gym.make('gym_custom:foo-v0', data=data)
    env = DummyVecEnv([lambda: env])


   # # Stable Baselines provides you with make_vec_env() helper
    # which do es exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
    chkpt = CheckpointCallback(save_freq=288*500, save_path="./checkpoints", name_prefix="DQN_LNLMLP_hist_PER_v1")


    #model = PPO2(MlpLnLstmPolicy, env, verbose=1, nminibatches=8, learning_rate=.0025     )
    model = DQN(LnMlpPolicy, env, verbose=1, gamma=0.996,   exploration_fraction=.25,
                buffer_size=100000,
                learning_starts=288*50,
                prioritized_replay=True,
                policy_kwargs={
       'layers': [256, 128, 64, 32]
    })# higher weight on future planning

    #model = DQN.load('DQN_LNMLP_11_5_4pm', env=env)
    model.learn(total_timesteps=865656*15, callback=chkpt)
    model.save("DQN_LNLMLP_hist_PER")

    #del model # not needed, but good for saving / loading

    #model = DQN.load("DQN_LNMLP")  # PPO2(MlpLstmPolicy, env, verbose=1, nminibatches=1)
    obs = env.reset()
    while input("New episode?"):
        for i in range(288):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
        obs = env.reset()

    env.close()
