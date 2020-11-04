import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import ACKTR

from stable_baselines.common.callbacks import CheckpointCallback

from stable_baselines.common.policies import MlpLstmPolicy, FeedForwardPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv

from stable_baselines import PPO2, A2C
import pandas as pd
import gym
from gym import spaces
import numpy as np
from scipy.stats import truncnorm
import pandas as pd

data2018 = pd.read_csv('2018_node_set1.csv')
data2019 = pd.read_csv('2019_node_set1.csv')
data = data2018.append(data2019)
data = data.reset_index().drop(['index'], axis=1)
days = pd.unique(data['Date'])

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, data=data)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__ == '__main__':
    env_id = 'gym_custom:foo-v0'
    num_cpu = 4  # Number of processes to use
    env = gym.make('gym_custom:foo-v0', data=data)
    env = DummyVecEnv([lambda: env])
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
    chkpt = CheckpointCallback(save_freq=100, save_path="./checkpoints", name_prefix="MLPLNLSTM_chkpt_")
    if False:
        model = PPO2.load("MLPLSTM")  # PPO2(MlpLstmPolicy, env, verbose=1, nminibatches=1)
        obs = env.reset()
        while True:
            for i in range(500):
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
            obs = env.reset()
    for episode in range(5000):
        model = A2C(MlpLnLstmPolicy, env, verbose=1)
        model.learn(total_timesteps=288, callback=chkpt)
        obs = env.reset()
    model.save("MLPLNLSTM")
    env.close()
