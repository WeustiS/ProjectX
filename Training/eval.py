import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from stable_baselines.common import set_global_seeds, make_vec_env

import matplotlib.pyplot as plt

from stable_baselines.common.callbacks import CheckpointCallback

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec


# from stable_baselines.deepq.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines.td3.policies import MlpPolicy

from stable_baselines import DQN, DDPG, HER, TD3

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

data = np.load(r'large_data_cleaned.npy')
def fixed_time(obs):
    obs = obs[-1]
    if obs[-2] >= 140 and obs[-2] < 140+34:
        return [1], obs
    elif obs[-2] >= 224 and obs[-2] < 224+36:
        return [-1], obs
    else:
        return [0], obs


env_id = 'gym_custom:fooCont-v0'
data = np.load(r'large_data_cleaned.npy')
env = gym.make(env_id, data=data)
env = DummyVecEnv([lambda: env])
obs = env.reset()

env2 = gym.make(env_id, data=data)
env2 = DummyVecEnv([lambda: env2])
obs2 = env2.reset()
p_td3 = []
p_fixed = []

p_solar = []
p_solar2 = []
d = []
d2 = []
model = TD3.load("TD3_Large_3.zip")

for ep in range(100):
    info = {}
    for i in range(276):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        action2, _states2 = fixed_time(obs2)
        obs2, rewards2, dones2, info2 = env2.step(action2)

    p_td3.append(info[0]['p'])
    p_fixed.append(info2[0]['p'])
    obs = env.reset()
    obs2 = env2.reset()
    break

env.close()
env2.close()


def c(x):
    try:
        return x[0]
    except:
        return x


profits = np.array(list(map(c, p_td3)))
# run in console to work with values
