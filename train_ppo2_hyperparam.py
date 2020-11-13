import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv

from stable_baselines import DQN, DDPG, TD3, PPO2


from stable_baselines.td3.policies import MlpPolicy as td3MLP
from stable_baselines.td3.policies import LnMlpPolicy as td3LnMlp

from stable_baselines.common.policies import MlpPolicy as commonMlp
from stable_baselines.common.policies import MlpLnLstmPolicy as commonMlpLstm

from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlp
from stable_baselines.ddpg.policies import LnMlpPolicy as ddpgLnMlp


import optuna
import gym
import joblib

import numpy as np



data = np.load(r'large_data_cleaned.npy')
np.random.seed(42)
np.random.shuffle(data)

separator = int(len(data)*.95)

train_data = data[:separator]
test_data = data[separator:]

def optimize_ppo2(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }
def sample_ddpg_params(trial):
    """
    Sampler for DDPG hyperparams.
    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    # actor_lr = trial.suggest_loguniform('actor_lr', 1e-5, 1)
    # critic_lr = trial.suggest_loguniform('critic_lr', 1e-5, 1)
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    buffer_size = trial.suggest_categorical('memory_limit', [int(1e4), int(1e5), int(1e6)])
    noise_type = trial.suggest_categorical('noise_type', ['ornstein-uhlenbeck', 'normal', 'adaptive-param'])
    noise_std = trial.suggest_uniform('noise_std', 0, 1)
    normalize_observations = trial.suggest_categorical('normalize_observations', [True, False])
    normalize_returns = trial.suggest_categorical('normalize_returns', [True, False])

    hyperparams = {
        'gamma': gamma,
        'actor_lr': learning_rate,
        'critic_lr': learning_rate,
        'batch_size': batch_size,
        'memory_limit': buffer_size,
        'normalize_observations': normalize_observations,
        'normalize_returns': normalize_returns
    }

    if noise_type == 'adaptive-param':
        hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                            desired_action_stddev=noise_std)
        # Apply layer normalization when using parameter perturbation
        hyperparams['policy_kwargs'] = dict(layer_norm=True)
    elif noise_type == 'normal':
        hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(1),
                                                        sigma=noise_std * np.ones(1))
    elif noise_type == 'ornstein-uhlenbeck':
        hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(1),
                                                                   sigma=noise_std * np.ones(1))
    return hyperparams

def sample_td3_params(trial):
    """
    Sampler for TD3 hyperparams.
    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical('buffer_size', [int(1e4), int(1e5), int(1e6)])
    train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 1000, 2000])
    gradient_steps = train_freq
    noise_type = trial.suggest_categorical('noise_type', ['ornstein-uhlenbeck', 'normal'])
    noise_std = trial.suggest_uniform('noise_std', 0, 1)

    hyperparams = {
        'gamma': gamma,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
    }

    if noise_type == 'normal':
        hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(1),
                                                        sigma=noise_std * np.ones(1))
    elif noise_type == 'ornstein-uhlenbeck':
        hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(1),
                                                                   sigma=noise_std * np.ones(1))

    return hyperparams

class Objective(object):
    def __init__(self, train, test):
        # Hold this implementation specific arguments as the fields of the class.
        self.train_data = train_data
        self.test_data = test_data
        self.n_actions = 1

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        env_id = 'gym_custom:fooCont-v0'
        env = gym.make(env_id, data=self.train_data)
        env = DummyVecEnv([lambda: env])

        algo = trial.suggest_categorical('algo', ['PPO2', 'DQN', 'DDPG', 'TD3'])
        model = 0
        if algo == 'PPO2':

            policy = trial.suggest_categorical('policy', [commonMlp, commonMlpLstm])
            model_params = optimize_ppo2(trial)

            model = PPO2(policy, env, verbose=0, nminibatches=1, **model_params)
            model.learn(1000000)

        elif algo == 'DDPG':
            policy = trial.suggest_categorical('policy', [ddpgMlp, ddpgLnMlp])
            model_params = sample_ddpg_params(trial)

            model= DDPG(policy, env, verbose=0, **model_params)
            model.learn(1000000)

        elif algo == 'TD3':
            policy = trial.suggest_categorical('policy', [td3MLP, td3LnMlp])
            model_params = sample_td3_params(trial)

            model = TD3(policy, env, verbose=0, **model_params)
            model.learn(1000000)

        rewards = []
        reward_sum = 0.0
        env = gym.make(env_id, data=self.test_data)
        env = DummyVecEnv([lambda: env])

        obs = env.reset()
        for i in range(100):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward

            if done:
                rewards.append(reward_sum)
                reward_sum = 0.0
                obs = env.reset()
        reward_total = np.mean(rewards)
        return reward_total * -1

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(Objective(train_data, test_data), n_trials=500)
    joblib.dump(study, 'study.pkl')
    print('Best trial until now:')
    print(' Value: ', study.best_trial.value)
    print(' Params: ')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')
