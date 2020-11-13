from gym.envs.registration import register

register(
    id='foo-v0',
    entry_point='gym_custom.envs:FooEnv',
)


register(
    id='fooCont-v0',
    entry_point='gym_custom.envs:FooEnvCont',
)