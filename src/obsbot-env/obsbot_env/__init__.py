from gym.envs.registration import register

register(
        id='obsbot2d-v0',
        entry_point='obsbot_env.envs:ObsBot2D',
)
register(
        id='obsbot2dpoint-v0',
        entry_point='obsbot_env.envs:ObsBot2DPoint',
)
