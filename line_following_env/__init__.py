from gymnasium.envs.registration import register
from .line_following_env import LineFollowingEnv

register(
    id='LineFollowing-v0',
    entry_point='line_following_env:LineFollowingEnv',
    max_episode_steps=1000,
)
