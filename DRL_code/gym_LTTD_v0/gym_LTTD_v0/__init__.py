from gymnasium.envs.registration import register

register(
    id="LTTD-v0",
    entry_point="gym_LTTD_v0.envs:lttdENV",
    max_episode_steps=300,
)