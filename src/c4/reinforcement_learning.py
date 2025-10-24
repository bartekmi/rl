import gymnasium as gym
from stable_baselines3 import PPO
from c4.c4_env import ConnectFourEnv

env = gym.wrappers.TimeLimit(ConnectFourEnv(), max_episode_steps=42)   # type: ignore

model = PPO(
    "MlpPolicy",
    env,
    ent_coef=0.1,
    n_steps=4096,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=3e-4,
    vf_coef=0.5,
    clip_range=0.2,
    batch_size=256,
    n_epochs=10,
    verbose=1,
)
model.learn(total_timesteps=25000)

# Watch it play
obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(int(action))
    env.render()
