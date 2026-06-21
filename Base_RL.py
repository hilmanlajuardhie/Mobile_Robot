import gymnasium as gym
from stable_baselines3 import PPO

# Create environment with render mode
env = gym.make("CartPole-v1", render_mode="human")

# Train PPO agent (CPU recommended for MLP policy)
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=10000)

# Test agent
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
