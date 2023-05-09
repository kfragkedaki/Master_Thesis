import gym

env = gym.make("Taxi-v3", render_mode="human")

env.reset()
env.render()
