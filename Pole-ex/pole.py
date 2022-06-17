import gym
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42, return_info=True)

for _ in range(100000):
    ## Permit to show 
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print( "observation=", observation, "info=", info, "reward=", reward, "\n")
    if done:
        observation, info = env.reset(return_info=True)

env.close()