import gym
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42, return_info=True)

for _ in range(100000):
    ## Permit to render the enviroment 
    env.render()
    
    action = env.action_space.sample()
    
    # Observation is the object representing an observation of the environment. For example, the state of the robot in the terrain;
    # Info collect informations usefull for debugging;
    # Reward are the rewards gained by the previous action. For example, the reward gained by a robot on successfully moving forward;
    # Done is the Boolean; when it is true, it indicates that the episode has completed (that is, the robot learned to walk or failed completely). Once the episode has completed, we can initialize the environment for the next episode using env.reset();
    
    observation, reward, done, info = env.step(action)
    print( "observation=", observation, "info=", info, "reward=", reward, "\n")
    if done:
        observation, info = env.reset(return_info=True)

env.close()