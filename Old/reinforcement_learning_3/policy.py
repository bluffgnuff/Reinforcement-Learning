import numpy as np
import tensorflow as tf


class EpsilonGreedyPolicy:

    def __init__(self, model, action_space_size, episodes=1, min_epsilon=0):
        self.model = model
        self.action_space_size = action_space_size
        self.min_epsilon = min_epsilon
        self.episode = 1
        self.episodes = episodes

    def get_action(self, state):
        epsilon = max(1 - self.episode / self.episodes, self.min_epsilon)
        random = np.random.random()
        # print(random, epsilon)
        if random < epsilon:
            action = np.random.randint(self.action_space_size)
            return action
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
            return action

    def next_episode(self):
        self.episode += 1

    def reset_episodes(self):
        self.episode = 1
