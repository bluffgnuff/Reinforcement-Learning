from numpy import random
import tensorflow as tf


class EpsilonGreedyPolicy:

    def __init__(self, model, action_space_size, episodes=1, min_epsilon=0, decay_rate=1.35):
        self.model = model
        self.action_space_size = action_space_size
        self.min_epsilon = min_epsilon
        self.episode = 1
        self.episodes = episodes
        self.decay_rate = decay_rate

    def get_action(self, state):
        epsilon_decay = (self.episode / self.episodes)*self.decay_rate
        epsilon = max(1 - epsilon_decay, self.min_epsilon)
        rnd = random.random()
        # print(random, epsilon)
        if rnd < epsilon:
            action = random.randint(self.action_space_size)
            return action
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
            return action

    def next_episode(self):
        self.episode += 1

    def reset_episodes(self):
        self.episode = 1
