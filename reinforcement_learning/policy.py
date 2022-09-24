import numpy as np


class Policy:
    def __init__(self, model):
        self.model = model
        pass

    def get_action(self, state):
        pass


class EpsilonGreedyPolicy(Policy):

    def __init__(self, model, action_space_size, episodes=1, min_epsilon=0):
        super().__init__(model)
        self.action_space_size = action_space_size
        self.min_epsilon = min_epsilon
        self.episode = 1
        self.episodes = episodes

    def get_action(self, state):
        epsilon = max(1 - self.episode / self.episodes, self.min_epsilon)
        if self.episode == self.episodes:
            self.reset_episodes()
        random = np.random.rand()
        if random < epsilon:
            action = np.random.randint(self.action_space_size)
            return action
        else:
            q_values = self.model.predict(state[np.newaxis])
            action = np.argmax(q_values[0])
            return action

    def next_episode(self):
        self.episode += 1

    def reset_episodes(self):
        self.episode = 0
