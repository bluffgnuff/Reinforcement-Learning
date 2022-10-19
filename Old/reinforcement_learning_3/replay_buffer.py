# import heapq as heap
import numpy as np
import scipy.stats as stats


# We set a time to haepify to sort the buffer every K time step.
class PrioritizedExperienceReplayRankBased:
    """
    contains the tuples (TD_error, experience)
    replay_buffer --- it's the max size of the buffer, over which before add an experience one is remove
    max_buffer_size --- time step before sort the structure
    time_to_haepify --- the last time step
    mod_curr_step = 0  --- the alpha parameter used to calculate the probability of the i-th element P(i) to be sampled
    alpha -- alpha parameter
    """

    def __init__(self, max_buffer_size, step_to_heapify, alpha):
        self.max_buffer_size = max_buffer_size
        # (TD, experience)
        # Probably list is not the most efficient structure to use np array ?
        self.replay_buffer = []
        self.alpha = alpha
        self.heapify_threshold = step_to_heapify  # here we stock the threshold to sort the buffer
        self.step_to_heapify = step_to_heapify  # number of next steps before heapify
        self.max_td_error = 0

    def set_alpha(self, alpha):
        self.alpha = alpha

    # Add experience in the buffer mapping it with its last TD_error
    def add_experience(self, experience):
        if len(self.replay_buffer) == self.max_buffer_size:
            self.remove_experience()

        # New experience where td_error is unknown are set with the max td error
        # NB we are considering the max td error as the error of the experience in first position, but the buffer may
        # not have been sorted yet
        if len(self.replay_buffer) > 0:
            self.max_td_error = self.replay_buffer[0][0]

        self.replay_buffer.append((self.max_td_error, experience))
        self.step_to_heapify -= 1
        if self.step_to_heapify == 0:
            self.replay_buffer.sort(key=lambda el: el[0], reverse=True)
            self.step_to_heapify = self.heapify_threshold

    # Remove experience from the buffer
    def remove_experience(self):
        self.replay_buffer.pop(-1)

    @staticmethod
    def zip_f_sampling(alpha, n):
        x = np.arange(1, n + 1)
        weights = x ** (-alpha)
        weights /= weights.sum()
        zipf = stats.rv_discrete(values=(x, weights))
        return zipf.rvs() - 1

    # Get batch_size samples from the buffer; using the beta parameter to compute the importance sampling weight
    # Beta value can change while training we can delegate its control outside
    def sample_experience(self, batch_size, beta):
        experiences = []
        importance_sampling_weights = []
        n = len(self.replay_buffer) - 1
        indexes = []

        for i in range(0, batch_size):
            # Sample index and check the experience is not already present in the batch
            index = self.zip_f_sampling(self.alpha, n)
            while index in indexes:
                index = self.zip_f_sampling(self.alpha, n)
            indexes.append(index)
            # importance sampling weights computation
            rank = index + 1
            pj = 1 / rank
            importance_sampling_weights.append(((n * pj) ** (-beta)))
            experiences.append(self.replay_buffer[index][1])

        # Normalization step
        max_weight = max(importance_sampling_weights)
        importance_sampling_weights_normalized = np.divide(importance_sampling_weights, max_weight)
        return indexes, experiences, importance_sampling_weights_normalized

    def update_td_error(self, index, td_error):
        self.replay_buffer[index] = [td_error, self.replay_buffer[index][1]]
