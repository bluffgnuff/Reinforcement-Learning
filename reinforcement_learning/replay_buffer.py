# import heapq as heap
import numpy as np
import scipy.stats as stats
import heapq


# We set a time to haepify to sort the buffer every K time step.
class PrioritizedExperienceReplayRankBased:
    """
    replay_buffer       - contains the tuples (TD_error, transaction_id, experience)
    max_buffer_size     - it's the max size of the buffer, over which before add an experience one is remove
    alpha               - the alpha parameter used to calculate the probability of the i-th element P(i) to be sampled
    self.max_td_error   - max td in the buffer

    Old
    time_to_haepify - time steps before sort the structure
    """

    def __init__(self, max_buffer_size, alpha):
        self.max_buffer_size = max_buffer_size
        # (TD, experience)
        self.replay_buffer = []
        self.alpha = alpha
        # The experience added has the maximum priority but once it sampled it will be updated with a more correct
        # value.
        self.max_td_error = 0
        # self.heapify_threshold = step_to_heapify  # here we stock the threshold to sort the buffer
        # self.step_to_heapify = step_to_heapify  # number of next steps before heapify

    def set_alpha(self, alpha):
        self.alpha = alpha

    # Add experience in the buffer mapping it with its last TD_error.
    # Heapq structure try to sort the elements of different tuple comparing from the first element of the tuple and
    # continuing with next element until the two tuples have an element different. We are interested in sorting by TD,
    # we don't care to sort on states, actions, or rewards. So, we use the transaction id to sort the transaction
    # that is older in the buffer.
    # The transaction_id of a transaction could be the -ith frame number of the whole training representing
    # when the transaction happened.
    # NB This approach avoids headppush fails when try to compare two states.
    def add_experience(self, transaction_id, experience):
        if len(self.replay_buffer) == self.max_buffer_size:
            self.remove_experience()

        # New experiences where td_error is unknown are set with the max td_error
        if len(self.replay_buffer) > 0:
            self.max_td_error = self.replay_buffer[0][0]
        heapq.heappush(self.replay_buffer, (-self.max_td_error, transaction_id, experience))
        # Old
        # self.step_to_heapify -= 1
        # if self.step_to_heapify == 0:
        #   self.replay_buffer.sort(key=lambda el: el[0], reverse=True)
        #   self.step_to_heapify = self.heapify_threshold

    # Remove experience from the buffer
    def remove_experience(self, index=-1):
        self.replay_buffer.pop(index)

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
        transaction_id = []

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
            transaction_id.append(self.replay_buffer[index][1])
            experiences.append(self.replay_buffer[index][2])

        # Normalization step
        max_weight = max(importance_sampling_weights)
        importance_sampling_weights_normalized = np.divide(importance_sampling_weights, max_weight)
        return indexes, transaction_id, experiences, importance_sampling_weights_normalized

    def update_td_error(self, index, td_error, transaction_id):
        experience = self.replay_buffer[index][2]
        self.remove_experience(index)
        heapq.heappush(self.replay_buffer, (-td_error, transaction_id, experience))
