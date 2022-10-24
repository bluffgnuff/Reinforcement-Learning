import math as mt

import tensorflow as tf
import numpy as np


class DuelDQNAgent:

    # We keep the creation model outside the agent to ensure a fine-grained control on it
    def __init__(self, env, model, policy, model_target=None, optimizer=None, replay_buffer=None):
        self.env = env
        self.model_primary = model
        self.model_target = model_target
        self.policy = policy
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer

    def set_policy(self, policy):
        self.policy = policy

    # Execs one action receiving in input the environment, its state, the current episode.
    # If training its true add the experience in the replay buffer
    def play_one_step(self, state):
        action = self.policy.get_action(state)
        # print("action {}".format(action))
        next_state, reward, done, info = self.env.step(action)
        return action, reward, next_state, done, info

    # Play
    def play(self):
        state = self.env.reset()
        steps = 0
        cumulative_reward = 0
        while True:
            action, reward, next_state, done, info = self.play_one_step(state)
            cumulative_reward += reward
            if done:
                print("DONE number of steps: {} reward:  {}".format(steps, cumulative_reward))
                break
            steps += 1
            state = next_state
        return steps, cumulative_reward

    # Double DQN Training
    @staticmethod
    def gradient_clipping(gradients, clipping_value):
        clipped_gradients = [(tf.clip_by_norm(grad, clipping_value)) for grad in gradients]
        return clipped_gradients

    def weighted_gradient(self, best_on_target_q_values, importance_sampling_weights, states, loss_function, mask,
                          step_size=1):
        with tf.GradientTape() as tape:
            tape.watch(importance_sampling_weights)
            all_q_values = self.model_primary(states)
            q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)
            loss_value = loss_function(best_on_target_q_values, q_values)
            loss_corrected = loss_value * importance_sampling_weights * step_size
        grads = tape.gradient(loss_corrected, self.model_primary.trainable_variables)
        return grads, loss_value

    @staticmethod
    def rescale_grad(gradients, rescale_value, index):
        tensor_to_scale = gradients[index]
        rescaled_tensor = tensor_to_scale * rescale_value
        gradients[index] = rescaled_tensor
        return gradients

    # Collects samples of the previous experiences from the replay buffer
    # and use them to improve the weights update of the Neural Network.
    def double_dqn_training_step(self, batch_size, loss_function, discount_factor, clipping_value, beta, step_size=1):
        indexes, transaction_ids, experiences, importance_sampling_weights = self.replay_buffer.sample_experience(
            batch_size, beta)
        states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in experiences]
                                                                 ) for field_index in range(5)]

        action_space = self.env.action_space.n
        # Predict using the primary network
        next_q_values = self.model_primary.predict(next_states)
        next_q_values_target = self.model_target.predict(next_states)

        # Select the action that lead us to the higher next Q value
        best_actions = np.argmax(next_q_values, axis=1)
        best_action_mask = tf.one_hot(best_actions, action_space)

        next_q_value_target = tf.reduce_sum(next_q_values_target * best_action_mask, axis=1)
        best_on_target_q_values = (rewards + (1 - dones) * discount_factor * next_q_value_target)
        # best_on_target_q_values = best_on_target_q_values * (1-dones) - dones

        mask = tf.one_hot(actions, action_space)
        importance_sampling_weights = tf.convert_to_tensor(importance_sampling_weights, tf.float32)
        weighted_gradient, loss_value = self.weighted_gradient(best_on_target_q_values, importance_sampling_weights,
                                                               states, loss_function, mask, step_size)

        for index, td_error, transaction_id in zip(indexes, loss_value, transaction_ids):
            self.replay_buffer.update_td_error(index, abs(td_error), transaction_id)

        # We rescale the last convolutional layer to 1/sqrt(2) to balance the double backpropagation
        rescale_value = (1 / mt.sqrt(2))
        # The index of the last sequential layer
        index_gradient_to_rescale = 4
        rescaled_grads = self.rescale_grad(weighted_gradient, rescale_value, index_gradient_to_rescale)

        # Since we are in a custom loop we have to clip the gradient by hand, we can't delegate it to the optimizer
        clipped_gradients = self.gradient_clipping(rescaled_grads, clipping_value)
        # Application gradient descent trough optimizer
        self.optimizer.apply_gradients(zip(clipped_gradients, self.model_primary.trainable_variables))

    # We use the training step just when there is enough samples on the replay buffer
    def double_dqn_training(self, batch_size, loss_function, discount_factor, freq_replacement, training_freq,
                            clipping_value, beta_min, beta_max, max_episodes=600, max_steps=108000):
        rewards_stock = []
        steps_stock = []
        cumulative_steps = 0

        for episode in range(1, max_episodes + 1):
            state = self.env.reset()
            rewards = 0
            steps = 0
            beta = max(beta_min, (beta_max * episode / max_episodes))

            while True:
                action, reward, next_state, done, info = self.play_one_step(state)
                experience = [state, action, reward, next_state, done]
                rewards += reward
                self.replay_buffer.add_experience(cumulative_steps, experience)
                cumulative_steps += 1
                steps += 1

                if len(self.replay_buffer.replay_buffer) > batch_size and (cumulative_steps % training_freq) == 0:
                    self.double_dqn_training_step(batch_size, loss_function, discount_factor, clipping_value, beta)
                if (cumulative_steps % freq_replacement) == 0:
                    self.model_target.set_weights(self.model_primary.get_weights())
                if steps == max_steps:
                    print(
                        "ABORTED episode = {} number of steps = {} reward = {}".format(episode, steps, rewards))
                if done:
                    print(
                        "DONE episode = {} number of steps = {} reward = {}".format(episode, steps, rewards))
                if done or steps == max_steps:
                    rewards_stock.append(rewards)
                    steps_stock.append(steps)
                    break
                state = next_state

            self.policy.next_episode()

        return steps_stock, rewards_stock
