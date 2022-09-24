import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
env = gym.make("CartPole-v1")
input_shape = [4] #=env.observation_space.shape
n_outputs = 2 # ==env.action_space.n

model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=input_shape),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

replay_buffer = deque(maxlen=2000)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch]) for field_index in range(5)]
    return states, actions, rewards, next_states, dones

batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error

import tensorflow as tf
def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1-dones)*discount_factor*max_next_Q_values)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values*mask, axis=1, keepdims=True)
        loss = loss_fn(target_Q_values, Q_values)
    grads = tape.gradient(loss, model.trainable_variables)
    print("Grad", grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


for episode in range(600):
    obs = env.reset()
    for step in range(200):
        epsilon = max(1- episode/500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        env.render()
        if done:
            print("done step ", step,)
            print("reward = ", reward)
            break
        if episode>1:
            training_step(batch_size)