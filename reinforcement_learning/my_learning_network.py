from gym import envs

from reinforcement_learning.agent import DuelDQNAgent
from reinforcement_learning.policy import EpsilonGreedyPolicy
from reinforcement_learning.replay_buffer import PrioritizedExperienceReplayRankBased

# Searching for available environments
game_name = "BeamRider"
all_envs = envs.registry.values()
env_ids = [env_spec.id for env_spec in all_envs]

for id in sorted(env_ids):
    if game_name in id:
        print(id)

# Environment Configuration
import matplotlib.pyplot as plt
# from gym import envs
from gym.wrappers import AtariPreprocessing

# Make Parameters:
game_name = "BeamRider"
game_mode = "NoFrameskip"  # [Deterministic | NoFrameskip | ram | ramDeterministic | ramNoFrameskip ]
game_version = "v4"  # [v0 | v4 | v5]
env_name = '{}{}-{}'.format(game_name, game_mode, game_version)
env_render_mode = 'human'  # [human | rgb_array]
env_frame_skip = 4

env = envs.make(env_name, render_mode=env_render_mode)

print("Environment observation: ", env.observation_space)
print("Environment action space: ", env.action_space)
print("Action list: ", env.unwrapped.get_action_meanings())

# plt.imshow(env.render(mode="rgb_array"))

env_prep = AtariPreprocessing(env, frame_skip=env_frame_skip, grayscale_obs=True, noop_max=30)
env_prep.reset()
print("Environment preprocessed observation: ", env_prep.observation_space.shape)
print("Environment preprocessed action space: ", env.action_space)
# plt.imshow(env_prep.render(mode=env_render_mode))

# Neural Network Creation
from tensorflow import math
from tensorflow.keras import layers
from tensorflow.keras import Model


def create_dueling_model(input_shape, number_actions):
    inputs = layers.Input(shape=input_shape)

    # Convolutions on the frames on the screen
    layer1 = layers.Conv1D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv1D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv1D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)

    value_stream_1 = layers.Dense(512)(layer4)
    value_stream_2 = layers.Dense(1)(value_stream_1)  # scalar output size

    advantage_stream_1 = layers.Dense(512)(layer4)
    advantage_stream_2 = layers.Dense(number_actions)(advantage_stream_1)  # output size equal to the actions available

    # Combination of the streams: a Q value for each state
    q_values = value_stream_2 + math.subtract(advantage_stream_2, math.reduce_mean(advantage_stream_2, axis=1,
                                                                                   keepdims=True))

    # Alternative q_value
    # q_value = value_stream_2 + (advantage_stream_2 - backend.max(advantage_stream_2, axis=1, keepdims=True))
    return Model(inputs=[inputs], outputs=[q_values])


# Network Parameters
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model

# Environment info
input_shape = env_prep.observation_space.shape
actions_number = env_prep.action_space.n

# Model persistent file
primary_model_file_name = "dueling_primary_model.h5"

# Training Parameters
loss_function = losses.mean_squared_error
batch_size = 32
discount_factor = 0.95
learning_rate = 6.25e-5
episodes = 600
clipping_value = 10

# Dual DQN Training
freq_replacement = 50

# Replay buffer parameters
buffer_size = 2000
step_to_heapify = 50
alpha = 0.5
beta = 0.5

# Policy parameters
min_epsilon = 0.001


# Plot result
def plot_result(x_label, y_label, x, y):
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.plot(x, y)
    plt.show()

    return


# Run Training
from pathlib import Path

# Model creation
file_primary = Path(primary_model_file_name)
if file_primary.exists():
    model = load_model(primary_model_file_name)
else:
    model = create_dueling_model(input_shape, actions_number)

model_target = create_dueling_model(input_shape, actions_number)
# Print a summary about the model
print(model.summary())

# Setting the optimizer with the clipping to have the norm <= 10
optimizer = optimizers.Adam(learning_rate=learning_rate)
policy_training = EpsilonGreedyPolicy(model, actions_number, episodes, min_epsilon)
replay_buffer = PrioritizedExperienceReplayRankBased(buffer_size, step_to_heapify, alpha, beta)
agent = DuelDQNAgent(env_prep, model, model_target, policy_training, optimizer, replay_buffer)
steps, rewards = agent.double_dqn_training(batch_size, loss_function, discount_factor, freq_replacement, clipping_value,
                                           episodes)

# env_prep.close()
# model_primary.save(primary_model_file_name)
# plot_result("Episode", "Steps", max_episodes, steps)

# Play
# policy_training = EpsilonGreedyPolicy(model, actions_number, episodes = 0 , min_epsilon)
# agent.set_policy(policy_training)
# steps, reward = agent.play(env)
# plot_result("Episode", "Rewards", max_episodes, rewards)
