from gym import envs

# TODO change position in Jupyter
from reinforcement_learning.agent import DuelDQNAgent
from reinforcement_learning.policy import EpsilonGreedyPolicy
from reinforcement_learning.replay_buffer import PrioritizedExperienceReplayRankBased

# Searching for available environments
game_name = "Phoenix"
all_envs = envs.registry.values()
env_ids = [env_spec.id for env_spec in all_envs]

for id in sorted(env_ids):
    if game_name in id:
        print(id)

# Environment Configuration
# TODO change position in Jupyter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# from gym import envs
from gym.wrappers import AtariPreprocessing

# Make Parameters:
game_name = "Phoenix"
game_mode = "NoFrameskip"  # [Deterministic | NoFrameskip | ram | ramDeterministic | ramNoFrameskip ]
game_version = "v4"  # [v0 | v4 | v5]
env_name = '{}{}-{}'.format(game_name, game_mode, game_version)
env_render_mode = 'rgb_array'  # [human | rgb_array]
env_frame_skip = 4

env = envs.make(env_name, render_mode=env_render_mode)

print("Environment observation: ", env.observation_space)
print("Environment action space: ", env.action_space)
print("Action list: ", env.unwrapped.get_action_meanings())


env_prep = AtariPreprocessing(env, frame_skip=env_frame_skip, grayscale_obs=True, noop_max=30)
env_prep.reset()
print("Environment preprocessed observation: ", env_prep.observation_space.shape)
print("Environment preprocessed action space: ", env_prep.action_space)

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
primary_model_file_name = "{}_dueling_model".format(game_name)

# Training Parameters
loss_function = losses.mean_squared_error
batch_size = 32
discount_factor = 0.95
learning_rate = 6.25e-5
episodes = 50
clipping_value = 10

# Dual DQN Training
freq_replacement = 50

# Replay buffer parameters
buffer_size = 2000
step_to_heapify = 50
alpha = 0.5
beta_max = 1
beta_min = 0.5

# Policy parameters
min_epsilon = 0.001


# Plot result
def plot_result(x_label, y_label, x, y, name):
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.plot(x, y)
    plt.savefig(name)
    plt.close()


# Run Training
from pathlib import Path

# Model creation
file_primary = Path(primary_model_file_name)
if file_primary.exists():
    print("Found an existing model")
    model = load_model(primary_model_file_name)
else:
    print("Model not found, a new one will be crate")
    model = create_dueling_model(input_shape, actions_number)

# Print a summary about the model
print(model.summary())

# Setting the optimizer
training = True
if training:
    model_target = create_dueling_model(input_shape, actions_number)
    model_target.set_weights(model.get_weights())
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    policy_training = EpsilonGreedyPolicy(model, actions_number, episodes=episodes, min_epsilon=min_epsilon)
    replay_buffer = PrioritizedExperienceReplayRankBased(buffer_size, step_to_heapify, alpha)
    agent = DuelDQNAgent(env_prep, model, model_target, policy_training, optimizer, replay_buffer)
    steps, rewards = agent.double_dqn_training(batch_size, loss_function, discount_factor, freq_replacement,
                                               clipping_value, beta_min, beta_max, episodes)
    env_prep.close()
    model.save(primary_model_file_name)

    ext = "png"
    name_plot_eps_steps = "{} Training Episodes Steps.{}".format(game_name, ext)
    name_plot_eps_rewards = "{} Training Episodes Rewards.{}".format(game_name, ext)
    file_plot_1 = Path(name_plot_eps_steps)
    file_plot_2 = Path(name_plot_eps_rewards)
    i = 1
    while file_plot_1.exists():
        file_plot_1 = Path(name_plot_eps_steps)
        name_plot_eps_steps = "{} Training Episodes Steps_{}.{}".format(game_name, i, ext)
        name_plot_eps_rewards = "{} Training Episodes Rewards_{}.{}".format(game_name, i, ext)
        i += 1

    plot_result("Episode", "Steps", range(1, episodes+1), steps, name_plot_eps_steps)
    plot_result("Episode", "Steps", range(1, episodes+1), rewards, name_plot_eps_rewards)

    import pandas as pd
    csv_name = "{}.csv".format(game_name)
    dict = {'steps': steps, 'rewards': rewards}
    df = pd.DataFrame(dict)
    df.to_csv(csv_name, mode='a', header=False)


play = False
if play:
    policy_play = EpsilonGreedyPolicy(model, actions_number, min_epsilon=min_epsilon)
    agent = DuelDQNAgent(env_prep, model, None, policy_play, None, None)
    steps, reward = agent.play()

    ext = "png"
    name_plot_steps_rewards = "{} Play Steps Rewards.{}".format(game_name, ext)
    file_plot_1 = Path(name_plot_steps_rewards)
    i = 1
    while file_plot_1.exists():
        file_plot_1 = Path(name_plot_steps_rewards)
        name_plot_steps_rewards = "{} Play Steps Rewards_{}.{}".format(game_name, i, ext)
        i += 1
    plot_result("Episode", "Steps", steps, reward, name_plot_steps_rewards)
