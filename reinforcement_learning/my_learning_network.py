import gym
from gym import envs

from agent import DuelDQNAgent
from policy import EpsilonGreedyPolicy
from replay_buffer import PrioritizedExperienceReplayRankBased

# Searching for available environments
game_name = "Asterix"
all_envs = envs.registry.values()
env_ids = [env_spec.id for env_spec in all_envs]

for id in sorted(env_ids):
    if game_name in id:
        print(id)

# Environment Configuration
# from gym import envs
from gym.wrappers import AtariPreprocessing
from gym.wrappers import FrameStack

# Make Parameters:
game_name = "Asterix"
game_mode = "NoFrameskip"  # [Deterministic | NoFrameskip | ram | ramDeterministic | ramNoFrameskip ]
game_version = "v4"  # [v0 | v4 | v5]
env_name = '{}{}-{}'.format(game_name, game_mode, game_version)
env_render_mode = 'human'  # [human | rgb_array]
env_frame_skip = 4

env = gym.make(env_name, render_mode=env_render_mode)

print("Environment observation: ", env.observation_space)
print("Environment action space: ", env.action_space)
print("Action list: ", env.unwrapped.get_action_meanings())

env = AtariPreprocessing(env, frame_skip=env_frame_skip, grayscale_obs=True, terminal_on_life_loss=True,
                         noop_max=30)
env = FrameStack(env, 4)
env.reset()
print("Environment preprocessed observation: ", env.observation_space.shape)
print("Environment preprocessed action space: ", env.action_space)

# Neural Network Creation
from tensorflow import math
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend


def create_dueling_model(input_shape, number_actions):
    backend.set_image_data_format('channels_first')

    inputs = layers.Input(shape=input_shape)

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
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
input_shape = env.observation_space.shape
actions_number = env.action_space.n
# Model persistent file
primary_model_file_name = "{}_dueling_model".format(game_name)

# Training Parameters
loss_function = losses.mean_squared_error
batch_size = 32
discount_factor = 0.95
learning_rate = 6.25e-5
episodes = 2000
clipping_value = 10
training_freq = 4

# Dual DQN Training
freq_replacement = 500

# Replay buffer parameters
buffer_size = 100000
# step_to_heapify = 200
alpha = 0.7
beta_max = 1
beta_min = 0.5

# Policy parameters
min_epsilon = 0.01

# Plot result
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def plot_result(x_label, y_label, x, y, name):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y)
    plt.savefig(name)
    plt.close()


# Run Training
from pathlib import Path
import pandas as pd

# Model creation
file_primary = Path(primary_model_file_name)
if file_primary.exists():
    print("Found an existing model")
    model = load_model(primary_model_file_name)
else:
    print("Model not found, a new one will be create")
    model = create_dueling_model(input_shape, actions_number)

# Print a summary about the model
print(model.summary())


def training():
    try:
        model_target = create_dueling_model(input_shape, actions_number)
        model_target.set_weights(model.get_weights())
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        policy_training = EpsilonGreedyPolicy(model, actions_number, episodes=episodes, min_epsilon=min_epsilon)
        replay_buffer = PrioritizedExperienceReplayRankBased(buffer_size, alpha)
        agent = DuelDQNAgent(env, model, policy_training, model_target, optimizer, replay_buffer)
        steps, rewards = agent.double_dqn_training(batch_size, loss_function, discount_factor, freq_replacement,
                                                   training_freq, clipping_value, beta_min, beta_max, episodes)

        ext = "png"
        name_plot_eps_steps = "{} Training Episodes Steps.{}".format(game_name, ext)
        name_plot_eps_rewards = "{} Training Episodes Rewards.{}".format(game_name, ext)
        file_plot_1 = Path(name_plot_eps_steps)
        i = 1
        while file_plot_1.exists():
            i += 1
            name_plot_eps_steps = "{} Training Episodes Steps_{}.{}".format(game_name, i, ext)
            name_plot_eps_rewards = "{} Training Episodes Rewards_{}.{}".format(game_name, i, ext)
            file_plot_1 = Path(name_plot_eps_steps)

        plot_result("Episode", "Steps", range(1, episodes + 1), steps, name_plot_eps_steps)
        plot_result("Episode", "Rewards", range(1, episodes + 1), rewards, name_plot_eps_rewards)

        csv_name = "{}.csv".format(game_name)
        dict = {'steps': steps, 'rewards': rewards}
        df = pd.DataFrame(dict)
        df.to_csv(csv_name, mode='a', header=False)

    finally:
        model.save(primary_model_file_name)


def play():
    policy_play = EpsilonGreedyPolicy(model, actions_number, min_epsilon=min_epsilon)
    agent = DuelDQNAgent(env, model, policy_play)
    steps, reward = agent.play()


let_training = True
let_play = False

if let_training:
    training()

if let_play:
    play()

env.close()
