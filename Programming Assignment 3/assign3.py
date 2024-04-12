import gym

import numpy as np
import seaborn as sns

from tqdm import tqdm
import io
import sys
import matplotlib.pyplot as plt
import glob
import random
from IPython.display import HTML

# Setting up the Taxi-v3 scenario
env = gym.make("Taxi-v3")
env.reset()

# Displaying the state space information
print("State space details:", env.observation_space)

# Displaying the action space information
print("Action space details:", env.action_space)

# Retrieving details from the environment state
taxi_row, taxi_column, passenger_index, destination_index = list(env.decode(env.s))
print("Current position of the taxi (Row, Column):", (taxi_row, taxi_column))
print("Location of the passenger:", passenger_index)
print("Destination of the passenger:", destination_index)

env.render()

# Randomly choosing an action from a range of 6 actions
random_action = np.random.choice(np.arange(6))
print("Action selected:", random_action)

# Taking a step in the environment based on the chosen action
new_state, earned_reward, is_done, transition_prob = env.step(random_action)
print("Next state after action:", new_state)
print("Reward received:", earned_reward)
print("Is the new state terminal?", is_done)
print("Transition probability:", transition_prob)

env.render()


def navigate_to_R(state, environment):

    # Decode the state to extract row and column details
    agent_row, agent_col, _, _ = list(environment.decode(state))

    # Initialize default values for action and termination check
    is_goal_reached = False
    selected_move = 0  # Default move

    # Check if the agent has reached the goal position
    if (agent_row == 0 and agent_col == 0):
        is_goal_reached = True
        return is_goal_reached, selected_move

    # Move upwards if the agent is in column 0
    if (agent_col == 0):
        selected_move = 1  # Move up
        return is_goal_reached, selected_move

    # Move downwards if at (0, 2) or (1, 2)
    if ((agent_row == 0 and agent_col == 2) or (agent_row == 1 and agent_col == 2)):
        selected_move = 0  # Move down
        return is_goal_reached, selected_move

    # Move upwards if at (3, 1), (3, 3), (4, 1), or (4, 3)
    if ((agent_row == 3 and agent_col == 1) or (agent_row == 3 and agent_col == 3) or
        (agent_row == 4 and agent_col == 1) or (agent_row == 4 and agent_col == 3)):
        selected_move = 1  # Move up
        return is_goal_reached, selected_move

    # Move left otherwise
    selected_move = 3  # Move left
    return is_goal_reached, selected_move

env.reset()
env.render()
_, selected_move = navigate_to_R(env.s, env)
env.step(selected_move)
env.render()


def navigate_to_Y(state, environment):

    # Decode the state to extract row and column information
    agent_row, agent_col, _, _ = list(environment.decode(state))

    # Initialize default values for action and termination condition
    is_goal_reached = False
    selected_move = 0  # Default move

    # Check if the agent has reached the target position
    if (agent_row == 4 and agent_col == 0):
        is_goal_reached = True
        return is_goal_reached, selected_move

    # Move down if the agent is in column 0
    if (agent_col == 0):
        selected_move = 0  # Move down
        return is_goal_reached, selected_move

    # Move down if at (0, 2) or (1, 2)
    if ((agent_row == 0 and agent_col == 2) or (agent_row == 1 and agent_col == 2)):
        selected_move = 0  # Move down
        return is_goal_reached, selected_move

    # Move up if at (3, 1), (3, 3), (4, 1), or (4, 3)
    if ((agent_row == 3 and agent_col == 1) or (agent_row == 3 and agent_col == 3) or
        (agent_row == 4 and agent_col == 1) or (agent_row == 4 and agent_col == 3)):
        selected_move = 1  # Move up
        return is_goal_reached, selected_move

    # Otherwise, move left
    selected_move = 3  # Move left
    return is_goal_reached, selected_move

env.reset()
env.render()
_, selected_move = navigate_to_Y(env.s, env)
env.step(selected_move)
env.render()

def navigate_to_G(state, environment):

    # Decode the state to extract row and column information
    agent_row, agent_col, _, _ = list(environment.decode(state))

    # Initialize default values for action and termination condition
    is_goal_reached = False
    selected_move = 0  # Default move

    # Check if the agent has reached the goal position
    if (agent_row == 0 and agent_col == 4):
        is_goal_reached = True
        return is_goal_reached, selected_move

    # Move up if the agent is in column 4
    if (agent_col == 4):
        selected_move = 1  # Move up
        return is_goal_reached, selected_move

    # Move down if at (0, 1) or (1, 1)
    if ((agent_row == 0 and agent_col == 1) or (agent_row == 1 and agent_col == 1)):
        selected_move = 0  # Move down
        return is_goal_reached, selected_move

    # Move up if at (3, 0), (3, 2), (4, 0), or (4, 2)
    if ((agent_row == 3 and agent_col == 0) or (agent_row == 3 and agent_col == 2) or
        (agent_row == 4 and agent_col == 0) or (agent_row == 4 and agent_col == 2)):
        selected_move = 1  # Move up
        return is_goal_reached, selected_move

    # Otherwise, move right
    selected_move = 2  # Move right
    return is_goal_reached, selected_move

env.reset()
env.render()
_, selected_move = navigate_to_G(env.s, env)
env.step(selected_move)
env.render()


def navigate_to_B(state, environment):

    # Decode the state to extract row and column information
    agent_row, agent_col, _, _ = list(environment.decode(state))

    # Initialize default values for action and termination condition
    is_goal_reached = False
    selected_move = 0  # Default move

    # Check if the agent has reached the destination
    if (agent_row == 4 and agent_col == 3):
        is_goal_reached = True
        return is_goal_reached, selected_move

    # Move left if at (4, 4)
    if (agent_row == 4 and agent_col == 4):
        selected_move = 3  # Move left
        return is_goal_reached, selected_move

    # Move down if in column 3 or 4
    if (agent_col == 3 or agent_col == 4):
        selected_move = 0  # Move down
        return is_goal_reached, selected_move

    # Move down if at (0, 1) or (1, 1)
    if ((agent_row == 0 and agent_col == 1) or (agent_row == 1 and agent_col == 1)):
        selected_move = 0  # Move down
        return is_goal_reached, selected_move

    # Move up if at (3, 0), (3, 2), (4, 0), or (4, 2)
    if ((agent_row == 3 and agent_col == 0) or (agent_row == 3 and agent_col == 2) or
        (agent_row == 4 and agent_col == 0) or (agent_row == 4 and agent_col == 2)):
        selected_move = 1  # Move up
        return is_goal_reached, selected_move

    # Otherwise, move right
    selected_move = 2  # Move right
    return is_goal_reached, selected_move

env.reset()
env.render()
_, selected_move = navigate_to_B(env.s, env)
env.step(selected_move)
env.render()

option_funcs = [navigate_to_R, navigate_to_Y, navigate_to_G, navigate_to_B]

def epsilon_greedy_policy(q_values, current_state, actions_available, epsilon_value):
    state_action_values = q_values[current_state, np.array(actions_available)]
    if ( (np.random.rand() < epsilon_value) or (not state_action_values.any()) ):
        return np.random.choice(actions_available)
    else:
        return actions_available[np.argmax(state_action_values)]

def generate_available_options(state, environment):

    agent_row, agent_col, _, _ = list(environment.decode(state))

    available_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    if (agent_row == 0 and agent_col == 0):
        available_actions.remove(6)
        return available_actions

    if (agent_row == 4 and agent_col == 0):
        available_actions.remove(7)
        return available_actions

    if (agent_row == 0 and agent_col == 4):
        available_actions.remove(8)
        return available_actions

    if (agent_row == 4 and agent_col == 3):
        available_actions.remove(9)
        return available_actions

    return available_actions

class SMDPTrainer:
    """
    Helper class for SMDP Q-Learning training and visualization.
    """

    def __init__(self, gamma_rate=0.9, learn_rate=0.1, exploration_rate=0.1, option_functions=None, available_options_fn=None):
        self.gamma_rate = gamma_rate
        self.learn_rate = learn_rate
        self.exploration_rate = exploration_rate
        self.exp_name = f'gamma_{int(self.gamma_rate*100)}_learn_{int(self.learn_rate*1000)}_exploration_{int(self.exploration_rate*1000)}'
        self.q_values = np.zeros((500, 10))
        self.update_freq = np.zeros((500, 10))
        self.option_functions = option_functions
        self.available_options_fn = available_options_fn
        self.env = gym.make("Taxi-v3")

    def train(self, num_episodes=3000, verbose=True):
        self.num_episodes = num_episodes
        self.episode_rewards = np.zeros(num_episodes)
        self.verbose = verbose

        for episode in (tqdm(range(num_episodes)) if verbose else range(num_episodes)):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                available_actions = self.available_options_fn(state, self.env)
                action = self.egreedy_action(self.q_values, state, available_actions)

                if action < 6:
                    next_state, reward, done, _ = self.env.step(action)
                    self.q_values[state, action] += self.learn_rate * (
                        reward + self.gamma_rate * np.max(self.q_values[next_state, :]) - self.q_values[state, action]
                    )
                    self.update_freq[state, action] += 1
                    state = next_state
                    episode_reward += reward

                if action >= 6:
                    reward_accumulator = 0
                    option_done = False
                    starting_state = state
                    time_steps = 0
                    opt_fn = self.option_functions[action-6]

                    while not option_done:
                        option_done, option_action = opt_fn(state, self.env)

                        if option_done:
                            self.q_values[starting_state, action] += self.learn_rate * (
                                reward_accumulator + (self.gamma_rate**time_steps) * np.max(self.q_values[state, :]) - self.q_values[starting_state, action]
                            )
                            self.update_freq[starting_state, action] += 1
                            break

                        next_state, reward, done, _ = self.env.step(option_action)
                        time_steps += 1
                        reward_accumulator += (self.gamma_rate**(time_steps - 1)) * reward
                        episode_reward += reward
                        state = next_state

            self.episode_rewards[episode] = episode_reward

        return self.episode_rewards, self.q_values, self.update_freq

    def plot_rewards(self, save=False):
        avg_100_reward = np.array([np.mean(self.episode_rewards[max(0, i-100):i]) for i in range(1, len(self.episode_rewards)+1)])

        plt.xlabel('Episode')
        plt.ylabel('Total Episode Reward')
        plt.title('Rewards vs Episodes: Avg Reward: %.3f' % np.mean(self.episode_rewards))
        plt.plot(np.arange(self.num_episodes), self.episode_rewards, 'b')
        plt.plot(np.arange(self.num_episodes), avg_100_reward, 'r', linewidth=1.5)
        if save:
            plt.savefig(f'./smdp/{self.exp_name}_rewards.jpg', pad_inches=0)
        plt.show()

    def plot_updates(self, save=False):
        total_updates = np.sum(self.update_freq, axis=1)
        grid_updates = np.zeros((5, 5))

        for state in range(500):
            row, col, _, _ = self.env.decode(state)
            grid_updates[row, col] += total_updates[state]

        sns.heatmap(grid_updates, annot=True, fmt='g', square=True, cmap='viridis')
        plt.title('Update Frequency Table for SMDP Q-Learning')
        if save:
            plt.savefig(f'./smdp/{self.exp_name}_updates.jpg', pad_inches=0)
        plt.show()

    def plot_q_values(self, save=False):
        q_values_pickup = np.zeros((4, 5, 5, 10))
        q_values_drop = np.zeros((4, 5, 5, 10))

        for state in range(500):
            row, col, src, dest = self.env.decode(state)
            if src < 4 and src != dest:
                q_values_pickup[src][row][col] += self.q_values[state]
            if src == 4:
                q_values_drop[dest][row][col] += self.q_values[state]

        for phase, q_values in zip(['Pick', 'Drop'], [q_values_pickup, q_values_drop]):
            for pos in ['R', 'G', 'Y', 'B']:
                sns.heatmap(q_values[pos], annot=True, square=True, cbar=False,
                            cbar_kws={'ticks': range(10)}, vmin=0, vmax=9, cmap='viridis')
                plt.title(f'Q-Values for SMDP Q-Learning: {phase} at {pos}')
                if save:
                    plt.savefig(f'./smdp/{self.exp_name}_q_vals_{phase}_{pos}.jpg', pad_inches=0)
                plt.show()

def test_hyperparameters(file_path):
    with open(file_path, 'w') as log_file:
        sys.stdout = log_file

        alphas = [0.5, 0.1, 0.05, 0.01]
        gammas = [0.90]
        epsilons = [0.1, 0.05, 0.01, 0.005, 0.001]

        best_reward = -np.inf
        best_params = {'alpha': None, 'gamma': None, 'epsilon': None}

        config_count = 1
        for alpha in alphas:
            for gamma in gammas:
                for epsilon in epsilons:
                    print("Testing Configuration:", config_count)
                    print('Hyperparameters: [alpha = {}, gamma = {}, epsilon = {}]'.format(alpha, gamma, epsilon))

                    agent = SMDPTrainer(alpha=alpha, epsilon=epsilon, gamma_rate=gamma)
                    rewards, _, _ = agent.train(verbose=False)
                    avg_reward = np.mean(rewards)
                    print('Average Reward:', avg_reward)
                    print('***************************************************************************\n')

                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        best_params['alpha'] = alpha
                        best_params['gamma'] = gamma
                        best_params['epsilon'] = epsilon

                    config_count += 1

        print('\nBest Reward:', best_reward)
        print('Best Hyperparameters:', best_params)

test_hyperparameters('./logger/smdp_log1.txt')

# Set the style for plotting
plt.style.use('dark_background')

# Initialize an empty list to store rewards
all_rewards = []

# Run training for 10 iterations
for _ in range(10):
    agent = SMDPTrainer(alpha_value=0.5, epsilon_value=0.1, gamma_rate=0.9)
    episode_rewards, _, _ = agent.train(verbose=True)
    all_rewards.append(episode_rewards)

# Convert the list to a NumPy array
all_rewards = np.array(all_rewards)

# Calculate the average rewards across all iterations
average_rewards = np.mean(all_rewards, axis=0)

# Calculate the average reward over a rolling window of 100 episodes
rolling_avg_rewards = np.array([np.mean(average_rewards[max(0, i-100):i]) for i in range(1, len(average_rewards) + 1)])

# Plotting the rewards
plt.xlabel('Episode')
plt.ylabel('Total Episode Reward')
plt.title(f'Rewards vs Episodes: Avg Reward: {np.mean(average_rewards):.3f}')
plt.plot(np.arange(3000), average_rewards, 'b')
plt.plot(np.arange(3000), rolling_avg_rewards, 'r', linewidth=1.5)
plt.savefig('./smdp/rewards.jpg', pad_inches=0)
plt.show()

agent_zero = SMDPTrainer(alpha_value=0.5, epsilon_value=0.1, gamma_rate=0.9)

# Train the agent and retrieve rewards, Q-values, and update frequencies
rewards_zero, q_values_zero, update_frequencies_zero = agent_zero.train(verbose=True)

# Plot the reward curve and save the plot
agent_zero.plot_rewards(save=True)

# Plot the update frequency and save the plot
agent_zero.plot_update_frequency(save=True)

# Plot the Q-values and save the plot
agent_zero.plot_q_values(save=True)

class IntraOption:
    """
    Custom Intra Option
    """
    def __init__(self, gamma_val=0.9, alpha_val=0.1, epsilon_val=0.1, opt_functions=option_funcs, gen_available_options=gen_available_options):
        self.gamma = gamma_val
        self.alpha = alpha_val
        self.epsilon = epsilon_val
        self.experiment_name = 'a' + str(int(self.alpha * 1000)) + '_e' + str(int(self.epsilon * 1000)) + '_g' + str(int(self.gamma * 100))
        self.q_values = np.zeros((500, 10))
        self.update_frequency = np.zeros((500, 10))
        self.gen_available_options = gen_available_options
        self.opt_functions = opt_functions

        self.environment = gym.make("Taxi-v3")

    def training(self, num_episodes=3000, is_verbose=True):

        self.num_episodes = num_episodes
        self.episode_rewards = np.zeros(num_episodes)
        self.is_verbose = is_verbose

        for episode in (tqdm(range(num_episodes)) if is_verbose else range(num_episodes)):

            state = self.environment.reset()
            done = False
            episode_reward = 0

            while not done:

                available_actions = self.gen_available_options(state, self.environment)

                action = epsilon_greedy_policy(self.q_values, state, available_actions, self.epsilon)

                if action < 6:
                    next_state, reward, done, _ = self.environment.step(action)
                    self.q_values[state, action] += self.alpha * (reward + self.gamma * np.max(self.q_values[next_state, :]) - self.q_values[state, action])
                    self.update_frequency[state, action] += 1

                    for j in range(4):
                        opt_fn = self.opt_functions[j]
                        opt_id = j + 6
                        opt_done, opt_action = opt_fn(state, self.environment)

                        if opt_action == action:
                            self.q_values[state, opt_id] += self.alpha * (reward + self.gamma * np.max(self.q_values[next_state, :]) - self.q_values[state, opt_id])
                            self.update_frequency[state, opt_id] += 1

                    state = next_state
                    episode_reward += reward

                elif action >= 6:
                    opt_done = False
                    start_state = state

                    opt_fn = self.opt_functions[action - 6]
                    time_steps = 0

                    while not opt_done:
                        opt_done, opt_action = opt_fn(state, self.environment)
                        next_state, reward, done, _ = self.environment.step(opt_action)
                        start_state = state
                        state = next_state

                        time_steps += 1
                        episode_reward += reward * (self.gamma ** (time_steps - 1))

                        if opt_done:
                            self.q_values[start_state, action] += self.alpha * (reward + (self.gamma) * np.max(self.q_values[state, :]) - self.q_values[start_state, action])
                            self.update_frequency[start_state, action] += 1

                            self.q_values[start_state, opt_action] += self.alpha * (reward + (self.gamma) * np.max(self.q_values[state, :]) - self.q_values[start_state, opt_action])
                            self.update_frequency[start_state, opt_action] += 1

                            for j in range(4):
                                opt2_fn = self.opt_functions[j]
                                opt2_id = j + 6
                                opt2_done, opt2_action = opt_fn(state, self.environment)

                                if opt_action == opt2_action:
                                    self.q_values[start_state, opt2_id] += self.alpha * (reward + self.gamma * np.max(self.q_values[state, :]) - self.q_values[start_state, opt2_id])
                                    self.update_frequency[state, opt2_id] += 1

                        self.q_values[start_state, action] += self.alpha * (reward + (self.gamma) * (self.q_values[state, action]) - self.q_values[start_state, action])
                        self.update_frequency[start_state, action] += 1

                        self.q_values[start_state, opt_action] += self.alpha * (reward + (self.gamma) * (self.q_values[state, opt_action]) - self.q_values[start_state, opt_action])
                        self.update_frequency[start_state, opt_action] += 1

                        for j in range(4):
                            opt2_fn = self.opt_functions[j]
                            opt2_id = j + 6
                            opt2_done, opt2_action = opt_fn(state, self.environment)

                            if opt_action == opt2_action:
                                self.q_values[start_state, opt2_id] += self.alpha * (reward + self.gamma * (self.q_values[state, opt2_id]) - self.q_values[start_state, opt2_id])
                                self.update_frequency[state, opt2_id] += 1

            self.episode_rewards[episode] = episode_reward
        return self.episode_rewards, self.q_values, self.update_frequency

    def plot_rewards(self, save=False):
        sns.set_style("darkgrid")
        avg_100_reward = np.array([np.mean(self.episode_rewards[max(0, i - 100):i]) for i in range(1, len(self.episode_rewards) + 1)])

        plt.xlabel('Episode')
        plt.ylabel('Total Episode Reward')
        plt.title('Rewards vs Episodes: Avg Reward: %.3f' % np.mean(self.episode_rewards))
        plt.plot(np.arange(self.num_episodes), self.episode_rewards, 'b')
        plt.plot(np.arange(self.num_episodes), avg_100_reward, 'r', linewidth=1.5)
        if save:
            plt.savefig('./intraop/' + self.experiment_name + '_rewards.jpg', pad_inches=0)
        plt.show()

    def plot_update_frequency(self, save=False):
        total_updates = np.sum(self.update_frequency, axis=1)
        grid_updates = np.zeros((5, 5))

        for state in range(500):
            row, col, src, dst = self.environment.decode(state)
            grid_updates[row, col] += total_updates[state]

        sns.heatmap(grid_updates, annot=True, fmt='g', square=True, cmap='viridis')
        plt.title('Update Frequency Table for Intra-Option Q-Learning')
        if save:
            plt.savefig('./intraop/' + self.experiment_name + '_updates.jpg', pad_inches=0)
        plt.show()

    def plot_q_values(self, save=False):
        pickup_q_values = np.zeros((4, 5, 5, 10))
        q_values = np.zeros((2, 4, 5, 5))
        for state in range(500):
            row, col, src, dest = self.environment.decode(state)
            if src < 4 and src != dest:
                pickup_q_values[src][row][col] += self.q_values[state]

        for state in range(500):
            row, col, src, dest = self.environment.decode(state)
            if src < 4 and src != dest:
                q_values[0][src][row][col] = np.argmax(pickup_q_values[src][row][col])
            if src == 4:
                q_values[1][dest][row][col] = np.argmax(self.q_values[state])

        phase = ['Pick', 'Drop']
        positions = ['R', 'G', 'Y', 'B']
        for i in range(2):
            for j in range(4):
                sns.heatmap(q_values[i][j], annot=True, square=True, cbar=False, cbar_kws={'ticks': range(10)}, vmin=0, vmax=9, cmap='viridis')
                plt.title('Q-Values for Intra-Option Q-Learning: {} at {}'.format(phase[i], positions[j]))
                if save:
                    plt.savefig('./intraop/' + self.experiment_name + '_q_vals_' + phase[i] + '_' + positions[j] + '.jpg', pad_inches=0)
                plt.show()

def test_hyperparameters(alphas, gammas, epsilons):
    best_reward = -np.inf
    best_hyperparams = {'alpha': None, 'gamma': None, 'epsilon': None}

    config_count = 1

    with open('./logs/intraop_log1.txt', 'w') as f:
        sys.stdout = f  # Redirect stdout to the log file

        for alpha in alphas:
            for gamma in gammas:
                for epsilon in epsilons:
                    print("Testing Configuration:", config_count)
                    print('Hyperparameters: [alpha = {}, gamma = {}, epsilon = {}]'.format(alpha, gamma, epsilon))

                    agent = IntraOption(alpha=alpha, epsilon=epsilon, gamma=gamma)

                    rewards, q_values, update_freq = agent.training(verbose=False)
                    avg_reward = np.mean(rewards)

                    print('Average Reward:', avg_reward)
                    print('***************************************************************************\n')

                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        best_hyperparams['alpha'] = alpha
                        best_hyperparams['gamma'] = gamma
                        best_hyperparams['epsilon'] = epsilon

                    config_count += 1

        print('\nBest Reward:', best_reward)
        print('Best Hyperparameters:', best_hyperparams)

# Define hyperparameters
alphas = [0.5, 0.1, 0.05, 0.01]
gammas = [0.90]
epsilons = [0.1, 0.05, 0.01, 0.005, 0.001]

# Test hyperparameters
test_hyperparameters(alphas, gammas, epsilons)

plt.style.use('dark_background')

def test_rewards():
    reward_list = []

    for _ in range(10):
        agent = IntraOption(alpha=0.5, epsilon=0.001, gamma=0.9)
        rewards, _, _= agent.trainer(verbose=True)
        reward_list.append(rewards)

    reward_list = np.array(reward_list)
    mean_rewards = np.mean(reward_list, axis=0)
    avg_100_rewards = np.array([np.mean(mean_rewards[max(0,i-100):i]) for i in range(1, len(mean_rewards)+1)])

    plt.xlabel('Episode')
    plt.ylabel('Total Episode Reward')
    plt.title('Rewards vs Episodes: Avg Reward: %.3f' % np.mean(mean_rewards))
    plt.plot(np.arange(3000), mean_rewards, 'b')
    plt.plot(np.arange(3000), avg_100_rewards, 'r', linewidth=1.5)
    plt.savefig('./intraop/rewards.jpg', pad_inches=0)
    plt.show()

# Call the function to test rewards
test_rewards()

agent_one = IntraOption(alpha_value=0.5, epsilon_value=0.001, gamma_rate=0.9)

# Train the agent and retrieve rewards, Q-values, and update frequencies
rewards_one, q_values_one, update_frequencies_one = agent_one.train(verbose=True)

# Plot the reward curve and save the plot
agent_one.plot_rewards(save=True)

# Plot the update frequency and save the plot
agent_one.plot_update_frequency(save=True)

# Plot the Q-values and save the plot
agent_one.plot_q_values(save=True)


plt.style.use('darkgrid')

avg10_reward_smdp = np.array([np.mean(rewards_zero[max(0, i-10):i]) for i in range(1, len(rewards_zero)+1)])
avg10_reward_intraop = np.array([np.mean(rewards_one[max(0, i-10):i]) for i in range(1, len(rewards_one)+1)])

plt.xlabel('Episode')
plt.ylabel('Total Episode Reward')
plt.title('Rewards vs Episodes: SMDP & IntraOp')
plt.plot(np.arange(len(rewards_zero)), avg10_reward_smdp)
plt.plot(np.arange(len(rewards_one)), avg10_reward_intraop)
plt.legend(['SMDP', 'IntraOp'])
plt.savefig('./gen_imgs/smdp_vs_intraop_rewards.jpg', pad_inches=0)
plt.show()

sns.set_style("darkgrid")

# Calculate total updates for SMDP and IntraOp
total_updates_smdp = np.sum(update_frequencies_zero, axis=1)
total_updates_intraop = np.sum(update_frequencies_one, axis=1)

# Initialize grid updates matrices
grid_updates_smdp = np.zeros((5, 5))
grid_updates_intraop = np.zeros((5, 5))

# Update grid updates matrices
for state in range(500):
    row, col, src, dst = env.decode(state)
    grid_updates_smdp[row, col] += total_updates_smdp[state]
    grid_updates_intraop[row, col] += total_updates_intraop[state]

# Determine color scale limits
vmin = min(np.min(grid_updates_smdp), np.min(grid_updates_intraop))
vmax = max(np.max(grid_updates_smdp), np.max(grid_updates_intraop))

# Create subplots for SMDP and IntraOp update frequency comparison
fig, axs = plt.subplots(1, 2)
fig.suptitle('Update Frequency Table: SMDP vs IntraOp')

# Plot heatmaps for SMDP and IntraOp
sns.heatmap(grid_updates_smdp, annot=True, fmt='g', square=True, cmap='viridis', cbar=False, ax=axs[0], vmin=vmin, vmax=vmax)
sns.heatmap(grid_updates_intraop, annot=True, fmt='g', square=True, cmap='viridis', cbar=False, ax=axs[1], vmin=vmin, vmax=vmax)

# Set figure size and save the plot
fig.set_figwidth(10)
fig.set_figheight(4.5)
plt.savefig('./gen_imgs/smdp_vs_intraop_updates.jpg', pad_inches=0)

plt.show()


def move_south(state, environment):
    # Decode the state information
    current_row, current_col, _, _ = list(environment.decode(state))

    # Initialize default values for action and termination condition
    is_done = False
    selected_action = 0

    # Check if the target row (South direction) is reached
    if current_row == 4:
        is_done = True
        return is_done, selected_action

    # Move in the South direction
    selected_action = 0  # Represents the action for moving South
    return is_done, selected_action

def move_north(state, environment):
    # Decode the state information
    current_row, current_col, _, _ = list(environment.decode(state))

    # Initialize default values for action and termination condition
    is_done = False
    selected_action = 0

    # Check if the target row (North direction) is reached
    if current_row == 0:
        is_done = True
        return is_done, selected_action

    # Move in the North direction
    selected_action = 1  # Represents the action for moving North
    return is_done, selected_action

def move_east(state, environment):

    # Decode the state information
    current_row, current_col, _, _ = list(environment.decode(state))

    # Initialize default values for action and termination condition
    is_done = False
    selected_action = 0

    # Check if the target (East direction) is reached
    if (
        current_col == 4
        or (
            (current_row == 0 and current_col == 1)
            or (current_row == 1 and current_col == 1)
            or (current_row == 3 and current_col == 0)
            or (current_row == 3 and current_col == 2)
            or (current_row == 4 and current_col == 0)
            or (current_row == 4 and current_col == 2)
        )
    ):
        is_done = True
        return is_done, selected_action

    # Move in the East direction
    selected_action = 2  # Represents the action for moving East
    return is_done, selected_action

def move_west(state, environment):

    # Decode the state information
    current_row, current_col, _, _ = list(environment.decode(state))

    # Initialize default values for action and termination condition
    is_done = False
    selected_action = 0

    # Check if the target (West direction) is reached
    if (
        current_col == 0
        or (
            (current_row == 0 and current_col == 2)
            or (current_row == 1 and current_col == 2)
            or (current_row == 3 and current_col == 1)
            or (current_row == 3 and current_col == 3)
            or (current_row == 4 and current_col == 1)
            or (current_row == 4 and current_col == 3)
        )
    ):
        is_done = True
        return is_done, selected_action

    # Move in the West direction
    selected_action = 3  # Represents the action for moving West
    return is_done, selected_action

def generate_available_options(state, environment):

    # Decode the state information
    current_row, current_col, _, _ = list(environment.decode(state))

    # Initialize the list of available actions
    available_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Check for conditions and remove unavailable actions accordingly
    if current_row == 4:
        available_actions.pop(6)
        return available_actions

    if current_row == 0:
        available_actions.pop(7)
        return available_actions

    if (
        (current_col == 4)
        or ((current_row == 0 and current_col == 1) or (current_row == 1 and current_col == 1))
        or ((current_row == 3 and current_col == 0) or (current_row == 3 and current_col == 2) or (current_row == 4 and current_col == 0) or (current_row == 4 and current_col == 2))
    ):
        available_actions.pop(8)
        return available_actions

    if (
        (current_col == 0)
        or ((current_row == 0 and current_col == 2) or (current_row == 1 and current_col == 2))
        or ((current_row == 3 and current_col == 1) or (current_row == 3 and current_col == 3) or (current_row == 4 and current_col == 1) or (current_row == 4 and current_col == 3))
    ):
        available_actions.pop(9)
        return available_actions

    return available_actions

options_funcs_new = [move_south, move_north, move_east, move_west]

def test_hyperparameters(logfile_path='./logs/intraop_log2.txt'):
    with open(logfile_path, 'w') as logfile:
        sys.stdout = logfile

        learning_rates = [0.5, 0.1, 0.05, 0.01]
        discount_factors = [0.90]
        exploration_rates = [0.1, 0.01, 0.001]

        best_reward = -float('inf')
        best_hyperparameters = {'alpha': None, 'gamma': None, 'epsilon': None}

        config_number = 1
        for alpha in learning_rates:
            for gamma in discount_factors:
                for epsilon in exploration_rates:
                    print("Testing Configuration:", config_number)
                    print(f'Hyperparameters: [alpha = {alpha}, gamma = {gamma}, epsilon = {epsilon}]')

                    agent = IntraOption(alpha=alpha, epsilon=epsilon, gamma=gamma, opt_fns=options_funcs_new, gen_avl_options=generate_available_options)
                    rewards, _, _ = agent.trainer(verbose=False)
                    avg_reward = np.mean(rewards)
                    print('Average Reward:', avg_reward)
                    print('***************************************************************************\n')

                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        best_hyperparameters['alpha'] = alpha
                        best_hyperparameters['gamma'] = gamma
                        best_hyperparameters['epsilon'] = epsilon

                    config_number += 1

        print('\nBest Reward:', best_reward)
        print('Best Hyperparameters:', best_hyperparameters)

# Call the function to test hyperparameters
test_hyperparameters()

# Set the plot style
sns.set_style("darkgrid")

# Initialize an empty list to store rewards
reward_list = []

# Run the training loop 10 times
for _ in range(10):
    # Create an IntraOption agent with specified hyperparameters
    agent = IntraOption(alpha=0.5, epsilon=0.001, gamma=0.9, opt_fns=options_funcs_new, gen_avl_options=generate_available_options)

    # Train the agent and collect rewards
    rewards, _, _ = agent.trainer(verbose=True)
    reward_list.append(rewards)

# Convert the reward list to a numpy array
reward_array = np.array(reward_list)

# Calculate the mean rewards across episodes
mean_rewards = np.mean(reward_array, axis=0)

# Calculate the rolling average over 100 episodes
rolling_avg_rewards = np.array([np.mean(mean_rewards[max(0, i - 100):i]) for i in range(1, len(mean_rewards) + 1)])

# Plotting
plt.xlabel('Episode')
plt.ylabel('Total Episode Reward')
plt.title('Rewards vs Episodes: Avg Reward: {:.3f}'.format(np.mean(mean_rewards)))
plt.plot(np.arange(3000), mean_rewards, 'b')
plt.plot(np.arange(3000), rolling_avg_rewards, 'r', linewidth=1.5)
plt.savefig('./intraop/rewards_new.jpg', pad_inches=0)
plt.show()

agent_11 = IntraOption(alpha_value=0.5, epsilon_value=0.001, gamma_rate=0.9, opt_fns=options_funcs_new, gen_avl_options=generate_available_options)

# Train the agent and retrieve rewards, Q-values, and update frequencies
rewards_11, q_values_11, update_frequencies_11 = agent_11.train(verbose=True)

# Plot the reward curve and save the plot
agent_11.plot_rewards(save=True)

# Plot the update frequency and save the plot
agent_11.plot_update_frequency(save=True)

# Plot the Q-values and save the plot
agent_11.plot_q_values(save=True)

sns.set_style("darkgrid")

# Calculate the rolling average over 10 episodes for old and new options
avg_10_reward1 = np.array([np.mean(rewards_one[max(0, i - 10):i]) for i in range(1, len(rewards_one) + 1)])
avg_10_reward11 = np.array([np.mean(rewards_11[max(0, i - 10):i]) for i in range(1, len(rewards_11) + 1)])

# Plotting
plt.xlabel('Episode')
plt.ylabel('Total Episode Reward')
plt.title('Rewards for IntraOp - Old vs New Options')
plt.plot(np.arange(len(rewards_one)), avg_10_reward1)
plt.plot(np.arange(len(rewards_11)), avg_10_reward11)
plt.legend(['Old Options', 'New Options'])
plt.savefig('./intraop/old_v_new_rewards.jpg', pad_inches=0)
plt.show()


import sys

def test_hyperparameters():
    with open('./logs/smdp_log2.txt', 'w') as f:
        sys.stdout = f
        alphas = [0.5, 0.1, 0.05, 0.01]
        gammas = [0.90]
        epsilons = [0.1]

        best_reward = -float('inf')
        best_hyperparams = {'alpha': None, 'gamma': None, 'epsilon': None}

        config = 1
        for alpha in alphas:
            for gamma in gammas:
                for epsilon in epsilons:
                    print("Testing Configuration:", config)
                    print('Hyperparameters: [alpha = {}, gamma = {}, epsilon = {}]'.format(alpha, gamma, epsilon))
                    agent = SMDPTrainer(alpha=alpha, epsilon=epsilon, gamma=gamma, opt_fns=options_funcs_new, gen_avl_options=generate_available_options)
                    rewards, q_values, update_freq = agent.trainer(verbose=False)
                    avg_reward = np.mean(rewards)
                    print('Average Reward:', avg_reward)
                    print('***************************************************************************\n')

                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        best_hyperparams['alpha'] = alpha
                        best_hyperparams['gamma'] = gamma
                        best_hyperparams['epsilon'] = epsilon

                    config += 1

        print('\nBest Reward:', best_reward)
        print('Best Hyperparameters:', best_hyperparams)

# Call the function to execute the hyperparameter testing
test_hyperparameters()

alpha = 0.5
epsilon = 0.1
gamma = 0.9

reward_list = []

# Training the SMDP agent for 5 episodes
for i in range(5):
    agent = SMDPTrainer(alpha=alpha, epsilon=epsilon, gamma=gamma, opt_fns=options_funcs_new, gen_avl_options=generate_available_options)
    rewards, _, _ = agent.trainer(verbose=True)
    reward_list.append(rewards)

# Calculating average rewards and the moving average
reward_array = np.array(reward_list)
eps_rewards = np.mean(reward_array, axis=0)
avg100_reward = np.array([np.mean(eps_rewards[max(0, i - 100):i]) for i in range(1, len(eps_rewards) + 1)])

# Plotting the rewards vs episodes
plt.style.use("darkgrid")
plt.xlabel('Episode')
plt.ylabel('Total Episode Reward')
plt.title('Rewards vs Episodes: Avg Reward: %.3f' % np.mean(eps_rewards))
plt.plot(np.arange(3000), eps_rewards, 'b')
plt.plot(np.arange(3000), avg100_reward, 'r', linewidth=1.5)
plt.savefig('./smdp/rewards_new.jpg', pad_inches=0)
plt.show()

agent_01 = SMDPTrainer(alpha_value=0.5, epsilon_value=0.1, gamma_rate=0.9, opt_fns=options_funcs_new, gen_avl_options=generate_available_options)

# Train the agent and retrieve rewards, Q-values, and update frequencies
rewards_01, q_values_01, update_frequencies_01 = agent_01.train(verbose=True)

# Plot the reward curve and save the plot
agent_01.plot_rewards(save=True)

# Plot the update frequency and save the plot
agent_01.plot_update_frequency(save=True)

# Plot the Q-values and save the plot
agent_01.plot_q_values(save=True)

# Set the plotting style
plt.style.use("darkgrid")

# Calculate moving averages for old and new options
avg10_reward_old = np.array([np.mean(rewards_zero[max(0, i - 10):i]) for i in range(1, len(rewards_zero) + 1)])
avg10_reward_new = np.array([np.mean(rewards_01[max(0, i - 10):i]) for i in range(1, len(rewards_01) + 1)])

# Plotting the rewards vs episodes for old and new options
plt.xlabel('Episode')
plt.ylabel('Total Episode Reward')
plt.title('Rewards for IntraOp - Old vs New Options')
plt.plot(np.arange(len(rewards_zero)), avg10_reward_old)
plt.plot(np.arange(len(rewards_01)), avg10_reward_new)
plt.legend(['Old Options', 'New Options'])
plt.savefig('./smdp/old_v_new_rewards.jpg', pad_inches=0)
plt.show()