#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# %%
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from pathlib import Path

FILE_ROOT = Path(__file__).parent.parent

# set up parameters
HEIGHT = 4
WIDTH = 12

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1)
}

CLIFF = list(itertools.product([3], range(1, 12)))

# %%
START_STATE = (3, 0)
TERMINAL_STATE = (3, 11)

ALPHA = 0.5

EPSILON = 0.1

GAMMA = 1
# TODO: asymptotically reduce EPSILON

def step(state, action):
    row, col = state
    drow, dcol = ACTIONS[action]
    row += drow
    col += dcol

    # ensure row, col are in bounds
    row = min(max(row, 0), HEIGHT - 1)
    col = min(max(col, 0), WIDTH - 1)

    next_state = (row, col)
    
    if next_state == TERMINAL_STATE:
        reward = 0.0
    elif next_state in CLIFF:
        reward = -100.0
        next_state = START_STATE
    else:
        reward = -1.0

    return reward, next_state

def get_action_epsilon(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(list(ACTIONS.keys()))
    else:
        values_ = q_value[state[0], state[1]]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
    
    return action

# def get_action_greedy(state, q_value):
#     values_ = q_value[state[0], state[1]]
#     action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
    
#     return action

def run_sarsa(step_size=ALPHA, episode_limit=500, algorithm='sarsa'):
    # initialize table
    q_value = np.zeros((HEIGHT, WIDTH, len(ACTIONS)))
    rewards = []
    for iter in range(episode_limit):
        cum_reward = 0
        state = START_STATE
        action = get_action_epsilon(state, q_value)

        # iterate until end
        while state != TERMINAL_STATE:
            reward, new_state = step(state, action)
            new_action = get_action_epsilon(new_state, q_value)

            # print("state, action, new state, new action")
            # print(state, action, new_state, new_action)

            if algorithm == 'expected_sarsa':
                # calculate the expected value of new state
                # TODO: expected SARSA might be incorrect; should implement with Q learning
                target = 0.0
                q_next = q_value[new_state[0], new_state[1], :]
                best_actions = np.argwhere(q_next == np.max(q_next))
                for action_ in ACTIONS:
                    if action_ in best_actions:
                        target += ((1.0 - EPSILON) / len(best_actions) + EPSILON / len(ACTIONS)) * q_value[new_state[0], new_state[1], action_]
                    else:
                        target += EPSILON / len(ACTIONS) * q_value[new_state[0], new_state[1], action_]
            else:
                target = q_value[new_state[0], new_state[1], new_action]
            
            q_value[state[0], state[1], action] += \
                step_size * (reward + GAMMA * target - q_value[state[0], state[1], action])
            state = new_state
            action = new_action
            cum_reward += reward
        rewards.append(cum_reward)
    
    return np.array(rewards)

def run_q_learning(step_size=ALPHA, episode_limit = 500):
    q_value = np.zeros((HEIGHT, WIDTH, len(ACTIONS)))
    rewards = []
    for iter in range(episode_limit):
        cum_reward = 0
        state = START_STATE

        # iterate until end
        while state != TERMINAL_STATE:
            action = get_action_epsilon(state, q_value)
            reward, new_state = step(state, action)
            # print("state, action, new state, new action")
            # print(state, action, new_state, new_action)
            q_value[state[0], state[1], action] += \
                step_size * (reward + GAMMA * np.max(q_value[new_state[0], new_state[1], :]) - q_value[state[0], state[1], action])
            state = new_state
            cum_reward += reward
        rewards.append(cum_reward)
    
    return np.array(rewards)

# print optimal policy
def print_optimal_policy(q_value):
    optimal_policy = []
    for i in range(0, HEIGHT):
        optimal_policy.append([])
        for j in range(0, WIDTH):
            if (i, j) == TERMINAL_STATE:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == UP:
                optimal_policy[-1].append('U')
            elif bestAction == DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == RIGHT:
                optimal_policy[-1].append('R')
    for row in optimal_policy:
        print(row)
    
def figure_6_4():
    episodes = 500
    runs = 50
    sarsa_rewards = np.zeros(episodes)
    q_learning_rewards = np.zeros(episodes)
    
    for r in tqdm(range(runs)):
        sarsa_rewards += np.array(run_sarsa(algorithm='sarsa'))
        q_learning_rewards += np.array(run_q_learning())

    sarsa_rewards /= runs
    q_learning_rewards /= runs

    # print("sarsa", sarsa_rewards)
    # print("q_learning_rewards", q_learning_rewards)

    plt.plot(sarsa_rewards, label="SARSA")
    plt.plot(q_learning_rewards, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    # plt.savefig(FILE_ROOT / 'images/figure_6_4.png')
    plt.show()
    plt.close()


def figure_6_5():
    alphas = np.arange(0.1, 1.1, 0.1)
    episodes = 100
    runs = 5

    ASY_SARSA = 0
    ASY_EXPECTED_SARSA = 1
    ASY_QLEARNING = 2
    INT_SARSA = 3
    INT_EXPECTED_SARSA = 4
    INT_QLEARNING = 5

    performance = np.zeros((6, len(alphas)))

    for run in range(runs):
        for ind, step_size in tqdm(enumerate(alphas)):
            sarsa_rewards = run_sarsa(step_size=step_size, algorithm='sarsa', episode_limit=episodes)
            expected_sarsa_rewards = run_sarsa(step_size=step_size, algorithm='expected_sarsa', episode_limit=episodes)
            q_learning_rewards = run_q_learning(step_size=step_size, episode_limit=episodes)
            performance[INT_SARSA, ind] += sum(sarsa_rewards)
            performance[INT_EXPECTED_SARSA, ind] += sum(expected_sarsa_rewards)
            performance[INT_QLEARNING, ind] += sum(q_learning_rewards)

    performance[:3, :] /= episodes * runs
    performance[3:, :] /= episodes * runs
    labels = ['Asymptotic Sarsa', 'Asymptotic Expected Sarsa', 'Asymptotic Q-Learning',
              'Interim Sarsa', 'Interim Expected Sarsa', 'Interim Q-Learning']

    for method, label in enumerate(labels):
        plt.plot(alphas, performance[method, :], label=label)
    plt.xlabel('alpha')
    plt.ylabel('reward per episode')
    plt.legend()

    # plt.savefig('../images/figure_6_6.png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    # figure_6_4()
    figure_6_5()
