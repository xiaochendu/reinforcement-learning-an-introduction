#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

FILE_ROOT = Path(__file__).parent.parent

# set up parameters
HEIGHT = 7
WIDTH = 10

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

WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

START_STATE = (3, 0)
TERMINAL_STATE = (3, 7)

ALPHA = 0.5

EPSILON = 0.1

def step(state, action):
    row, col = state
    row -= WIND[col]
    drow, dcol = ACTIONS[action]
    row += drow
    col += dcol

    # ensure row, col are in bounds
    row = min(max(row, 0), HEIGHT - 1)
    col = min(max(col, 0), WIDTH - 1)

    next_state = (row, col)
    
    if next_state == TERMINAL_STATE:
        reward = 0.0
    else:
        reward = -1.0

    return reward, next_state

def get_action(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(list(ACTIONS.keys()))
    else:
        values_ = q_value[state[0], state[1]]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
    
    return action

def run_sarsa(episode_limit = 500):
    # initialize table
    q_value = np.zeros((HEIGHT, WIDTH, len(ACTIONS)))

    steps = []

    for iter in tqdm(range(episode_limit)):
        # SARSA algorithm
        time = 0
        state = START_STATE
        action = get_action(state, q_value)

        # iterate until end
        while state != TERMINAL_STATE:
            reward, new_state = step(state, action)
            new_action = get_action(new_state, q_value)

            # print("state, action, new state, new action")
            # print(state, action, new_state, new_action)

            q_value[state[0], state[1], action] += \
                ALPHA * (reward + q_value[new_state[0], new_state[1], new_action] - q_value[state[0], state[1], action])
            state = new_state
            action = new_action
            time += 1
        steps.append(time)
    
    return steps, q_value
    

def figure_6_3():
    steps, q_value  = run_sarsa()
    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig(FILE_ROOT / 'images/figure_6_3.png')
    plt.close()

    # display the optimal policy
    optimal_policy = []
    for i in range(0, HEIGHT):
        optimal_policy.append([])
        for j in range(0, WIDTH):
            if [i, j] == TERMINAL_STATE:
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
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))

if __name__ == '__main__':
    figure_6_3()
