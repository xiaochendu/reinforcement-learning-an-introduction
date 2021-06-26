#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size

matplotlib.use('Agg')

FILE_ROOT = Path(__file__).parent.parent

print(FILE_ROOT)

GOAL = 100
GOAL_REWARD = 1
HEAD_PROB = 0.4

DISCOUNT = 1

STATES = np.arange(GOAL + 1)

THRESHOLD = 1e-9

def get_new_value(state_value, state, action):
    return HEAD_PROB * state_value[state + action] + (1-HEAD_PROB) * state_value[state - action]

# policy evaluation
def policy_evaluation(state_value, policy):
    while True:
        old_state_value = state_value.copy()
        for state in STATES[1:GOAL]:
            action = policy[state]
            new_value = get_new_value(old_state_value, state, action)
            state_value[state] = new_value
        delta = abs(state_value - old_state_value).max()
        # print("old state values")
        # print(old_state_value)
        # print("new state values")
        # print(state_value)
        if delta < THRESHOLD:
            break
    return state_value


# policy improvement
def policy_improvement(state_value, policy, sweeps_history):
    old_policy = policy.copy()
    for state in STATES[1:GOAL]:
        possible_actions = np.arange(0, min(state, GOAL - state) + 1)
        action_returns = []
        for action in possible_actions:
            action_returns.append((action, get_new_value(state_value, state, action)))
        policy[state] = max(action_returns, key=lambda x: x[1])[0]
    
    print("old policy")
    print(old_policy)
    print("new policy")
    print(policy)

    sweeps_history.append(state_value)

    # import pdb; pdb.set_trace()

    return is_policy_changed(old_policy, policy), policy, sweeps_history

def is_policy_changed(old_policy, curr_policy) -> bool:
    return np.abs(old_policy - curr_policy).max() > THRESHOLD
    
def figure_4_3_policy_iteration():
    state_value = np.zeros(GOAL + 1)

    # set final state value = 1 to acccount for reward = 1
    state_value[GOAL] = GOAL_REWARD

    policy = np.zeros_like(STATES)

    sweeps_history = []

    while True:
        state_value = policy_evaluation(state_value, policy)
        policy_changed, policy, sweeps_history = policy_improvement(state_value, policy, sweeps_history)
        if not policy_changed:
            break

    plt.figure()
    fig, ax = plt.subplots(2, 1, figsize=(10, 20), dpi=200)

    for sweep, state_value in enumerate(sweeps_history):
        ax[0].plot(state_value, label='sweep {}'.format(sweep))
    ax[0].set_xlabel('Capital')
    ax[0].set_ylabel('Value estimates')
    ax[0].legend(loc='best')

    ax[1].scatter(STATES, policy)
    ax[1].set_xlabel('Capital')
    ax[1].set_ylabel('Final policy (stake)')

    plt.savefig(Path(FILE_ROOT) / 'images/figure_4_3_policy_iteration.png')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    figure_4_3_policy_iteration()
