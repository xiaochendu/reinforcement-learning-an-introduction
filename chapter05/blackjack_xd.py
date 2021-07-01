## SUTTON and MARIO refinforcement learning chapter 6
## Monte-Carlo methods

# Start with two cards
# Dealer has one card open


# Actions
# Hit
# Stick

# States (sums)
# When either player or dealer
# 1 ...  20
# 21 (terminal, win)
# >21 (terminal, lose)

# Dealer strategy
# Stick on 17 or more, hit on less

# Player strategy
# Hit if sum is 20 or 21
# Else stick

# State space (200 states)
# User's current sum (12 - 21)
# Dealer's showing card
# Usable ace

# Can iterate for the dealer first
# returns two values, (showing_card, dealer_sum)
# wait what if there's usable ace for the dealer


# %%
from pathlib import Path
from typing import NamedTuple, Tuple, Union
import random
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

FILE_ROOT = Path(__file__).parent.parent

def get_next_card():
    # randomly generate next card
    sampleList = range(1, 10+1)
    return random.choices(sampleList, weights=[*itertools.repeat(1, 9), 4], k=1)[0]

# %%
MAX_SUM = 21

LAMBDA = 1

HIT = 1
STICK = 0

TERMINAL_STATE = 21

# def step(user_sum, action):
#     draw = get_next_card()
#     return user_sum + draw

def default_policy(player_sum, *args):
    if player_sum < 20:
        return HIT
    else:
        return STICK

def generate_episode(initial_state=None, initial_action=None, policy=default_policy):
    # starting out
    if not initial_state:
        player_cards = [get_next_card(), get_next_card()]
        player_sum = sum(player_cards)

        dealer_visible_card = get_next_card()
        dealer_hidden_card = get_next_card()
        dealer_cards = [dealer_visible_card, dealer_hidden_card]
        dealer_sum = sum(dealer_cards)

        player_usable_ace = 1 in player_cards
        dealer_usable_ace = 1 in dealer_cards

        while player_sum < 12:
            # TODO: multiple usable aces
            if player_usable_ace:
                # use usable ace
                player_sum += 10
            else:
                player_draw = get_next_card()
                player_cards.append(player_draw)
                player_sum += player_draw
                if player_draw == 1:
                    player_usable_ace = True

        while dealer_sum < 12:
            if dealer_usable_ace:
                # use usable ace
                dealer_sum += 10
            else:
                dealer_draw = get_next_card()
                dealer_cards.append(dealer_draw)
                dealer_sum += dealer_draw
                if dealer_draw == 1:
                    dealer_usable_ace = True
    
    else:
        player_usable_ace, player_sum, dealer_visible_card = initial_state

        # fake the first two cards
        if player_usable_ace:
            player_cards = [11, player_sum-11]
        else:
            player_cards = [10, player_sum-10]

        dealer_hidden_card = get_next_card()
        dealer_cards = [dealer_visible_card, dealer_hidden_card]
        dealer_sum = sum(dealer_cards)
        dealer_usable_ace = 1 in dealer_cards

    player_states = [player_sum]
    player_actions = []

    # player moves
    while True:
        if initial_action:
            action = initial_action
            initial_action = False
        else:
            action = policy(player_sum, dealer_visible_card, player_usable_ace)
        
        player_actions.append(action)

        if action == HIT:
            player_draw = get_next_card()
            player_cards.append(player_draw)
            player_sum += player_draw
            if player_sum > TERMINAL_STATE:
                break
            else:
                player_states.append(player_sum)
            
        if action == STICK:
            break
    
    # dealer moves
    while True:        
        if dealer_sum < 17:
            dealer_draw = get_next_card()
            dealer_cards.append(dealer_draw)
            dealer_sum += dealer_draw
            
        else:
            break

    # check if player natural
    if player_cards[0] + player_cards[1] == 21:
        if dealer_cards[0] + dealer_cards[1] == 21:
            reward = 0
        else:
            reward = 1
    elif player_sum > TERMINAL_STATE:
        reward = -1

    # If dealer goes bust, player wins
    elif dealer_sum > TERMINAL_STATE:
        reward = 1

    # else determine who is closer to 21
    else:
        dealer_dist = abs(TERMINAL_STATE - dealer_sum)
        player_dist = abs(TERMINAL_STATE - player_sum)
        if dealer_dist > player_dist:
            reward = 1
        elif dealer_dist < player_dist:
            reward = -1
        else:
            reward = 0
    
    # print("dealer cards", dealer_cards)
    # print("dealer sum", dealer_sum)
    # print("player cards", player_cards)
    # print("player sum", player_sum)

    # print("player states")
    # print(player_states)

    # print("player actions", player_actions)

    # rewards = np.pad([reward], (len(player_states)-1, 0), 'constant')

    assert len(player_states) == len(player_actions), "num player states = num player actions"

    return player_states, dealer_visible_card, player_usable_ace, player_actions, reward

# %%

def get_curr_state(player_state, dealer_visible_card, player_usable_ace):
    """Convert state params into index"""
    return player_state-12, dealer_visible_card-1, int(player_usable_ace)

# plotting function
def figure_5_1():
    states_1 = mc_prediction(10000)
    states_2 = mc_prediction(500000)
    states_usable_ace_1, states_no_usable_ace_1 = states_1[:, :, 1], states_1[:, :, 0]
    states_usable_ace_2, states_no_usable_ace_2 = states_2[:, :, 1], states_2[:, :, 0]

    states = [states_usable_ace_1,
              states_usable_ace_2,
              states_no_usable_ace_1,
              states_no_usable_ace_2]

    titles = ['Usable Ace, 10000 Episodes',
              'Usable Ace, 500000 Episodes',
              'No Usable Ace, 10000 Episodes',
              'No Usable Ace, 500000 Episodes']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for state, title, axis in zip(states, titles, axes):
        fig = sns.heatmap(np.flipud(state), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    # plt.savefig(FILE_ROOT / 'images/figure_5_1.png')
    plt.show()
    plt.close()


def figure_5_2():
    state_action_values = mc_es_control(500000)

    state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
    state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)

    # get the optimal policy
    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)

    images = [action_usable_ace,
              state_value_usable_ace,
              action_no_usable_ace,
              state_value_no_usable_ace]

    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for image, title, axis in zip(images, titles, axes):
        fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    # plt.savefig(FILE_ROOT / 'images/figure_5_2.png')
    plt.show()
    plt.close()


def figure_5_3():
    true_value = -0.27726
    episodes = 10000
    runs = 100
    error_ordinary = np.zeros(episodes)
    error_weighted = np.zeros(episodes)
    for i in tqdm(range(0, runs)):
        ordinary_sampling_, weighted_sampling_ = mc_off_policy_evaluation(episodes)
        # get the squared error
        error_ordinary += np.power(ordinary_sampling_ - true_value, 2)
        error_weighted += np.power(weighted_sampling_ - true_value, 2)
    error_ordinary /= runs
    error_weighted /= runs

    plt.plot(np.arange(1, episodes + 1), error_ordinary, color='green', label='Ordinary Importance Sampling')
    plt.plot(np.arange(1, episodes + 1), error_weighted, color='red', label='Weighted Importance Sampling')
    plt.ylim(-0.1, 5)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel(f'Mean square error\n(average over {runs} runs)')
    plt.xscale('log')
    plt.legend()

    # plt.savefig('../images/figure_5_3.png')
    plt.show()
    plt.close()

# %%
def mc_prediction(stop_iter=100):
    # First-visit MC prediction
    Returns = np.zeros((10, 10, 2)) # 200 values
    # Returns = np.zeros_like(V)
    Returns_count = np.zeros_like(Returns)

    STOP_ITER = stop_iter
    for iter in tqdm(range(STOP_ITER)):
        # generate player run of S_0, A_0, R_1, S_1, ..., S_t-1, A_t-1, R_t
        player_states, dealer_visible_card, player_usable_ace, player_actions, reward = generate_episode()
        T = len(player_states)
        G = reward
        for t in reversed(range(T)):
            # print("time =", t)
            curr_state = get_curr_state(player_states[t], dealer_visible_card, player_usable_ace)
            # print("curr state", curr_state)
            Returns[curr_state] += G
            Returns_count[curr_state] += 1

    mask = Returns_count > 0
    Returns[mask] /= Returns_count[mask]
    # print("Returns")
    # print(Returns)
    return Returns

# %%
def mc_es_control(stop_iter=100):
    action_values = np.zeros((10, 10, 2, 2)) # (player_state, dealer_visible_card, usable_ace, policy)
    action_values_count = np.ones_like(action_values)
    
    def greedy_policy(player_sum, dealer_visible_card, player_usable_ace):
        player_state = player_sum - 12
        dealer_state = dealer_visible_card - 1
        usable_ace = int(player_usable_ace)

        curr_values = action_values[player_state, dealer_state, usable_ace] / action_values_count[player_state, dealer_state, usable_ace]
        
        choices = [action for action, value in enumerate(curr_values) if value == np.max(curr_values)]

        # arbitrarily break ties
        return np.random.choice(choices)

    STOP_ITER = stop_iter
    for iter in tqdm(range(STOP_ITER)):
        # generate player run of S_0, A_0, R_1, S_1, ..., S_t-1, A_t-1, R_t

        initial_state = [bool(np.random.choice([0, 1])),
            np.random.choice(range(12, 22)),
            np.random.choice(range(1, 11))]
        initial_action = np.random.choice([HIT, STICK])
        # curr_policy = greedy_policy if iter else default_policy
        curr_policy = greedy_policy
        player_states, dealer_visible_card, player_usable_ace, player_actions, reward = generate_episode(initial_state, initial_action, curr_policy)
        T = len(player_states)
        G = reward
        for t in reversed(range(T)):
            curr_state = get_curr_state(player_states[t], dealer_visible_card, player_usable_ace)
            curr_action = player_actions[t]
            curr_index = (*curr_state, curr_action)

            action_values[curr_index] += G
            action_values_count[curr_index] += 1

    mask = action_values_count > 0
    action_values[mask] /= action_values_count[mask]

    return action_values

# %%
def mc_off_policy_evaluation(stop_iter=100):
    # incremental implementation of weighted
    values = np.zeros((10, 10, 2)) # (player_state, dealer_visible_card, usable_ace, policy)
    values_ordinary = np.zeros_like(values)
    C_weighted = np.zeros_like(values)
    C_ordinary = np.zeros_like(C_weighted)



    returns = []
    ord_returns = []

    def random_policy(*args):
        return np.random.choice([HIT, STICK])

    STOP_ITER = stop_iter
    for iter in tqdm(range(STOP_ITER)):
        # generate player run of S_0, A_0, R_1, S_1, ..., S_t-1, A_t-1, R_t
        initial_state = [True, 13, 2]
        initial_action = random_policy()

        W = 1
        curr_policy = random_policy
        player_states, dealer_visible_card, player_usable_ace, player_actions, reward = generate_episode(initial_state, initial_action, curr_policy)
        T = len(player_states)
        G = reward
        numerator = 1.0
        denominator = 1.0 # denominator for uniform policy

        for t in reversed(range(T)):
            curr_state = get_curr_state(player_states[t], dealer_visible_card, player_usable_ace)
            curr_action = player_actions[t]
            # curr_index = (*curr_state, curr_action)
            C_weighted[curr_state] += W
            C_ordinary[curr_state] += 1
            values[curr_state] += W / C_weighted[curr_state] * (G - values[curr_state])
            values_ordinary[curr_state] += W / C_ordinary[curr_state] * (G - values_ordinary[curr_state])

            if curr_action == default_policy(player_states[t]):
                denominator *= 0.5
            else:
                numerator = 0.0
                break
            rho = numerator/denominator
            W *= rho

        eval_state = get_curr_state(13, 2, 1)
        returns.append(values[eval_state])
        ord_returns.append(values_ordinary[eval_state])
    # print(ord_returns)
    # print(returns)
    ord_returns = np.array(ord_returns)
    returns = np.array(returns)
    return ord_returns, returns

# %%
if __name__ == "__main__":
    # mc_prediction(100)
    # figure_5_1()
    # mc_es_control(100)
    # issue with simulation resulting in policy mismatch
    # figure_5_2()
    # mc_off_policy_evaluation(100)
    figure_5_3()
