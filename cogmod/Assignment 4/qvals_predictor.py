import numpy as np
import random

def name():
  return 'qvals_predictor'

q_vals = None

def play(n_round, own_prev_moves, other_prev_moves, total_rounds, alpha=0.5):
    global q_vals
    if n_round == 0:
        # First move
        print('Initialized')
        q_vals = np.zeros((4, 2))
        return random.randint(0, 1)
    elif n_round == 1:
        return random.randint(0, 1)
    else:
        state = get_state(own_prev_moves[-1], other_prev_moves[-1])
        prev_state = get_state(own_prev_moves[-2], other_prev_moves[-2])
        q_vals = calculate_qvals(q_vals, state, prev_state, other_prev_moves, alpha)
        print('Q-values:', q_vals)
        other_action, certainty = predict_action(q_vals, state)
        action = find_best_move(other_action, certainty)
        print('State:', state, 'Action:', action, 'Certainty:', certainty, 'Prediction:', other_action, 'True opp move:', other_prev_moves[-1], 'Own move:', own_prev_moves[-1])
        return action

def find_best_move(other_action, certainty):
    if other_action == 0:
        if certainty >= 0.2:
            return 1
        else:
            return 0
    else:
        if certainty >= 0.2:
            return 0
        else:
            return 1

def get_state(own_move, other_move):
    # States: 0 = (0, 0), 1 = (0, 1), 2 = (1, 0), 3 = (1, 1)
    # Map the moves to states
    if own_move == 0 and other_move == 0:
        return 0
    elif own_move == 0 and other_move == 1:
        return 1
    elif own_move == 1 and other_move == 0:
        return 2
    else:
        return 3
    
def predict_action(q_vals, state):
    q_max = np.max(q_vals[state])
    max_actions = np.where(q_vals[state] == q_max)[0]
    a = random.choice(max_actions)
    certainty = np.abs(q_vals[state, 0] - q_vals[state, 1])
    return a, certainty

def calculate_qvals(q_vals, state, prev_state, other_moves, alpha):
    gamma = 0.9
    # Calculate the state based on the previous moves
    reward0 = 1 if (other_moves[-1] == 0) else -0.5
    reward1 = 1 if (other_moves[-1] == 1) else -0.5
    # Update Q-values using the Q-learning formula
    q_vals[prev_state, 0] = q_vals[prev_state, 0] + alpha * (reward0+ gamma * np.max(q_vals[state]) - q_vals[prev_state, 0])
    q_vals[prev_state, 1] = q_vals[prev_state, 1] + alpha * (reward1+ gamma * np.max(q_vals[state]) - q_vals[prev_state, 1])
    return q_vals
