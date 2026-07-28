import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    states = np.zeros(num_steps + 1, dtype=int)
    states[0] = initial_state
    current_state = initial_state
    num_states = transition_matrix.shape[0]
    all_states = np.arange(num_states)

    for i in range(1, num_steps + 1):
        probabilities = transition_matrix[current_state]        
        current_state = np.random.choice(all_states, p=probabilities)
        states[i] = current_state
    return states