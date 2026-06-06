import numpy as np

def tanh(x):
	return ((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	Wx = np.array(Wx)
	Wh = np.array(Wh)
	b = np.array(b)
	input_sequence = np.array(input_sequence)
	for x in input_sequence:
		computation =   Wx @ x +  Wh @ initial_hidden_state  + b
		initial_hidden_state = tanh(computation)
	return initial_hidden_state