import numpy as np

def qlora_forward(
	x: list[list[float]],
	quantized_W: list[list[int]],
	scale: float,
	zero_point: float,
	A: list[list[float]],
	B: list[list[float]],
	alpha: float = 1.0
) -> list[list[float]]:
	x_arr = np.array(x)
	q_w_arr = np.array(quantized_W)
	A_arr = np.array(A,dtype = np.float32)
	B_arr = np.array(B,dtype = np.float32)

	W_dequantized = q_w_arr * scale + zero_point
	base_output = np.matmul(x_arr, W_dequantized)
	rank = A_arr.shape[0]
	scaling_factor = alpha / rank
	lora_output = np.matmul(np.matmul(x_arr, B_arr), A_arr) * scaling_factor
	final_output = base_output + lora_output 
	return final_output.tolist()
