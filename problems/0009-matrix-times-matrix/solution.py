import numpy as np
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
            try:
                return np.matmul(a, b)
            except Exception as e:
                return -1