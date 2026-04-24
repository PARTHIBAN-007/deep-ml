import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
    g = np.array(gradient, dtype=float)
    
    magnitude = np.linalg.norm(g)
    
    if magnitude == 0:
        zero_vec = [0.0] * len(g)
        return {
            'magnitude': 0.0,
            'direction': zero_vec,
            'descent_direction': zero_vec
        }
    
    direction = (g / magnitude).tolist()
    descent_direction = (-g / magnitude).tolist()
    
    return {
        'magnitude': round(magnitude, 4),
        'direction': [round(x, 4) for x in direction],
        'descent_direction': [round(x, 4) for x in descent_direction]
    }