import numpy as np

def exponential_distribution(x: list, lam: float) -> dict:
    if lam<0:
        return {
            "pdf": None,
            "cdf": None,
            "mean": None,
            "variance": None
        }
    pdf = []
    cdf = []
    for num in x:
        if num>=0:
            f_x = lam * np.exp(-lam * num)
            c_fx = 1 - np.exp(-lam * num)
        else:
            f_x = 0
            c_fx = 0
        
        pdf.append(np.round(f_x,4))
        cdf.append(np.round(c_fx,4))
    mean = round(1 / lam,4)
    variance = round(1 / (lam**2),4)

    return {
        "pdf": pdf,
        "cdf": cdf,
        "mean": mean,
        "variance": variance
    }
