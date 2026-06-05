def law_of_total_probability(priors: dict, conditionals: dict) -> float:
    prob = 0
    for k,v in priors.items():
        prob += priors[k] * conditionals[k]
    return round(prob,4)
