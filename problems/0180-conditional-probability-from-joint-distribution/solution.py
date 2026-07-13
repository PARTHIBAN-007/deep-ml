def conditional_probability(joint_distribution: dict) -> float:
	p_a_and_b = joint_distribution.get(('A', 'B'), 0.0)
	p_b = joint_distribution.get(('A', 'B'), 0.0) + joint_distribution.get(('`A', 'B'), 0.0)
	if p_b == 0:
		return 0.0  
	return round(p_a_and_b / p_b, 4)