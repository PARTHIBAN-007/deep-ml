from collections import defaultdict
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	centroids = list(initial_centroids)
	num_features = len(points[0]) if points else 0
	for _ in range(max_iterations):
		clusters = defaultdict(list)

		for point in points:
			min_dist = float('inf')
			nearest_centroid_idx = None

			for i, centroid in enumerate(centroids):
				dist = sum((x - xi) ** 2 for x, xi in zip(centroid, point))
				if dist < min_dist:
					min_dist = dist
					nearest_centroid_idx = i

			clusters[nearest_centroid_idx].append(point)

		new_centroids = []
		for i in range(k):
			cluster_points = clusters[i]
			if not cluster_points:
				new_centroids.append(centroids[i])
				continue

			mean_point = tuple(
			sum(p[dim] for p in cluster_points) / len(cluster_points)
			for dim in range(num_features)
			)
			new_centroids.append(mean_point)

			centroids = new_centroids

	return centroids