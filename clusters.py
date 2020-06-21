import math

import sklearn.cluster

def calculateClusterCount(sampleSize):
	return round(math.sqrt(sampleSize/2))
def calculateClusters(descriptors, count):
	classifier = sklearn.cluster.MiniBatchKMeans(n_clusters=count)
	classifier.fit(descriptors)
	return classifier
def closestCluster(clusters, descriptor):
	return clusters.predict([descriptor])[0]
def closestClusters(clusters, descriptors):
	return clusters.predict(descriptors)
def exportClusters(clusters):
	return {
		'words': clusters.cluster_centers_.tolist(),
		'size': len(clusters.cluster_centers_),
		'distortion': clusters.inertia_
	}

def nearestRoundNumbers(x):
	size = math.pow(10, int(math.log10(x)))
	factor = int(x / size)
	return size * factor, size * (factor + 1)
def nearRoundNumbers(x, amount=1):
	nearest = nearestRoundNumbers(x)
	scale = nearest[1] - nearest[0]
	result = []
	i = amount
	while i > 0:
		i -= 1
		result.append(nearest[0] - scale * i)
	i = 0
	while i < amount:
		result.append(nearest[1] + scale * i)
		i += 1
	return result