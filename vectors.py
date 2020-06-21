import numpy

def normalizeSum(vector):
	# Normalize a vector by the sum of its dimensions (Amount of features counted in the histogram)
	return vector / numpy.sum(vector)
def normalizeLength(vector):
	# Normalize a vector to always have length 1
	return vector / numpy.sqrt(vector.dot(vector))

def differenceEuclidianDistance(vector1, vector2):
	# Euclidian distance between two vectors
	return numpy.linalg.norm(vector1 - vector2)
def differenceCosineSimilarity(vector1, vector2):
	# Flipped cosine similarity between two vectors, now 0 is exactly the same, and 2 is exactly opposite
	return 1 - numpy.dot(vector1, vector2) / (numpy.linalg.norm(vector1) * numpy.linalg.norm(vector2))
