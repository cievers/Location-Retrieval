import sys

import clusters
import features
import models
import utility
import vectors

availableModels = {
	'base': models.Baseline,
	'grid': models.Grid,
	'skyline': models.Skyline
}
availableCompleteDataSets = {
	'w16': models.Dataset('W16', 6000, features.filterTwentiethImage),
	'w17': models.Dataset('W17', 3000, features.filterTenthImage),
	'w18': models.Dataset('W18', 4000, features.filterTwentiethImage)
}
availableDataSets = {
	'sample': models.Dataset('Sample', 200, lambda path: True),
	'example': models.Dataset('Sample', 200, lambda path: True),
	**availableCompleteDataSets
}
availableExtractors = {
	'sift': features.SIFT
}
availableVectorNormalizers = {
	'sum': vectors.normalizeSum,
	'length': vectors.normalizeLength
}
availableVectorComparators = {
	'euclid': vectors.differenceEuclidianDistance,
	'cosine': vectors.differenceCosineSimilarity
}

def test(sourceData, queryData, model, extractor, vectorNormalize, vectorDifference, *modelArguments):
	# Start the test function for a model based on string arguments, and generate a corresponding result hdf5 file
	if not sourceData in availableDataSets:
		print("Unknown data set '" + sourceData + "', available data sets: " + ", ".join(availableDataSets.keys()))
		return
	if not queryData in availableDataSets:
		print("Unknown data set '" + queryData + "', available data sets: " + ", ".join(availableDataSets.keys()))
		return
	if not model in availableModels:
		print("Unknown model '"+model+"', available models: "+", ".join(availableModels.keys()))
		return
	if not extractor in availableExtractors:
		print("Unknown feature extractor '"+extractor+"', available extractors: "+", ".join(availableExtractors.keys()))
		return
	if not vectorNormalize in availableVectorNormalizers:
		print("Unknown function '"+vectorNormalize+"' for vector normalization, available functions: "+", ".join(availableVectorNormalizers.keys()))
		return
	if not vectorDifference in availableVectorComparators:
		print("Unknown function '"+vectorDifference+"' for vector comparison, available functions: "+", ".join(availableVectorComparators.keys()))
		return
	utility.duration(
		executeTest,
		sourceData,
		queryData,
		model,
		extractor,
		vectorNormalize,
		vectorDifference,
		*modelArguments,
		task="Test"
	)
def executeTest(sourceData, queryData, model, extractor, vectorNormalize, vectorDifference, *modelArguments):
	dataFile = '_'.join([sourceData, extractor])
	modelInstance = availableModels[model](
		availableDataSets[sourceData],
		availableExtractors[extractor],
		dataFile,
		*modelArguments
	)
	outputFile = 'results/' + '_'.join([utility.nameSourceQuery(sourceData, queryData), '-'.join([model, *[str(argument) for argument in modelArguments]]), str(modelInstance.dataset.vocabularySize), extractor, vectorNormalize, vectorDifference]) + '.hdf5'
	utility.duration(
		modelInstance.test,
		availableDataSets[queryData],
		availableVectorNormalizers[vectorNormalize],
		availableVectorComparators[vectorDifference],
		outputFile,
		task="Full matrix comparison"
	)
	print("Result written to "+outputFile)

def elbow(dataset, measureDistance, clusterFilter, extractor, output):
	# Measure various distortion amounts for difference cluster counts for a dataset
	initialize = models.Dataset(dataset, 1, clusterFilter)
	initialize.initializeFeatures(extractor, output)
	estimate = clusters.calculateClusterCount(initialize.featuresSize)
	clusterCounts = clusters.nearRoundNumbers(estimate, measureDistance)
	distortions = {}
	for count in clusterCounts:
		model = models.Dataset(dataset, max(int(count), 1), clusterFilter)
		model.initialize(extractor, output)
		distortions[count] = model.vocabularyInertia
	return distortions

if __name__ == '__main__':
	test(*sys.argv[1:])
