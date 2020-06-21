import json
import os
import re

import h5py
import numpy
from scipy.spatial.distance import squareform

import test
import utility
import vectors

def readSimilarities(sourceData, queryData):
	with h5py.File('data/'+queryData+'/'+utility.nameSourceQuery(queryData, sourceData).replace('-', '_')+'_similarity.h5', 'r') as file:
		if sourceData == queryData:
			# Self-similarities are stored compacted
			return squareform(file['sim'][:].flatten())
		else:
			# Two different data sets are stored as full matrix
			return file['sim'][:]
def readPoses(dataset):
	with open('data/'+dataset+'/'+dataset+'.json', 'r') as file:
		data = json.load(file)
	return numpy.array(data['poses'])
def readDifferences(result):
	file = h5py.File('results/'+result, 'r')
	return file, file['matches']

def areMatches(similarities, query, source):
	# Return whether two images depict the same scene
	return bool(similarities[query][int(source)])
def countMatches(similarities, image):
	# Return the number of images that depict the same scene in the dataset
	return numpy.count_nonzero(similarities[image] == True)
def imagePhysicalAccuracy(sourcePoses, queryPoses, query, source):
	# Return physical difference between a query image and a match
	return vectors.differenceEuclidianDistance(queryPoses[query][:3], sourcePoses[int(source)][:3]), vectors.differenceCosineSimilarity(queryPoses[query][3:], sourcePoses[int(source)][3:])
def imageDifferences(matrix, image):
	# Return a sorted list of the differences between one image and the others in a given matrix
	if len(matrix) == len(matrix[0]):
		# Matrix is square, assume self-comparison
		differences = numpy.empty((len(matrix)-1, 2), dtype=numpy.float64)
		row = matrix[image]
		i = 0
		while i < image:
			# All similarities before the diagonal
			differences[i] = numpy.array([row[i], i])
			i += 1
		while i < differences.shape[0]:
			# All similarities beyond the diagonal
			differences[i] = numpy.array([row[i + 1], i + 1])
			i += 1
	else:
		# Non-square matrix, must be differing source and query
		differences = numpy.empty((len(matrix[0]), 2), dtype=numpy.float64)
		utility.iterate(matrix[image], lambda difference, i: differences.__setitem__(i, (difference, i)), False, None)

	return differences[differences[:,0].argsort()]
def imagePrecision(similarities, differences, image, amount):
	# Return the precision of the amount of items retrieved (matches/items)
	i = 0
	matches = 0
	while i < amount:
		if areMatches(similarities, image, differences[i][1]):
			matches += 1
		i += 1
	return matches/amount
def imageRecall(similarities, differences, image, amount):
	# Return the recall of the matches for one image (matches/allMatches)
	totalMatches = countMatches(similarities, image)
	if totalMatches == 0:
		return 0
	i = 0
	matches = 0
	while i < amount:
		if areMatches(similarities, image, differences[i][1]):
			matches += 1
		i += 1
	return matches/totalMatches
def imageRecallRate(similarities, differences, image, amount):
	# Recall whether at least one of the retrieved items is a match
	i = 0
	while i < amount:
		if areMatches(similarities, image, differences[i][1]):
			return 1
		i += 1
	return 0
def imageMetrics(similarityMatrix, differenceMatrix, sourcePoses, queryPoses, image, amounts):
	differenceList = imageDifferences(differenceMatrix, image)
	physicalAccuracy = imagePhysicalAccuracy(sourcePoses, queryPoses, image, differenceList[0][1])
	return {
		'precision': {amount:imagePrecision(similarityMatrix, differenceList, image, amount) for amount in amounts},
		'recall': {amount:imageRecall(similarityMatrix, differenceList, image, amount) for amount in amounts},
		'recallRate': {amount:imageRecallRate(similarityMatrix, differenceList, image, amount) for amount in amounts},
		'transform': physicalAccuracy[0],
		'rotation': physicalAccuracy[1]
	}
def fileMetrics(result, sourceData, queryData, amounts):
	similarityMatrix = utility.duration(
		readSimilarities,
		sourceData.name,
		queryData.name,
		task="Read similarity matrix"
	)
	differenceFile, differenceMatrix = utility.duration(
		readDifferences,
		result,
		task="Read difference matrix"
	)
	sourcePoses = utility.duration(
		readPoses,
		sourceData.name,
		task="Read poses",
	)
	queryPoses = utility.duration(
		readPoses,
		queryData.name,
		task="Read poses",
	)

	totalMetrics = {
		'precision': {amount:numpy.empty(queryData.size, dtype=numpy.float64) for amount in amounts},
		'recall': {amount:numpy.empty(queryData.size, dtype=numpy.float64) for amount in amounts},
		'recallRate': {amount:numpy.empty(queryData.size, dtype=numpy.float64) for amount in amounts},
		'transform': numpy.empty(queryData.size, dtype=numpy.float64),
		'rotation': numpy.empty(queryData.size, dtype=numpy.float64)
	}

	utility.iterate(range(queryData.size), lambda _, i: registerMetrics(totalMetrics, imageMetrics(similarityMatrix, differenceMatrix, sourcePoses, queryPoses, i, amounts), i), False)

	differenceFile.close()

	return {
		'precision': {
			amount: {
				'mean': numpy.mean(totalMetrics['precision'][amount]),
				'variance': numpy.var(totalMetrics['precision'][amount]),
				'standardDeviation': numpy.std(totalMetrics['precision'][amount])
			} for amount in totalMetrics['precision']
		},
		'recall': {
			amount: {
				'mean': numpy.mean(totalMetrics['recall'][amount]),
				'variance': numpy.var(totalMetrics['recall'][amount]),
				'standardDeviation': numpy.std(totalMetrics['recall'][amount])
			} for amount in totalMetrics['recall']
		},
		'recallRate': {
			amount: {
				'mean': numpy.mean(totalMetrics['recallRate'][amount]),
				'variance': numpy.var(totalMetrics['recallRate'][amount]),
				'standardDeviation': numpy.std(totalMetrics['recallRate'][amount])
			} for amount in totalMetrics['recallRate']
		},
		'transform': {
			'mean': numpy.mean(totalMetrics['transform']),
			'variance': numpy.var(totalMetrics['transform']),
			'standardDeviation': numpy.std(totalMetrics['transform'])
		},
		'rotation': {
			'mean': numpy.mean(totalMetrics['rotation']),
			'variance': numpy.var(totalMetrics['rotation']),
			'standardDeviation': numpy.std(totalMetrics['rotation'])
		}
	}
def registerMetrics(total, entry, index):
	for amount in entry['precision']:
		total['precision'][amount][index] = entry['precision'][amount]
	for amount in entry['recall']:
		total['recall'][amount][index] = entry['recall'][amount]
	for amount in entry['recallRate']:
		total['recallRate'][amount][index] = entry['recallRate'][amount]
	total['transform'][index] = entry['transform']
	total['rotation'][index] = entry['rotation']
def metrics(file, amounts):
	outputPath = os.path.join('evaluations', file.replace('.hdf5', '.json'))
	print("Evaluating "+file)
	print("Storing in "+outputPath)
	sourceData, queryData = utility.getSourceQuery(file.split('_')[0])
	utility.store(fileMetrics(file, test.availableDataSets[sourceData], test.availableDataSets[queryData], amounts), outputPath)
def allMetrics(amounts):
	for path, directories, files in os.walk('results'):
		for file in files:
			if file.endswith('.hdf5'):
				metrics(file, amounts)
def combine(selector):
	total = {}
	for path, directories, files in os.walk('evaluations'):
		for file in files:
			if file.endswith('.json'):
				if re.search(selector, file):
					total[file[:-5]] = utility.load(os.path.join('evaluations', file))
		break
	return total
def table(data, metric, attribute):
	# Create a table of mean, variance or standardDeviation for precision, recall or recallRate from a combined evaluation object
	return [[
		'model',
	    *list(data.values())[0][metric].keys()
        ], *[
		[
			model,
			*[data[model][metric][k][attribute] for k in data[model][metric]]
		] for model in data
	]]
def formatTable(data):
	string = '\t'.join(data[0])+'\n'
	for row in data[1:]:
		string += row[0]
		for value in row[1:]:
			string += '\t' + str(round(value * 100, 3)).replace('.', ',') + '%'
		string += '\n'
	print(string)
def series(data, metric):
	return [(model, data[model][metric]['mean'], data[model][metric]['standardDeviation']) for model in data]
def formatSeries(data):
	string = ''
	for entry in data:
		string += entry[0]+'\t'+str(round(entry[1], 3)).replace('.', ',')+'\t'+str(round(entry[2], 3)).replace('.', ',')+'\n'
	print(string)

