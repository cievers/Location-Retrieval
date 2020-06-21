import math
import os
import pickle

import h5py as h5py
import numpy

import clusters
import features
import utility

class Dataset:
	def __init__(self, name, clusterCount, clusterFilter):
		self.name = name
		self.clusterFilter = clusterFilter
		self.size = len(self.images())
		self.featuresSize = 0
		self.positions = None
		self.descriptors = None
		self.vocabulary = None
		self.vocabularySize = clusterCount
		self.vocabularyInertia = 0
	def initialize(self, extractor, output):
		self.initializeFeatures(extractor, output)
		self.initializeVocabulary(output)
	def initializeFeatures(self, extractor, output):
		outputPositions = 'positions/'+output+'.hdf5'
		outputDescriptors = 'descriptors/'+output+'.hdf5'
		if not os.path.isfile(outputPositions) and not os.path.isfile(outputDescriptors):
			self.positions = h5py.File(outputPositions, 'w')
			self.descriptors = h5py.File(outputDescriptors, 'w')
			utility.duration(
				utility.iterate,
				self.images(),
				lambda image, i: self.initializeFeature(extractor, image),
				False,
				task="Initialize dataset "+self.name+" features"
			)
		else:
			self.positions = h5py.File(outputPositions, 'r')
			self.descriptors = h5py.File(outputDescriptors, 'r')
		self.featuresSize = numpy.sum([len(self.imageDescriptors(image)) for image in self.images()])
	def initializeFeature(self, extractor, image):
		points, descriptors = features.imageFeatures(extractor, image)
		self.positions.create_dataset(image, data=[(point.pt[1], point.pt[0]) for point in points], compression='gzip', compression_opts=9)
		self.descriptors.create_dataset(image, data=descriptors, compression='gzip', compression_opts=9)
	def initializeVocabulary(self, output):
		outputArguments = output.split('_')
		outputVocabulary = 'vocabulary/'+outputArguments[0]+'_'+str(self.vocabularySize)+'_'+'_'.join(outputArguments[1:])+'.pickle'
		if not os.path.isfile(outputVocabulary):
			self.vocabulary = utility.duration(
				clusters.calculateClusters,
				self.clusterDescriptors(),
				self.vocabularySize,
				task="Initialize dataset "+self.name+" vocabulary"
			)
			with open(outputVocabulary, 'wb') as file:
				pickle.dump(self.vocabulary, file)
		else:
			with open(outputVocabulary, 'rb') as file:
				self.vocabulary = pickle.load(file)
		self.vocabularyInertia = self.vocabulary.inertia_

	def imagePositions(self, image):
		return self.positions[image]
	def imageDescriptors(self, image):
		return self.descriptors[image]
	def images(self):
		return features.imagePaths(self.name)
	def clusterImages(self):
		return features.filterPaths(self.images(), self.clusterFilter)
	def clusterDescriptors(self):
		return [descriptor for image in self.clusterImages() for descriptor in self.imageDescriptors(image)]

class Model:
	def __init__(self, dataset, extractor, output):
		self.dataset = dataset
		self.extractor = extractor
		self.dataset.initialize(extractor, output)
		self.representations = self.calculateRepresentations()
	def calculateVocabulary(self):
		return utility.duration(
			clusters.calculateClusters,
			utility.duration(
				features.imagesFeatures,
				self.extractor,
				self.dataset.clusterImages(),
				task="Image features"
			),
			clusters.calculateClusterCount(self.dataset.size),
			task="Clustering"
		)
	def calculateRepresentations(self):
		return utility.duration(
			self.representDataset,
			task="Dataset representations"
		)
	def representationSize(self):
		raise NotImplementedError()
	def representImage(self, image):
		raise NotImplementedError()
	def representDataset(self):
		representations = numpy.empty((self.dataset.size, self.representationSize()), dtype=numpy.int16)
		utility.iterate(self.dataset.images(), lambda image, i: (representations.__setitem__(i, self.representImage(image))), False)
		return representations
	def compareRepresentation(self, query, source, vectorNormalize, vectorDifference):
		raise NotImplementedError()
	def matchRepresentation(self, query, vectorNormalize, vectorDifference):
		# Returns a list of the differences between the given representation and each representation in the dataset
		matches = numpy.empty(self.dataset.size, dtype=numpy.float64)
		utility.iterate(self.representations, lambda representation, i: (matches.__setitem__(i, self.compareRepresentation(query, representation, vectorNormalize, vectorDifference))), False, None)
		return matches
	def matchRepresentationPartial(self, query, offset, vectorNormalize, vectorDifference):
		matches = numpy.zeros(self.dataset.size, dtype=numpy.float64)
		i = offset
		while i < self.dataset.size:
			matches[i] = self.compareRepresentation(query, self.representations[i], vectorNormalize, vectorDifference)
			i += 1
		return matches
	def matchRepresentationOthers(self, query, index, vectorNormalize, vectorDifference):
		# Match all other representations, except the representation of self at index
		matches = numpy.zeros(self.dataset.size, dtype=numpy.float64)
		i = 0
		while i < self.dataset.size:
			if not i == index:
				matches[i] = self.compareRepresentation(query, self.representations[i], vectorNormalize, vectorDifference)
			i += 1
		return matches
	def test(self, queryData, vectorNormalize, vectorDifference, output):
		file = h5py.File(output, 'w')
		results = file.create_dataset('matches', (queryData.size, self.dataset.size), dtype=numpy.float64, compression='gzip', compression_opts=9)
		if queryData == self.dataset:
			# If self-testing, do not match an image with itself
			utility.iterate(self.representations, lambda representation, i: results.__setitem__(i, self.matchRepresentationOthers(representation, i, vectorNormalize, vectorDifference)), False)
		else:
			# Else, test every query image against the complete set
			utility.iterate(queryData.images(), lambda image, i: results.__setitem__(i, self.matchRepresentation(self.representImage(image), vectorNormalize, vectorDifference)), False)
		file.close()
class Baseline(Model):
	def __init__(self, dataset, extractor, output):
		super().__init__(dataset, extractor, output)
	def representationSize(self):
		return self.dataset.vocabularySize
	def representImage(self, image):
		descriptors = features.imageFeatures(self.extractor, image)[1]
		words = numpy.zeros(self.dataset.vocabularySize, dtype=numpy.int16)
		for word in self.dataset.vocabulary.predict(descriptors):
			words[word] += numpy.int16(1)
		return words
	def compareRepresentation(self, query, source, vectorNormalize, vectorDifference):
		return vectorDifference(vectorNormalize(query), vectorNormalize(source))
class Grid(Baseline):
	def __init__(self, dataset, extractor, output, gridSize):
		self.gridSize = int(gridSize)
		super().__init__(dataset, extractor, output)
	def representationSize(self):
		return self.dataset.vocabularySize * self.gridSize * self.gridSize
	def representImage(self, image):
		# Creates a combined vector of N*N representations, one for each cell in the grid
		# The cells are ordered from left to right, top to bottom
		imageData = features.imageData(image)
		imageHeight, imageWidth = imageData.shape
		points, descriptors = features.imageDataFeatures(self.extractor, imageData)
		predictions = self.dataset.vocabulary.predict(descriptors)
		words = numpy.zeros(self.representationSize(), dtype=numpy.int16)
		i = 0
		while i < len(points):
			row = self.selectCell(points[i].pt[0], imageWidth)
			column = self.selectCell(points[i].pt[1], imageHeight)
			words[predictions[i] + column * self.dataset.vocabularySize + row * self.dataset.vocabularySize * self.gridSize] += numpy.int16(1)
			i += 1
		return words
	def selectCell(self, position, size):
		return math.floor(position / (size/self.gridSize))
class Skyline(Model):
	def __init__(self, dataset, extractor, output, skylineHeight, skylineWeight):
		self.skylineHeight = float(skylineHeight)
		self.skylineWeight = float(skylineWeight)
		super().__init__(dataset, extractor, output)
	def representationSize(self):
		return self.dataset.vocabularySize * 2
	def representImage(self, image):
		# Creates a combined vector of the representation of features above and below the skyline
		# The first half is the representation of features above the skyline, and the second half that of those below the skyline
		imageData = features.imageData(image)
		imageHeight = imageData.shape[0]
		points, descriptors = features.imageDataFeatures(self.extractor, imageData)
		predictions = self.dataset.vocabulary.predict(descriptors)
		words = numpy.zeros(self.representationSize(), dtype=numpy.int16)
		i = 0
		while i < len(points):
			belowSkyline = self.selectSky(points[i].pt[1], imageHeight)
			words[predictions[i] + belowSkyline * self.dataset.vocabularySize] += numpy.int16(1)
			i += 1
		return words
	def compareRepresentation(self, query, source, vectorNormalize, vectorDifference):
		queryNormalized = vectorNormalize(query)
		sourceNormalized = vectorNormalize(source)
		querySky = queryNormalized[:self.dataset.vocabularySize]
		sourceSky = sourceNormalized[:self.dataset.vocabularySize]
		queryGround = queryNormalized[self.dataset.vocabularySize:]
		sourceGround = sourceNormalized[self.dataset.vocabularySize:]
		return self.skylineWeight * vectorDifference(querySky, sourceSky) + (1 - self.skylineWeight) * vectorDifference(queryGround, sourceGround)
	def selectSky(self, position, height):
		# The skyline height needs to be given as percentage from the top of the image
		return 0 if (position / height) < self.skylineHeight else 1