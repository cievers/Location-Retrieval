import json

import cv2

SIFT = cv2.xfeatures2d.SIFT_create()

def filterPaths(paths, function):
	return [path for path in paths if function(path)]
def filterTenthImage(path):
	return path.endswith('0.png')
def filterTwentiethImage(path):
	return path.endswith('10.png') or path.endswith('30.png') or path.endswith('50.png') or path.endswith('70.png') or path.endswith('90.png')

def imagePaths(dataset):
	with open('data/'+dataset+'/'+dataset+'.json', 'r') as f:
		data = json.load(f)
	return data['im_paths']
def imageData(image):
	# Read image, and convert to grayscale
	return cv2.cvtColor(cv2.imread('data/'+image), cv2.COLOR_RGB2GRAY)
def imageDataFeatures(extractor, data):
	return extractor.detectAndCompute(data, None)
def imageFeatures(extractor, image):
	# Returns feature object, descriptor vector, image shape
	return imageDataFeatures(extractor, imageData(image))
def imagesFeatures(extractor, images):
	# Returns list of descriptor vectors for each image in the given list
	return [descriptor for image in images for descriptor in imageFeatures(extractor, image)[1]]
