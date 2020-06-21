import json
import time

import h5py
import numpy

def duration(function, *arguments, task=''):
	startTime = time.time()
	result = function(*arguments)
	endTime = time.time()
	if task == '':
		print("Duration: " + str(endTime - startTime))
	else:
		print("Duration (" + task + "): " + str(endTime - startTime))
	return result
def iterate(items, function, results=True, milestoneStart=2, milestoneFactor=2, milestoneLinear=None):
	nextMilestone = milestoneStart
	values = []
	for i in range(len(items)):
		value = function(items[i], i)
		if results:
			values.append(value)
		if milestoneLinear is not None and i % milestoneLinear == 0 and not i == 0:
			print("Milestone "+str(i))
		if nextMilestone is not None and i == nextMilestone:
			print("Milestone "+str(i))
			nextMilestone = int(nextMilestone * milestoneFactor)

	return values
def store(data, path):
	with open(path, 'w+') as file:
		json.dump(data, file)
def load(path):
	with open(path, 'r') as file:
		return json.load(file)

def nameSourceQuery(source, query):
	if source == query:
		return source
	return source + '-' + query
def getSourceQuery(name):
	if '-' in name:
		return name.split('-')[0], name.split('-')[1]
	return name, name