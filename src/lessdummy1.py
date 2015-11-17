import sys
import json
import numpy as np
import operator
from collections import OrderedDict
sys.path.insert(0, './../PythonHelperTools')
from vqaTools.vqa import VQA

dataDir='../../cs446-project'
taskType='OpenEnded'
dataType='mscoco' # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType='train2014'
annFile='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

def readGloveData(glove_word_vec_file):
    f = open(glove_word_vec_file, 'r')
    rawData = f.readlines()
    word_vec_dict = {}
    for line in rawData:
        line = line.strip().split()
        tag = line[0]
        vec = line[1:]
        word_vec_dict[tag] = np.array(vec, dtype=float) 
    return word_vec_dict

def getWordVector(word, word_vec_dict):
	if word in word_vec_dict:
		return word_vec_dict[word]
	return np.zeros_like(word_vec_dict['hi'])

def getBOWVector(question, word_vec_dict):
	vector = np.zeros_like(word_vec_dict['hi'])
	for word in question:
		vector = vector + getWordVector(word, word_vec_dict)
	return vector

def createOneHotFeature(questions):
	wordFreq = {}
	firstWordFreq = {}
	secondWordFreq = {}
	thirdWordFreq = {}
	features = []
	for question in questions:
		question_split = question['question'].strip().replace('?',' ?').split()
		if question_split[0] in firstWordFreq:
			firstWordFreq[question_split[0]] += 1
		else:
			firstWordFreq[question_split[0]] = 1
		if question_split[1] in secondWordFreq:
			secondWordFreq[question_split[1]] += 1
		else:
			secondWordFreq[question_split[1]] = 1
		if question_split[2] in thirdWordFreq:
			thirdWordFreq[question_split[2]] += 1
		else:
			thirdWordFreq[question_split[2]] = 1


		for word in question_split:
			if word.lower() in wordFreq:
				wordFreq[word.lower()] = wordFreq[word.lower()] + 1
			else:
				wordFreq[word.lower()] = 1
	sortedWordFreq = sorted(wordFreq.items(), key=operator.itemgetter(1), reverse = True)
	sortedFirstWordFreq = sorted(firstWordFreq.items(), key = operator.itemgetter(1), reverse = True)
	sortedSecondWordFreq = sorted(secondWordFreq.items(), key = operator.itemgetter(1), reverse = True)
	sortedThirdWordFreq = sorted(thirdWordFreq.items(), key = operator.itemgetter(1), reverse = True)
	# for key, value in sortedWordFreq.iteritems():
	# 	print key + ' ' + str(value)
	index = 0
	for word, count in sortedWordFreq:
		if index >= 1000:
			break
		index += 1
		features.append(word)
	index = 0
	for word, count in sortedFirstWordFreq:
		if index >= 10:
			break
		index += 1
		features.append(word)
	index = 0
	for word, count in sortedSecondWordFreq:
		if index >= 10:
			break
		index += 1
		features.append(word)
	index = 0
	for word, count in sortedThirdWordFreq:
		if index >= 10:
			break
		index += 1
		features.append(word)
	return features

def getOneHotVector(question, oneHotFeatures):
	featureVector = np.zeros(len(oneHotFeatures))
	for word in question.strip().replace('?', ' ?').split(' '):
		if word in oneHotFeatures:
			featureVector[oneHotFeatures.index(word)] = 1
	return featureVector

def getQuestionVectors(imageID):
	pass



def main():
	glove_word_vec_file = './../glove/glove.6B.50d.txt'
	word_vec_dict = readGloveData(glove_word_vec_file)
	# dummyVec = getWordVector('hi', word_vec_dict)
	# print dummyVec
	vqaTrain = VQA(annFile, quesFile)
	annotations = vqaTrain.dataset['annotations']
	questions = vqaTrain.questions['questions']
	oneHotFeatures = createOneHotFeature(questions)
	BOWVector = getBOWVector('Who is that pokemon?', word_vec_dict)
	oneHotVector = getOneHotVector('Who is that pokemon?', oneHotFeatures)
	print BOWVector
	print oneHotVector
	

if __name__ == "__main__":
	main()