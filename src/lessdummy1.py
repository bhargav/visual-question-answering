import sys
import json
import numpy as np
import operator
from collections import OrderedDict
import cocoIDToFeatures as pramod
import scipy.io as sio

## Bhargav
from keras.models import Sequential
from keras.layers.core import Dense, Dropout



##

sys.path.insert(0, './../VQA/PythonHelperTools')
from vqaTools.vqa import VQA

## For getting image vectors
tfile = './../features/coco_vgg_IDMap.txt'



##
dataDir='../../cs446-project/data'
taskType='MultipleChoice'
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

def createOneHotFeatures(questions):
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

def createAnswerFeatures(annotations):
	answerCount = {}
	answerFeatures = []
	for annotation in annotations:
		answer = annotation['multiple_choice_answer']
		# print question
		for word in answer.split():
			if word in answerCount:
				answerCount[word] += 1
			else:
				answerCount[word] = 1
	sortedAnswerCount = sorted(answerCount.items(), key=operator.itemgetter(1), reverse = True)
	index = 0
	for word, count in sortedAnswerCount:
		if (index >= 1000):
			break
		index = index + 1
		answerFeatures.append(word)
	# print len(answerFeatures)
	# print answerFeatures
	return answerFeatures
	# print questions[1]
	# for question in questions:
	# 	question_split = question['question'].strip().replace('?',' ?').split()
	# 	for

def getAnswerVector(answer, answerFeatures):
	featureVector = np.zeros(len(answerFeatures))
	for word in answer.strip().split(' '):
		if word in answerFeatures:
			featureVector[answerFeatures.index(word)] = 1
	return featureVector

def getMLPModel(input_size, output_size):
    model = Sequential()

    # Two hidden layers
    model.add(Dense(1000, input_dim = input_size, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, input_dim = input_size, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))

    # Output layer for probability
    model.add(Dense(output_size, init='uniform', activation='softmax'))

    model.compile(loss='mean_square_error', optimizer='sgd')

    return model

def main():
	glove_word_vec_file = './../glove/glove.6B.50d.txt'
	word_vec_dict = readGloveData(glove_word_vec_file)
	vqaTrain = VQA(annFile, quesFile)
	annotations = vqaTrain.dataset['annotations']
	questions = vqaTrain.questions['questions']
	answerFeatures = createAnswerFeatures(annotations)

	## For getting image vectors
	imageDict = pramod.generateDictionary(tfile)
	feats = sio.loadmat('./../features/coco/vgg_feats.mat')['feats']
	X_train = []
	Y_train = []
	##
	for question in questions:
		quesString = question['question'].replace('?', ' ?').split(' ')
		wordVector = getBOWVector(quesString, word_vec_dict)
		imgID = question['image_id']
		imageVector = np.asarray(feats[:,imageDict[imgID]])
		anns = vqaTrain.loadQA(ids = [question['question_id']])
		for ann in anns:
			# print quesString
			ansString = ann['multiple_choice_answer']
			answerVector = getAnswerVector(ansString, answerFeatures)
			temp_X_train = np.append(imageVector, wordVector)
			# X_train.append(wordVector)
			temp_Y_train = answerVector
			X_train.append(temp_X_train)
			Y_train.append(temp_Y_train)
			print len(X_train)
			# print X_train
			# print Y_train 
		




		# break
	
	# model = getMLPModel(len(X_train), len(Y_train))


if __name__ == "__main__":
	main()
