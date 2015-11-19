import sys
import numpy as np
import operator
import json

import cocoIDToFeatures as pramod
import scipy.io as sio

sys.path.insert(0, './../VQA/PythonHelperTools')
from vqaTools.vqa import VQA

## For getting image vectors
tfile = './../features/coco_vgg_IDMap.txt'

##

## GLOBAL CONSTANTS
FILE_LIMIT = 50000
FILE_PATH = './../data/'
X_TRAIN_FILE_NAME = 'X_train'
Y_TRAIN_FILE_NAME = 'Y_train'
X_TEST_FILE_NAME = 'X_test'
Y_TEST_FILE_NAME = 'Y_test'

##

dataDir = '../../cs446-project/data'
taskType = 'MultipleChoice'
dataType = 'mscoco' # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType = 'train2014'
annFile = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType)
quesFile = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)




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

def getAnswerVector(answer, answerFeatures):
	featureVector = np.zeros(len(answerFeatures))
	for word in answer.strip().split(' '):
		if word in answerFeatures:
			featureVector[answerFeatures.index(word)] = 1
	return featureVector

def main():
    glove_word_vec_file = './../glove/glove.6B.300d.txt'
    word_vec_dict = readGloveData(glove_word_vec_file)
    vqaTrain = VQA(annFile, quesFile)
    annotations = vqaTrain.dataset['annotations']
    questions = vqaTrain.questions['questions']
    answerFeatures = createAnswerFeatures(annotations)

    # Dumping answer features
    answer_features_list = open('answer_feature_list.json', 'w')
    answer_features_list.write(json.dumps(answerFeatures))

    # For getting image vectors
    imageDict = pramod.generateDictionary(tfile)
    feats = sio.loadmat('./../features/coco/vgg_feats.mat')['feats']

    data = []
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    FILE_INDEX = 0
    for question in questions:
        # quesItem = {}
        # print question
        questionVector = getBOWVector(question['question'].strip().replace('?', ' ?').split(), word_vec_dict)
        imgID = question['image_id']
        imageVector = np.asarray(feats[:,imageDict[imgID]])
        # quesItem['image_id'] = imgID
        # quesItem['question'] = question['question'].replace('?', ' ?').split(' ')
        annotations = vqaTrain.loadQA(ids = [question['question_id']])
        for annotation in annotations:
        	ansString = annotation['multiple_choice_answer']
        	answerVector = getAnswerVector(ansString, answerFeatures)
        	temp_X_train = np.append(imageVector, questionVector)
        	temp_Y_train = answerVector
        	X_train.append(temp_X_train)
        	Y_train.append(temp_Y_train)
        	if len(X_train) >= FILE_LIMIT:
        		train_x_file = open(FILE_PATH+X_TRAIN_FILE_NAME+str(FILE_INDEX)+'.npy', 'w')
        		train_y_file = open(FILE_PATH+Y_TRAIN_FILE_NAME+str(FILE_INDEX)+'.npy', 'w')
        		np.save(train_x_file, X_train)
        		np.save(train_y_file, Y_train)
        		X_train = []
        		Y_train = []
        		FILE_INDEX = FILE_INDEX + 1
        	# print len(X_train)
        # if len(annotations) != 1:
            # print imgID, " has annotations ", len(annotations)

        # for ann in annotations:
            # quesItemCopy = dict(quesItem)
            # ansString = ann['multiple_choice_answer']
            # quesItemCopy['answer'] = ansString
            # data.append(quesItemCopy)
    if len(X_train) > 0:
      	train_x_file = open(FILE_PATH+X_TRAIN_FILE_NAME+str(FILE_INDEX)+'.npy', 'w')
       	train_y_file = open(FILE_PATH+Y_TRAIN_FILE_NAME+str(FILE_INDEX)+'.npy', 'w')
       	np.save(train_x_file, X_train)
       	np.save(train_y_file, Y_train)
       	X_train = []
       	Y_train = []


    




    # output_data_file = open('preprocessed_data.json', 'w')
    # output_data_file.write(json.dumps(data))
    # output_data_file.close()

if __name__ == "__main__":
    main()
