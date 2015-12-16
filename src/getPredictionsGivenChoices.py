
import numpy as np
import json
import sys
import lessdummy1 as ld
## Bhargav
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout
##
import scipy.io as sio

import cocoIDToFeatures as pramod
tfile = './../features/coco_vgg_IDMap.txt'


sys.path.insert(0, './../VQA/PythonHelperTools')
from vqaTools.vqa import VQA

RESULTS_FILE = './../data/results_val_new.txt'

X_TEST_FILE_NAME = 'X_test'
Y_TEST_FILE_NAME = 'Y_test'

def evalResults():
	dataDir = './../VQA'
	taskType = 'MultipleChoice'
	dataType = 'mscoco' # 'mscoco' for real and 'abstract_v002' for abstract
	dataSubType = 'train2014'
	annFile = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType)
	quesFile = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType)
	imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)
	vqaTrain = VQA(annFile, quesFile)
	dummyano = vqaTrain.dataset['annotations']
	answerFeatures = ld.createAnswerFeatures(dummyano)

	dataDir2 = './../VQA'
	taskType2 = 'MultipleChoice'
	dataType2 = 'mscoco' # 'mscoco' for real and 'abstract_v002' for abstract
	dataSubType2 = 'val2014'
	# number = '100'
	annFile2 = '%s/Annotations/%s_%s_annotations.json' % (dataDir2, dataType2, dataSubType2)
	quesFile2 = '%s/Questions/%s_%s_%s_questions.json' % (dataDir2, taskType2, dataType2, dataSubType2)
	resultFile = './../Results/MultipleChoice_mscoco_analysis1_second_results.json' 
	imgDir2 = '%s/Images/%s/%s/' % (dataDir2, dataType2, dataSubType2)

	modelReader = open('./model_definition_100iter.json')
	json_read = modelReader.read()
	model = model_from_json(json_read)
	model.load_weights('./model_weights_100iter.h5py')
	
	vqaVal = VQA(annFile2, quesFile2)
	FILE_INDEX = 0
    
	total = 0.0
	correct = 0.0

	resultsDicts = []
	x_test = []
	y_test = []
	glove_word_vec_file = './../glove/glove.6B.300d.txt'
	word_vec_dict = ld.readGloveData(glove_word_vec_file)
	imageDict = pramod.generateDictionary(tfile)
	feats = sio.loadmat('./../features/coco/vgg_feats.mat')['feats']
	for quesID, annotation in vqaVal.qa.iteritems():
		# print quesID
		# if quesID not in vqaVal.qqa.keys():
		# 	continue
		question = vqaVal.qqa[quesID]
		choicesList = vqaVal.qqa[quesID]['multiple_choices']
		# print choicesList
		setChoices = set(choicesList)
		setAnswers = set(answerFeatures)
		choiceAndAnswer = list(setChoices.intersection(setAnswers))
		choiceIndex = []
		for choice in choiceAndAnswer:
			choiceIndex.append(answerFeatures.index(choice))
		#print choiceIndex
		questionVector = ld.getBOWVector(question['question'].strip().replace('?', ' ?').split(), word_vec_dict) 
		imgID = annotation['image_id']
		imageVector = np.asarray(feats[:, imageDict[imgID]])
		temp_dict = {}
		ansString = annotation['multiple_choice_answer']
		temp_dict['question_id'] = quesID
		# answerVector = ld.getAnswerVector(ansString, answerFeatures)
		temp_x_test = np.append(imageVector, questionVector)
		# temp_y_test = answerVector
		x_test = np.asarray([temp_x_test])
		# y_test = np.asarray([temp_y_test])
		predictions = model.predict_classes(x_test, verbose = False)
		predict_probaResult = model.predict_proba(x_test,verbose = False)
		# print "###############Sanity Check############"
		# print predict_probaResult.size
		# print predict_probaResult
		# print predict_probaResult[7]
		# print predict_probaResult
		maxPred = 0.0
		# print "#######################################"
		print choiceIndex
		for item in choiceIndex:
			print len(choiceIndex), item,answerFeatures[item]
		for item in choiceIndex:
			print item,answerFeatures[item],predict_probaResult[0][item]
			if(maxPred < predict_probaResult[0][item]):
				maxPred = predict_probaResult[0][item]
				maxIndex = item
		print maxPred, maxIndex, answerFeatures[maxIndex]
		# temp_dict['answer'] = answerFeatures[predictions[0]]
		temp_dict['answer'] = answerFeatures[maxIndex]
		resultsDicts.append(temp_dict)
	writer = open(resultFile, 'w')
	json_dump = json.dumps(resultsDicts)
	writer.write(json_dump)
		
def main():
	evalResults()

if __name__ == "__main__":
	main()
