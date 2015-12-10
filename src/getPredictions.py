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

RESULTS_FILE = './../data/results.txt'

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
	annFile2 = '%s/Annotations/%s_%s_annotations.json' % (dataDir2, dataType2, dataSubType2)
	quesFile2 = '%s/Questions/%s_%s_%s_questions.json' % (dataDir2, taskType2, dataType2, dataSubType2)
	imgDir2 = '%s/Images/%s/%s/' % (dataDir2, dataType2, dataSubType2)

	modelReader = open('./../data/model_definition')
	json_read = modelReader.read()
	model = model_from_json(json_read)
	model.load_weights('./../data/model_weights')
	
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
		print quesID
		# if quesID not in vqaVal.qqa.keys():
		# 	continue
		question = vqaVal.qqa[quesID]
		# print question
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
		temp_dict['answer'] = answerFeatures[predictions[0]]
		resultsDicts.append(temp_dict)
	writer = open('./../Results/MultipleChoice_mscoco_val2014_second_results.json', 'w')
	json_dump = json.dumps(resultsDicts)
	writer.write(json_dump)
		
def main():
	evalResults()

if __name__ == "__main__":
	main()
