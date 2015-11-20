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
	dataDir = '../../cs446-project/data'
	taskType = 'MultipleChoice'
	dataType = 'mscoco' # 'mscoco' for real and 'abstract_v002' for abstract
	dataSubType = 'train2014'
	annFile = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType)
	quesFile = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType)
	imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)
	vqaTrain = VQA(annFile, quesFile)
	annotations = vqaTrain.dataset['annotations']
	answerFeatures = ld.createAnswerFeatures(annotations)




	# questionTypeCorrect  

	dataDir2 = '../../cs446-project/data'
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
	annotations = vqaVal.dataset['annotations']
	questions = vqaVal.questions['questions']
	FILE_INDEX = 0
    
	total = 0.0
	correct = 0.0
	
	
	


	resultsDicts = []
	questionTypeResult = {}
	answerTypeResult = {}

	x_test = []
	y_test = []
	glove_word_vec_file = './../glove/glove.6B.300d.txt'
	word_vec_dict = ld.readGloveData(glove_word_vec_file)
	imageDict = pramod.generateDictionary(tfile)
	feats = sio.loadmat('./../features/coco/vgg_feats.mat')['feats']

	for question in questions:
		questionVector = ld.getBOWVector(question['question'].strip().replace('?', ' ?').split(), word_vec_dict)
		imgID = question['image_id']
		imageVector = np.asarray(feats[:,imageDict[imgID]])
        # quesItem['image_id'] = imgID
        # quesItem['question'] = question['question'].replace('?', ' ?').split(' ')
		annotations = vqaVal.loadQA(ids = [question['question_id']])
		for annotation in annotations:
			temp_dict = {}
			ansString = annotation['multiple_choice_answer']
			temp_dict['actAns'] = ansString
			temp_dict['ques'] = question['question']
			temp_dict['imgID'] = question['image_id']
			answerVector = ld.getAnswerVector(ansString, answerFeatures)
			temp_x_test = np.append(imageVector, questionVector)
			temp_y_test = answerVector   
			x_test = np.asarray([temp_x_test])
			y_test = np.asarray([temp_y_test]) 	
			predictions = model.predict_classes(x_test, verbose = False)
			temp_dict['predAns'] = answerFeatures[predictions[0]]

			if annotation['answer_type'] in answerTypeResult:
				answerTypeResult[annotation['answer_type']][1] = answerTypeResult[annotation['answer_type']][1] + 1
			else:
				answerTypeResult[annotation['answer_type']] = [0, 1]

			if annotation['question_type'] in questionTypeResult:
				questionTypeResult[annotation['question_type']][1] = questionTypeResult[annotation['question_type']][1] + 1
			else:
				questionTypeResult[annotation['question_type']] = [0, 1]
			for i in range(0, len(predictions)):
				if sum(y_test[i] > 0):
					total += 1
				if (y_test[i][predictions[i]] == 1):
					correct += 1
					questionTypeResult[annotation['question_type']][0] = questionTypeResult[annotation['question_type']][0] + 1
					answerTypeResult[annotation['answer_type']][0] = answerTypeResult[annotation['answer_type']][0] + 1
					temp_dict['correct'] = True
				else:
					temp_dict['correct'] = False
			# if 
			resultsDicts.append(temp_dict)
	print correct/total
	# print model.predict_proba(x_test[0:1])
	for key, value in questionTypeResult.iteritems():
		questionTypeResult[key].append(value[0] / value[1])
		print key + ' ' + str(value[0]) + ' ' + str(value[1]) + ' ' + str(value[0]/value[1])
	for key, value in answerTypeResult.iteritems():
		answerTypeResult[key].append(value[0] / value[1])
		print key + ' ' + str(value[0]) + ' ' + str(value[1]) + ' ' + str(value[0]/value[1])	
	writer = open('./../data/results.txt', 'w')
	writer2 = open('./../data/questionTypeResult.txt', 'w')
	writer3 = open('./../data/answerTypeResult.txt', 'w')
	json_dump = json.dumps(resultsDicts)
	json_dump2 = json.dumps(questionTypeResult)
	json_dump3 = json.dumps(answerTypeResult)
	writer.write(json_dump)
	writer2.write(json_dump2)
	writer3.write(json_dump3)
	
def main():
	evalResults()

if __name__ == "__main__":
	main()