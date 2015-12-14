import numpy as np
import matplotlib.pyplot as plt
# import skimage.io as io
import sys
import json
import random
import os
import pickle
import lessdummy1 as ld
import os.path
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout
import cocoIDToFeatures as pramod
import scipy.io as sio

tfile = './../features/coco_vgg_IDMap.txt'

sys.path.insert(0, './../VQA/PythonHelperTools')
sys.path.insert(0, './../VQA/Images/train2014')
# from vqa import VQA
from vqaTools.vqa import VQA
# from vqaEvaluation.vqaEval import VQAEval


# # set up file names and paths
# taskType    ='MultipleChoice'
# dataType    ='mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
# dataSubType ='val2014'
# annFile     ='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
# quesFile    ='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
# imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)
# resultType  ='second'
# fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 
# vqaVal = VQA(annFile, quesFile)

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

sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
sys.path.insert(0, '%s/PythonEvaluationTools' %(dataDir))

dataDir = './../VQA'
taskType2 = 'MultipleChoice'
dataType2 = 'mscoco' # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType2 = 'val2014'
annFile2 = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType2)
quesFile2 = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType2)
imgDir2 = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType2)

modelReader = open('./model_definition_100iter.json')
json_read = modelReader.read()
model = model_from_json(json_read)
model.load_weights('./model_weights_100iter.h5py')

vqaVal = VQA(annFile2, quesFile2)


newdataSubType = 'analysis1'
outputQuestionFile = '%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, newdataSubType)
outputAnnotationFile = '%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, newdataSubType)
# vqaAnalysis = vqaVal
newQuestion = 'yes'
questionIndex = 0
ids = vqaVal.getQuesIds()
anns = vqaVal.loadQA(ids)


if not os.path.exists(outputAnnotationFile) or os.stat(outputAnnotationFile).st_size == 0:
	outputQuestionWriter = open(outputQuestionFile, 'w')
	outputAnnotationWriter = open(outputAnnotationFile, 'w')

	outputQuestions = {}
	outputAnnotations = {}


	
	
	outputAnnotations['info'] = {}
	outputAnnotations['info']['description'] = 'This is the dataset created for further analysis of the VQA task.'
	outputAnnotations['info']['url'] = ' '
	outputAnnotations['info']['version'] = '1.0'
	outputAnnotations['info']['year'] = 2015
	outputAnnotations['info']['contributor'] = 'vishaal'
	outputAnnotations['info']['date_created'] = '2015-12-11'
	outputAnnotations['data_type'] = dataType
	outputAnnotations['license'] = {}
	outputAnnotations['data_subtype'] = 'analysis1'
	outputAnnotations['annotations'] = []

	outputQuestions['info'] = {}
	outputQuestions['info']['description'] = 'This is the dataset created for further analysis of the VQA task.'
	outputQuestions['info']['url'] = ' '
	outputQuestions['info']['version'] = '1.0'
	outputQuestions['info']['year'] = 2015
	outputQuestions['info']['contributor'] = 'vishaal'
	outputQuestions['info']['date_created'] = '2015-12-11'
	outputQuestions['task_type'] = taskType
	outputQuestions['data_type'] = dataType
	outputQuestions['license'] = {}
	outputQuestions['data_subtype'] = 'analysis1'
	outputQuestions['questions'] = []
else:
	outputQuestionWriter = open(outputQuestionFile)
	outputAnnotationWriter = open(outputAnnotationFile)
	# question_json_reader = outputQuestionWriter.read()
	outputQuestions = json.load(outputQuestionWriter)
	# annotations_json_reader = outputAnnotationWriter.read()
	outputAnnotations = json.load(outputAnnotationWriter)


glove_word_vec_file = './../glove/glove.6B.300d.txt'
word_vec_dict = ld.readGloveData(glove_word_vec_file)
imageDict = pramod.generateDictionary(tfile)
feats = sio.loadmat('./../features/coco/vgg_feats.mat')['feats']

while newQuestion != 'no':
	print '\n'
	randomAnn = random.choice(anns)
	origquestion = vqaVal.qqa[randomAnn['question_id']]
	questionVector = ld.getBOWVector(origquestion['question'].strip().replace('?', ' ?').split(), word_vec_dict) 
	imgID = randomAnn['image_id']
	imageVector = np.asarray(feats[:, imageDict[imgID]])
	# temp_dict = {}
	ansString = randomAnn['multiple_choice_answer']
	# temp_dict['question_id'] = quesID
	# answerVector = ld.getAnswerVector(ansString, answerFeatures)
	temp_x_test = np.append(imageVector, questionVector)
	# temp_y_test = answerVector
	x_test = np.asarray([temp_x_test])
	# y_test = np.asarray([temp_y_test])
	predictions = model.predict_classes(x_test, verbose = False)
	predictedAnswer = answerFeatures[predictions[0]]
	print 'Predicted Answer: ' + predictedAnswer
	# temp_dict['answer'] = answerFeatures[predictions[0]]
	if predictedAnswer == str(randomAnn['multiple_choice_answer']):

		print 'Image ID: ' + str(randomAnn['image_id'])
		print 'Question: ' + str(vqaVal.qqa[randomAnn['question_id']]['question'])
		print 'Answer: ' + str(randomAnn['multiple_choice_answer'])
		quesNos = raw_input('How many questions do you want?:\n')
		# if isinstance(,int) == False:
		# 	quesNos = 0
		for i in xrange(int(quesNos)):
			annotation = {}
			question = {}
			questionString = raw_input('Enter question: ')
			temp_ques = questionString.split()
			answer = raw_input('Enter answer: ')
			
			annotation['question_type'] = temp_ques[0] + ' ' + temp_ques[1] + ' ' + temp_ques[2]
			annotation['multiple_choice_answer'] = answer
			subanswer = {}
			subanswer['answer'] = answer
			subanswer['answer_confidence'] = 'yes'
			subanswer['answer_id'] = 1
			annotation['answers'] = [subanswer]
			annotation['image_id'] = randomAnn['image_id']
			annotation['question_id'] = questionIndex
			annotation['answer_type'] = 'other'
			outputAnnotations['annotations'].append(annotation)

			question['image_id'] = randomAnn['image_id']
			question['question'] = questionString
			question['multiple_choices'] = []
			question['question_id'] = questionIndex
			outputQuestions['questions'].append(question)
			questionIndex += 1
		newQuestion = raw_input('Do you want to add one more question to analysis? (no to stop): ')
	# print randomAnn
dfile = open('questionAnnotationDump.txt','w')
pickle.dump(outputAnnotations,dfile)
pickle.dump(outputQuestions,dfile)
dfile.close
annotation_json_dump = json.dumps(outputAnnotations)
question_json_dump = json.dumps(outputQuestions)
outputQuestionWriter.write(question_json_dump)
outputAnnotationWriter.write(annotation_json_dump)


