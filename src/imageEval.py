import json
import sys
import os
dataDir = './../VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
sys.path.insert(0, '%s/PythonEvaluationTools' %(dataDir))
from vqa import VQA

taskType    ='MultipleChoice'
dataType    ='mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType ='analysis1'
overallCorrect = 0
overallLen = 0
for number in ['10','20','30','40','50','70','90','100']:
	annFile     ='%s/Annotations/%s_%s_annotations%s.json'%(dataDir, dataType, dataSubType,number)
	quesFile    ='%s/Questions/%s_%s_%s_questions%s.json'%(dataDir, taskType, dataType, dataSubType,number)
	imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)
	resultType  ='second'
	fileTypes   = ['results'] 
	[resFile] = ['%s/Results/%s_%s_%s_%s_%s_%s.json'%(dataDir, taskType, dataType, dataSubType, \
	resultType, fileType,number) for fileType in fileTypes]  

	# MultipleChoice_mscoco_analysis1_second_results_30

	imageIDMap = {}
	resultIDMap = {}
	print resFile
	# annReader = open(annFile)
	# annotations = json.load(annReader)
	# quesReader = open(quesFile)
	# questions = json.load(quesReader)
	resReader = open(resFile)
	results = json.load(resReader)
	prevImageID = ""
	correct = 0.0
	total = 0.0
	vqa = VQA(annFile, quesFile)
	for quesDict in results:
		total += 1
		id = int(vqa.qa[quesDict['question_id']]['image_id'])
		imageIDMap[id] = 0
		resultIDMap[id] = 0

	for quesDict in results:
		id = int(vqa.qa[quesDict['question_id']]['image_id'])
		# print id
		imageIDMap[id] = imageIDMap[id] + 1

	dictLen = len(imageIDMap)

	for quesDict in results:
		id = int(vqa.qa[quesDict['question_id']]['image_id'])
		# print id
		# print vqa.qa[quesDict['question_id']]['image_id']
		# print vqa.qqa[quesDict['question_id']]['question']
		# print 'predicted: ' + str(quesDict['answer'])
		# print 'correct: ' + str(vqa.qa[quesDict['question_id']]['multiple_choice_answer'])
		# print '\n'
		if quesDict['answer'] == vqa.qa[quesDict['question_id']]['multiple_choice_answer']:
			resultIDMap[id] = resultIDMap[id] + 1
		if imageIDMap[id] == resultIDMap[id]:
			correct += 1

	for k,v in imageIDMap.items():
		print k,v
	print correct
	print dictLen
	print correct/dictLen
	overallCorrect = overallCorrect + correct
	overallLen = overallLen + dictLen

print 'Number of images completely understood',int(overallCorrect)
print 'Number of overall images',overallLen
