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
annFile     ='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile    ='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)
resultType  ='second'
fileTypes   = ['results'] 
[resFile] = ['%s/Results/%s_%s_%s_%s_%s.json'%(dataDir, taskType, dataType, dataSubType, \
resultType, fileType) for fileType in fileTypes]  


# annReader = open(annFile)
# annotations = json.load(annReader)
# quesReader = open(quesFile)
# questions = json.load(quesReader)
resReader = open(resFile)
results = json.load(resReader)

correct = 0.0
total = 0.0
vqa = VQA(annFile, quesFile)
for quesDict in results:
	total += 1
	print vqa.qa[quesDict['question_id']]['image_id']
	print vqa.qqa[quesDict['question_id']]['question']
	print 'predicted: ' + str(quesDict['answer'])
	print 'correct: ' + str(vqa.qa[quesDict['question_id']]['multiple_choice_answer'])
	print '\n'
	if quesDict['answer'] == vqa.qa[quesDict['question_id']]['multiple_choice_answer']:
		correct += 1
print correct
print total
print correct/total

# print questions
# print annotations
