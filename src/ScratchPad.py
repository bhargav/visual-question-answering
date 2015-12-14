
# coding: utf-8

# In[1]:

import json
import numpy as np
import scipy.io as sio
import argparse

import lessdummy1 as utilities
import cocoIDToFeatures as cocoImageUtils

tfile = '../features/coco_vgg_IDMap.txt'

args = {}
args['answer_vector_file']='answer_feature_list.json'
args['glove_file']='../glove/glove.6B.300d.txt'


# In[2]:

print "Reading GloVE and VGG raw files"

glove_word_vec_file = args['glove_file']
word_vec_dict = utilities.readGloveData(glove_word_vec_file)

imageDict = cocoImageUtils.generateDictionary(tfile)
feats = sio.loadmat('./../features/coco/vgg_feats.mat')['feats']

print "Reading the data and creating features"

answer_vector_file = open(args['answer_vector_file'], 'r')
answerFeatureVector = json.loads(answer_vector_file.read())

answer_vector_file.close()


# In[3]:

import sys
sys.path.insert(0, './../VQA/PythonHelperTools')
from vqaTools.vqa import VQA

dataDir = './../VQA'
taskType = 'MultipleChoice'
dataType = 'mscoco' # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType = 'train2014'
annFile = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType)
quesFile = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)
vqaTrain = VQA(annFile, quesFile)
dummyano = vqaTrain.dataset['annotations']
answerFeatures = utilities.createAnswerFeatures(dummyano)

vqaVal = VQA(annFile, quesFile)


# In[4]:

dataset = []

for quesID, annotation in vqaVal.qa.iteritems():
    question = vqaVal.qqa[quesID]
    question_text = question['question'].strip().replace('?', ' ?').split()
    imgID = annotation['image_id']
    ansString = annotation['multiple_choice_answer']
    
    dataset.append({'question': question_text, 'answer': ansString, 'image': imgID})


# In[5]:

from collections import Counter

c = Counter([len(x['question']) for x in dataset])
maxlen = max(c.keys())
print c
print "Max Question Length = ", maxlen


# In[6]:

maxlen = 25
nb_train = len(dataset)
nb_timestep = maxlen + 1 # For Image Vector
word_vec_dim = len(word_vec_dict['hi'])
image_dim = 4096


# ### Building the LSTM Model###

# **Create the LSTM model**

# In[7]:

from lstm_single_answer import LSTMSingleAnswerModel

lstm_model = LSTMSingleAnswerModel()
model = lstm_model.getModel(4096, 300, 1000)


# **Generating X_train and Y_train**

# In[8]:

def transformToModelInput(self, dataset, answerFeatureVector, word_vec_dict):
        nb_train = len(dataset)
        input_size = 300
        X_train = np.zeros(shape=(nb_train, self.nb_timestep, input_size))
        Image_train = np.zeros(shape=(nb_train, 4096))
        Y_train = np.zeros(shape=(nb_train, len(answerFeatureVector)))

        maxlen = self.nb_timestep

        idx = 0
        for input_item in dataset:
            q = input_item['question']
            padding = maxlen - len(q)
            for i in xrange(padding):
                X_train[idx, i, :] = np.zeros(input_size)

            for word in q:
                X_train[idx, padding, :] = utilities.getWordVector(word, word_vec_dict)
            Y_train[idx, :] = utilities.getAnswerVector(input_item['answer'], answerFeatureVector)

            Image_train[idx, :] = np.asarray(feats[:, imageDict[input_item['image']]])

            idx += 1

        return ([Image_train, X_train], Y_train)


# In[ ]:

(X_train, Y_train) = transformToModelInput(lstm_model, dataset, answerFeatureVector, word_vec_dict)

model.fit(X_train, Y_train, nb_epoch=5, validation_split=0.1, show_accuracy=True, verbose=1)


# In[ ]:




# In[ ]:



