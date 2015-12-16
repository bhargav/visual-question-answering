
# coding: utf-8

# In[1]:

import json
import numpy as np
import scipy.io as sio
import argparse

import lessdummy1 as utilities
import cocoIDToFeatures as cocoImageUtils

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Reshape, Merge, RepeatVector
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, History

history = History()
checkpoint = ModelCheckpoint(filepath="SecondTry/weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=False)

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

nb_train = len(dataset)
nb_timestep = 23 # For Image Vector
word_vec_dim = len(word_vec_dict['hi'])
image_dim = 4096

# ### Building the LSTM Model###

# **Create the LSTM model**

# In[7]:

def getModel(image_size, question_vector_size, answer_vector_size = 1000, hidden_layer_size = 1000, lstm_layer_size = 1000):
    nb_timestep = 23
    imageModel = Sequential()
    imageModel.add(Reshape(input_shape=(image_size,), dims=(image_size, )))

    questionModel = Sequential()
    questionModel.add(LSTM(lstm_layer_size, input_shape=(nb_timestep, question_vector_size), return_sequences=True))
    questionModel.add(Dropout(0.2))
    questionModel.add(LSTM(lstm_layer_size, return_sequences=False))
    questionModel.add(Dropout(0.2))

    # Concatinate Image and Question Models
    model = Sequential()
    model.add(Merge([imageModel, questionModel], mode='concat'))

    model.add(Dense(hidden_layer_size, init='uniform', activation='tanh'))
    model.add(Dropout(0.2))

    model.add(Dense(answer_vector_size, init='uniform', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

print "Building LSTM model"

try:
    lstm_model = getModel(4096, 300, 1000)
    #lstm_model.load_weights('weights.hdf5')
except:
    print "Failed to load model weights"
    lstm_model = getModel(4096, 300, 1000)

# **Generating X_train and Y_train**

# In[8]:

def transformToModelInput(dataset, answerFeatureVector, word_vec_dict):
    nb_train = len(dataset)
    input_size = 300
    X_train = np.zeros(shape=(nb_train, nb_timestep, input_size))
    Image_train = np.zeros(shape=(nb_train, 4096))
    Y_train = np.zeros(shape=(nb_train, len(answerFeatureVector)))

    maxlen = nb_timestep

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

print "Extracting Features from Input"

(X_train, Y_train) = transformToModelInput(dataset, answerFeatureVector, word_vec_dict)

print "Training LSTM"

lstm_model.fit(X_train, Y_train, nb_epoch=100, validation_split=0.1, callbacks=[checkpoint, history])

print "Saving weights"

model.save_weights('final.hdf5')
