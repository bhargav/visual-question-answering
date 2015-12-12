import json
import numpy as np
import argparse

import lessdummy1 as utilities
import cocoIDToFeatures as cocoImageUtils

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Reshape, Merge, RepeatVector
from keras.layers.recurrent import LSTM

tfile = '../features/coco_vgg_IDMap.txt'

class LSTMSingleAnswerModel(object):

    def __init__(self):
        self.nb_timestep = 23

    def getModel(self, model_filename, model_weights = None):
        jsonString = json.loads(model_filename)
        model = model_from_json(json_string=jsonString)

        if model_weights != None:
            model.load_weights(model_weights)

        return model


    def getModel(self, image_size, question_vector_size, answer_vector_size = 1000, hidden_layer_size = 1000, lstm_layer_size = 1000):
        model = Sequential()

        imageModel = Sequential()
        imageModel.add(Dense(question_vector_size, input_shape=(image_size,)))
        imageModel.add(Dropout(0.2))
        imageModel.add(RepeatVector(self.nb_timestep))

        questionModel = Sequential()
        questionModel.add(Reshape(input_shape=(self.nb_timestep, question_vector_size,), dims=(self.nb_timestep, question_vector_size,)))

        # Concatinate Image and Question Models
        model.add(Merge([imageModel, questionModel], mode='concat', concat_axis=2))

        model.add(LSTM(lstm_layer_size, return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(hidden_layer_size, init='uniform', activation='tanh'))
        model.add(Dropout(0.2))

        model.add(Dense(answer_vector_size, init='uniform', activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        return model

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

        return (X_train, Image_train, Y_train)
