import sys
sys.path.insert(0, './../VQA/PythonHelperTools')
from vqaTools.vqa import VQA

from keras.models import Sequential
from keras.layers.core import Dense, Dropout

import lessdummy1

import json
import numpy as np
import scipy.io as sio
import argparse

import cocoIDToFeatures as pramod

dataDir = './../VQA'
taskType = 'MultipleChoice'
dataType = 'mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType = 'train2014'
annFile = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType)
quesFile = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)

tfile = '../features/coco_vgg_IDMap.txt'

def getMLPModel(input_size, output_size):
    model = Sequential()

    # Two hidden layers
    model.add(Dense(1000, input_dim=input_size, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, input_dim=input_size, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))

    # Output layer for probability
    model.add(Dense(output_size, init='uniform', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def main():
    parser = argparse.ArgumentParser(description="MLP for ML-Project")
    parser.add_argument('--X_train_file')
    parser.add_argument('--Y_train_file')
    parser.add_argument('--model_weights_file')
    parser.add_argument('--model_definition_file')

    args = parser.parse_args()
    FILE_INDEX = 0
    FILE_INDEX_LIMIT = 4
    # model = getMLPModel(len(X_train[0]), len(Y_train[0]))
    # print "Reading GloVe and VGG raw files"
    print '*******  Training on partition ' + str(FILE_INDEX) + ' *********'
    X_train = np.load(open(args.X_train_file+str(FILE_INDEX)+'.npy', 'r'))
    Y_train = np.load(open(args.Y_train_file+str(FILE_INDEX)+'.npy', 'r'))
    FILE_INDEX += 1
    model = getMLPModel(X_train.shape[1], Y_train.shape[1])    
    model.fit(X = X_train, y = Y_train, verbose = True, nb_epoch = 100)
    while FILE_INDEX <= FILE_INDEX_LIMIT:
        X_train = np.load(open(args.X_train_file+str(FILE_INDEX)+'.npy', 'r'))
        Y_train = np.load(open(args.Y_train_file+str(FILE_INDEX)+'.npy', 'r'))
        print '*******  Training on partition ' + str(FILE_INDEX) + ' *********'
        model.fit(X = X_train, y = Y_train, verbose = True, nb_epoch = 100)
        FILE_INDEX += 1
    model.save_weights(args.model_weights_file, overwrite = True)
    json_string = model.to_json()
    if args.model_definition_file != None:
        weightsWriter = open(args.model_definition_file, 'w')
        weightsWriter.write(json_string)
        
if __name__ == "__main__":
    main()
