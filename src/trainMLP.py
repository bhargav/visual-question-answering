import sys

sys.path.insert(0, './../VQA/PythonHelperTools')
from vqaTools.vqa import VQA

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.models import model_from_config

import numpy as np
import scipy.io as sio

import argparse

import lessdummy1
import cocoIDToFeatures

dataDir='../../cs446-project/data'
taskType='MultipleChoice'
dataType='mscoco' # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType='train2014'
annFile='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

tfile = '../features/coco_vgg_IDMap.txt'

def getMLPModel(input_size, output_size):
    model = Sequential()

    # Two hidden layers
    model.add(Dense(1000, input_dim = input_size, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, input_dim = input_size, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))

    # Output layer for probability
    model.add(Dense(output_size, init='uniform', activation='softmax'))

    model.compile(loss='mse', optimizer='sgd')

    return model

def main():
    parser = argparse.ArgumentParser(description="MLP for ML-Project")
    parser.add_argument('--train_data_file')
    parser.add_argument('--train_label_file')
    parser.add_argument('--model_weights_file')
    parser.add_argument('--model_definition_file')

    args = parser.parse_args()

    data = np.load(open(args.train_data_file, 'r'))
    label = np.load(open(args.train_label_file, 'r'))

    model = getMLPModel(data.shape[1], label.shape[1])

    # Train for 10 iterations
    model.fit(X=data, y=label, verbose=True, nb_epoch=10)

    model.save_weights(args.model_weights_file, overwrite=True)

    json_string = model.to_json()
    if args.model_definition_file != None:
        file = open(args.model_definition_file, 'w')
        file.write(json_string)

if __name__ == "__main__":
    main()
