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

dataDir = '../../cs446-project/data'
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

    model.compile(loss='mse', optimizer='sgd')

    return model


def main():
    parser = argparse.ArgumentParser(description="MLP for ML-Project")
    parser.add_argument('--preprocessed_file')
    parser.add_argument('--answer_vector_file')
    parser.add_argument('--glove_file', default='./../glove/glove.6B.300d.txt')
    parser.add_argument('--model_weights_file')
    parser.add_argument('--model_definition_file')

    args = parser.parse_args()

    print "Reading GloVE and VGG raw files"

    glove_word_vec_file = args.glove_file
    word_vec_dict = lessdummy1.readGloveData(glove_word_vec_file)

    imageDict = pramod.generateDictionary(tfile)
    feats = sio.loadmat('./../features/coco/vgg_feats.mat')['feats']

    print "Reading the data and creating features"

    preprocessed_file = open(args.preprocessed_file, 'r')
    data = json.loads(preprocessed_file.read())
    answer_vector_file = open(args.answer_vector_file, 'r')
    answerFeatureVector = json.loads(answer_vector_file.read())

    preprocessed_file.close()
    answer_vector_file.close()

    X_train = []
    Y_train = []

    for ques in data:
        image_id = ques['image_id']
        image_vector = np.asarray(feats[:, imageDict[image_id]])
        question_vector = lessdummy1.getBOWVector(ques['question'], word_vec_dict)
        answer_vector = lessdummy1.getAnswerVector(ques['answer'], answerFeatureVector)

        X_train.append(np.append(image_vector, question_vector))
        Y_train.append(answer_vector)

    word_vec_dict = None
    imageDict = None
    feats = None
    answerFeatureVector = None
    data = None
    
    print "Creating the MLP model"
    model = getMLPModel(len(X_train[0]), len(Y_train[0]))

    # Train for 10 iterations
    model.fit(X=X_train, y=Y_train, verbose=True, nb_epoch=10)
    model.save_weights(args.model_weights_file, overwrite=True)

    json_string = model.to_json()
    if args.model_definition_file != None:
        file = open(args.model_definition_file, 'w')
        file.write(json_string)

if __name__ == "__main__":
    main()
