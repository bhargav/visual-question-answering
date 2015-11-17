import sys

sys.path.insert(0, './../VQA/PythonHelperTools')
from vqaTools.vqa import VQA

from keras.models import Sequential
from keras.layers.core import Dense, Dropout

import numpy as np
import scipy.io as sio

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

    model.compile(loss='mean_square_error', optimizer='sgd')

    return model


def main():
    glove_word_vec_file = './../glove/glove.6B.50d.txt'
    word_vec_dict = lessdummy1.readGloveData(glove_word_vec_file)
    image_mapping_dict = cocoIDToFeatures.generateDictionary(tfile)
    matlab_images = sio.loadmat('../features/coco/vgg_feats.mat')['feats']
    vqaTrain = VQA(annFile, quesFile)

    anns = vqaTrain.dataset['annotations']
    questions = vqaTrain.questions['questions']

    for x in range(1):
        ann = anns[0]
        imageId = ann['image_id']
        print ann

        imagearray = np.asarray(matlab_images[:, image_mapping_dict[imageId]])
        print imagearray.shape





    # mlp_model = getMLPModel()


if __name__ == "__main__":
    main()
