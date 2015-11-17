#!/usr/bin/python
import itertools
from itertools import combinations
import collections
import inspect
import scipy.io as sio
import itertools
from itertools import combinations
import collections
import inspect
import operator
import sys
import os
import math

Usage : python cocoIDToFeatures.py <cocoID>
cocoID is from 1:123287

tfile = 'coco_vgg_IDMap.txt'
def generateDictionary(tfile):
   my_dict = {}
   file = open(tfile, 'r')
   lines = file.readlines()
   file.close()
   for line in lines:
   		parts = line.split()
   		my_dict[int(parts[0])] = int(parts[1])
   return my_dict

def mapCocoIDToFeatureVector(my_dict,cocoID):
   mat_Dict = sio.loadmat('vgg_feats.mat')
   featureVectors = mat_Dict['feats']
   return featureVectors[:,(cocoID-1)]


   



if __name__ == "__main__":
	if (len(sys.argv) == 2):
		cocoID = int(sys.argv[1])
	my_dict = {}
	my_dict = generateDictionary(tfile)
	featureVector = mapCocoIDToFeatureVector(my_dict,cocoID)
    

