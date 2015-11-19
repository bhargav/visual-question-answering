import numpy as np
import json
## Bhargav
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout
##
X_TEST_FILE_NAME = 'X_test'
Y_TEST_FILE_NAME = 'Y_test'

def evalResults():
	modelReader = open('./../data/model_definition')
	json_read = modelReader.read()
	model = model_from_json(json_read)
	model.load_weights('./../data/model_weights')
	total = 0.0
	correct = 0.0
	for index in range(0, 3):
		x_test = np.load(open('./../data/X_test'+str(index)+'.npy', 'r'))
		y_test = np.load(open('./../data/Y_test'+str(index)+'.npy', 'r'))
	# print model.predict(x_test[0])
		predictions = model.predict_classes(x_test)
		for i in range(0, len(predictions)):
			# print predictions[i]
			if sum(y_test[i] > 0):
				total += 1
			if (y_test[i][predictions[i]] == 1):
				correct += 1
			# if 
	print correct/total
	# print model.predict_proba(x_test[0:1])
    
def main():
	evalResults()

if __name__ == "__main__":
	main()