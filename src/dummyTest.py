import sys
sys.path.insert(0, './../PythonHelperTools')
from vqaTools.vqa import VQA


dataDir='../../VQA'
taskType='OpenEnded'
dataType='mscoco' # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType='train2014'
annFile='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)


def main():
	pass
	vqaTrain = VQA(annFile, quesFile)
	print vqaTrain.qa
	

if __name__ == "__main__":
	main()
