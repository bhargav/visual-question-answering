wget -c http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip
wget -c http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip

unzip Annotations_Train_mscoco.zip -d Annotations/
unzip Annotations_Val_mscoco.zip -d Annotations/

wget -c http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip
wget -c http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip

unzip Questions_Train_mscoco.zip -d Questions/
unzip Questions_Val_mscoco.zip -d Questions/
