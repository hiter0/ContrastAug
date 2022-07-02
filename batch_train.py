import os
cmd = 'python train.py --PatchMatch_Aug True --trainset "train_100%_Healthy" --model_name ShuffleNet --epochs 100 --data_path "../data/Corn" --num_classes 3'
os.system(cmd)
cmd = 'python train.py --PatchMatch_Aug True --trainset "train_100%_Healthy" --model_name ResNet --epochs 100 --data_path "../data/Corn" --num_classes 3'
os.system(cmd)
cmd = 'python train.py --PatchMatch_Aug True --trainset "train_100%_Healthy" --model_name MobileNet --epochs 100 --data_path "../data/Corn" --num_classes 3'
os.system(cmd)