#!/bin/bash
script -c "python /opt/caffe2/caffe2/python/examples/alexnet_trainer.py --train_data /data/20171110-004454-7443/train_db --db_type lmdb --gpus 0 --batch_size 128 --epoch_size 1281167 --num_epochs 1 --base_learning_rate 0.001" /data/caffe2/alexnet_1-1.txt
sleep 1m
script -c "python /opt/caffe2/caffe2/python/examples/alexnet_trainer.py --train_data /data/20171110-004454-7443/train_db --db_type lmdb --gpus 0,1 --batch_size 256 --epoch_size 1281167 --num_epochs 1 --base_learning_rate 0.001" /data/caffe2/alexnet_2-1.txt
sleep 1m
script -c "python /opt/caffe2/caffe2/python/examples/alexnet_trainer.py --train_data /data/20171110-004454-7443/train_db --db_type lmdb --gpus 0,1,2,3 --batch_size 512 --epoch_size 1281167 --num_epochs 1 --base_learning_rate 0.001" /data/caffe2/alexnet_4-1.txt
sleep 1m
script -c "python /opt/caffe2/caffe2/python/examples/alexnet_trainer.py --train_data /data/20171110-004454-7443/train_db --db_type lmdb --gpus 0,1,2,3,4,5 --batch_size 768 --epoch_size 1281167 --num_epochs 1 --base_learning_rate 0.001" /data/caffe2/alexnet_6-1.txt
sleep 1m
script -c "python /opt/caffe2/caffe2/python/examples/alexnet_trainer.py --train_data /data/20171110-004454-7443/train_db --db_type lmdb --gpus 0,1,2,3,4,5,6,7 --batch_size 1024 --epoch_size 1281167 --num_epochs 1 --base_learning_rate 0.001" /data/caffe2/alexnet_8-1.txt
sleep 1m


script -c "python /opt/caffe2/caffe2/python/examples/resnet50_trainer.py --train_data /data/20171110-004454-7443/train_db --db_type lmdb --gpus 0 --batch_size 16 --epoch_size 1281167 --num_epochs 1 --base_learning_rate 0.001" /data/caffe2/resnet50_1-1.txt
sleep 1m
script -c "python /opt/caffe2/caffe2/python/examples/resnet50_trainer.py --train_data /data/20171110-004454-7443/train_db --db_type lmdb --gpus 0,1 --batch_size 32 --epoch_size 1281167 --num_epochs 1 --base_learning_rate 0.001" /data/caffe2/resnet50_2-1.txt
sleep 1m
script -c "python /opt/caffe2/caffe2/python/examples/resnet50_trainer.py --train_data /data/20171110-004454-7443/train_db --db_type lmdb --gpus 0,1,2,3 --batch_size 64 --epoch_size 1281167 --num_epochs 1 --base_learning_rate 0.001" /data/caffe2/resnet50_4-1.txt
sleep 1m
script -c "python /opt/caffe2/caffe2/python/examples/resnet50_trainer.py --train_data /data/20171110-004454-7443/train_db --db_type lmdb --gpus 0,1,2,3,4,5 --batch_size 96 --epoch_size 1281167 --num_epochs 1 --base_learning_rate 0.001" /data/caffe2/resnet50_6-1.txt
sleep 1m
script -c "python /opt/caffe2/caffe2/python/examples/resnet50_trainer.py --train_data /data/20171110-004454-7443/train_db --db_type lmdb --gpus 0,1,2,3,4,5,6,7 --batch_size 128 --epoch_size 1281167 --num_epochs 1 --base_learning_rate 0.001" /data/caffe2/resnet50_8-1.txt
sleep 1m


