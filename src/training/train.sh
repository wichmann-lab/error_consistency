#!/bin/bash

cd /home/bethge/tklein16/ec2/error_consistency/src/training

python3.10 train_imagenet.py -wu thoklei --no_tqdm --seed 13
