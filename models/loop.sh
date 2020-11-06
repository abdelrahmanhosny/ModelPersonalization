#!/bin/bash

DATASET=CIFAR10
for BATCH_SIZE in 2 4 8 16 32 64 128 256 
do
    for MODEL in mobilenet_v2 googlenet resnet18 resnet50
    do
        sleep 3
        LOG_DIR=./experiments/$DATASET/$MODEL
        mkdir -p $LOG_DIR

        tegrastats --start --logfile $LOG_DIR/$BATCH_SIZE.txt

        time python models/train.py --batch_size=$BATCH_SIZE --output_dir=$LOG_DIR --model $MODEL

        tegrastats --stop
        sleep 3
    done
done

for BATCH_SIZE in 2 4 8 16 32 64 128 256 
do
    for MODEL in mobilenet_v2 googlenet resnet18 resnet50
    do
        sleep 3
        LOG_DIR=./experiments/$DATASET/$MODEL/NO_CUDA
        mkdir -p $LOG_DIR

        tegrastats --start --logfile $LOG_DIR/$BATCH_SIZE.txt

        time python models/train.py --batch_size=$BATCH_SIZE --output_dir=$LOG_DIR --model $MODEL --no-cuda

        tegrastats --stop
        sleep 3
    done
done

for BATCH_SIZE in 2 4 8 16 32 64 128 256 
do
    for MODEL in mobilenet_v2 googlenet resnet18 resnet50
    do
        sleep 3
        LOG_DIR=./experiments/$DATASET/$MODEL/FREEZE
        mkdir -p $LOG_DIR

        tegrastats --start --logfile $LOG_DIR/$BATCH_SIZE.txt

        time python models/train.py --batch_size=$BATCH_SIZE --output_dir=$LOG_DIR --model $MODEL --freeze

        tegrastats --stop
        sleep 3
    done
done

for BATCH_SIZE in 2 4 8 16 32 64 128 256 
do
    for MODEL in mobilenet_v2 googlenet resnet18 resnet50
    do
        sleep 3
        LOG_DIR=./experiments/$DATASET/$MODEL/FREEZE_NO_CUDA
        mkdir -p $LOG_DIR

        tegrastats --start --logfile $LOG_DIR/$BATCH_SIZE.txt

        time python models/train.py --batch_size=$BATCH_SIZE --output_dir=$LOG_DIR --model $MODEL --freeze --no-cuda

        tegrastats --stop
        sleep 3
    done
done