#!/bin/bash

BATCH_SIZE=${1:-64}
DATASET=${2:-CIFAR10}
MODEL=${3:-MOBILE_NET}
LOG_DIR=./experiments/$DATASET/$MODEL

mkdir -p $LOG_DIR

tegrastats --start --logfile $LOG_DIR/$BATCH_SIZE.txt

time python models/train.py --batch_size=$BATCH_SIZE --download

tegrastats --stop
