#!/bin/bash

WRITER_ID=${1:-0}
BATCH_SIZE=${2:-64}

tegrastats --start --logfile ./data/QMNIST/models-subject-out/$WRITER_ID/personalize-nano-gpu-$BATCH_SIZE.txt

time python mnist/2_personalize.py --log-interval=1 --writer_id=$WRITER_ID --batch_size=$BATCH_SIZE

tegrastats --stop
