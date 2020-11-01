#!/bin/bash

WRITER_ID=${1:-0}
BATCH_SIZE=${2:-64}

nvidia-smi --query-gpu=timestamp,pstate,utilization.gpu,utilization.memory,temperature.gpu,temperature.memory,power.draw --format=csv -l 1 -f data/QMNIST/models-subject-out/$WRITER_ID/personalize-server-gpu-$BATCH_SIZE.csv &
pid=$!

sleep 5        # used to report idle power when util is zero

time python mnist/2_personalize.py --log-interval=1 --writer_id=$WRITER_ID --batch_size=$BATCH_SIZE |& tee data/QMNIST/models-subject-out/$WRITER_ID/personalize-server-gpu-$BATCH_SIZE.log

sleep 5        # used to report idle power when util is zero

kill $pid
