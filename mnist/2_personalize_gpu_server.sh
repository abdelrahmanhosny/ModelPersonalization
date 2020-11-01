#!/bin/sh

WRITER_ID=${1:-0}

nvidia-smi --query-gpu=timestamp,pstate,utilization.gpu,utilization.memory,temperature.gpu,temperature.memory,power.draw \
        --format=csv -l 1 -f data/QMNIST/models-subject-out/$WRITER_ID/personalize-server-gpu.csv &
pid=$!

python 2_personalize.py --writer_id=$WRITER_ID |& data/QMNIST/models-subject-out/$WRITER_ID/personalize-server-gpu.log

kill $pid
