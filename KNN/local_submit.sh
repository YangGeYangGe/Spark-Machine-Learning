#!/bin/bash

spark-submit \
    --master yarn \
    --deploy-mode client \
    --executor-memory 2G \
    --num-executors $4 \
    --executor-cores $5\
    stage1knn.py --expname $1 --pcad $2 --knnd $3 --partition $6
