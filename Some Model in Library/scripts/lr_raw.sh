spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors $1 \
    --executor-cores $2\
    raw_lr.py
