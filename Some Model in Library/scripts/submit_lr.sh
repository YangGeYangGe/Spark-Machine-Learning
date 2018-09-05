spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors $1 \
    --executor-cores $2\
    lr.py \
    --dim $3
