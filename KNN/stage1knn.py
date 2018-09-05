# Cloud Computing Assignment 2 Stage 1 code
# arguments:
#   --expname       the name to show in Spark Web UI
#   --pcad          pca dimension
#   --knnd          knn dimension
#   --executornum
#   --executorcore

# The content of submit script file:
#!/bin/bash

#spark-submit \
#    --master yarn \
#    --deploy-mode client \
#    --num-executors $4 \
#    --executor-cores $5\
#    stage1knn.py --expname $1 --pcad $2 --knnd $3 --partition $6

import argparse
import numpy as np

from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler

def num(record):

    lab, vec = record
    return (np.array(vec) ,lab)

def knn(record):
    datatest, test_label = record

    distances1 = (trainfeatures.value-datatest)**2
    distSquareMatSums = distances1.sum(axis = 1)

    sorteIndices = np.argsort(distSquareMatSums)
    indices = sorteIndices[:nKNNDimension]


    labels = {}
    for i in indices:
        if trainlabels.value[i] not in labels:
            labels[trainlabels.value[i]] = 0

        labels[trainlabels.value[i]] += 1
    key_max = max(labels.keys(), key=(lambda o: labels[o]))

    #(predict label, correct label)
    return (key_max,test_label)

parser = argparse.ArgumentParser()
parser.add_argument("--expname", help="the experiment name", default='knndefault')
parser.add_argument("--pcad", help="the dimension in PCA", default=3)
parser.add_argument("--knnd", help="the distance in KNN", default=3)
parser.add_argument("--partition", help="the number of partitions", default=3)

args = parser.parse_args()
strExpName = args.expname
nPCADimension = int(args.pcad)
nKNNDimension = int(args.knnd)
nPartition = int(args.partition)

spark = SparkSession.builder\
.appName(strExpName)\
.getOrCreate()

train_data_label = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"
test_data_label = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"

train_dl = spark.read.csv(train_data_label,header=False,inferSchema="true")
test_dl = spark.read.csv(test_data_label,header=False,inferSchema="true")


assembler_train = VectorAssembler(inputCols=train_dl.columns[1:],
    outputCol="features")
train_vectors = assembler_train.transform(train_dl).select(["_c0","features"])


assembler_test = VectorAssembler(inputCols=test_dl.columns[1:],
    outputCol="features")
test_vectors = assembler_test.transform(test_dl).select(["_c0","features"])


pca = PCA(k=nPCADimension, inputCol="features", outputCol="pca")
train_model = pca.fit(train_vectors)
train_pca_result = train_model.transform(train_vectors).select(["_c0","pca"])
test_pca_result = train_model.transform(test_vectors).select(["_c0","pca"])


train_l = train_pca_result.select("_c0")
train_d = train_pca_result.select("pca")

train_l.show(2)
train_d.show(2)

trainlabels_list = train_l.rdd.map(lambda x: x.asDict()["_c0"]).collect()
trainfeatures_list = train_d.rdd.map(lambda x: np.array(x.asDict()["pca"])).collect()
sc = spark.sparkContext

trainlabels = sc.broadcast(trainlabels_list)
trainfeatures = sc.broadcast(trainfeatures_list)
temp_test = test_pca_result.rdd.map(num).cache()

# =================================
print("start calculation")
import time
start_calculation = time.time()
finalRDD = temp_test.repartition(nPartition).map(knn).cache()

strOutputPath = 'Assignment2/' + strExpName

finalRDD.saveAsTextFile(strOutputPath)

final_list = finalRDD.collect()
end_calculation = time.time()
print(end_calculation - start_calculation)

count = 0
for i in range(10000):
    predict, correct = final_list[i]
    if predict == correct:
        count += 1

print(count)

predict_dict = {}
correct_dict = {}
matched_dict = {}

for i in range(10000):
    predict, correct = final_list[i]
    if predict not in predict_dict:
        predict_dict[predict] = 0
    if correct not in correct_dict:
        correct_dict[correct] = 0
    predict_dict[predict] += 1
    correct_dict[correct] += 1
    if predict == correct:
        if predict not in matched_dict:
            matched_dict[predict] = 0
        matched_dict[predict] += 1

print("statistics")
for i in correct_dict:
    print(i)
    print("precision: " + "{0:.2f}".format(float(matched_dict[i]) * 100 /predict_dict[i]))
    print("recall: " + "{0:.2f}".format(float(matched_dict[i]) * 100 /correct_dict[i]))
    print("f1: " +"{0:.2f}".format((2 * 100 * float(matched_dict[i])/predict_dict[i] * float(matched_dict[i])/correct_dict[i]) / (float(matched_dict[i])/predict_dict[i] + float(matched_dict[i])/correct_dict[i])))
