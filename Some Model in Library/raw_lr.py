from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors,SparseVector, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



spark = SparkSession \
    .builder \
    .appName("Python Spark Machine Learning Example") \
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
train_vectors.show(5)
test_vectors.show(5)

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=10,labelCol="_c0", featuresCol="features")

# Fit the model
lrModel = lr.fit(train_vectors)

# Print the coefficients and intercept for multinomial logistic regression
print("Coefficients: \n" + str(lrModel.coefficientMatrix))
print("Intercept: " + str(lrModel.interceptVector))

predict = lrModel.transform(test_vectors)
predict.select(["_c0","prediction"]).show(10)

result_list = predict.select("_c0","prediction" ).collect()
count = 0
for i in range(10000):
    if result_list[i][0] == result_list[i][1]:
        count +=1
print(float(count)/10000)
