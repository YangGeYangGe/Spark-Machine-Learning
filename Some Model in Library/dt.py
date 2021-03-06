import argparse
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


#Take Dimension Parameter
parser = argparse.ArgumentParser()
parser.add_argument("--dim", help="dimension", default=50)
args = parser.parse_args()
d = int(args.dim)
print("d is: ",d)



#######Initiate the Spark Session #####################################################
spark = SparkSession \
    .builder \
    .appName("Test case DT 02") \
    .getOrCreate()

################Preparation and Fetching data from MNIST Source ###############################
train_data_label = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"
test_data_label = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"

#################Read the data from CSV#####################################################
train_dl = spark.read.csv(train_data_label,header=False,inferSchema="true")
test_dl = spark.read.csv(test_data_label,header=False,inferSchema="true")



################## Assemble Data ############################################################
assembler_train = VectorAssembler(inputCols=train_dl.columns[1:],
    outputCol="features")
train_vectors = assembler_train.transform(train_dl).select(["_c0","features"])
assembler_test = VectorAssembler(inputCols=test_dl.columns[1:],
    outputCol="features")
test_vectors = assembler_test.transform(test_dl).select(["_c0","features"])


#d = 100
pca = PCA(k=d, inputCol="features", outputCol="pca")
train_model = pca.fit(train_vectors)
train_pca_result = train_model.transform(train_vectors).select(["_c0","pca"])

test_pca_result = train_model.transform(test_vectors).select(["_c0","pca"])


###################Showing both train and test vector
train_vectors.show(5)
test_vectors.show(5)

######### Perform the Prediction #############################################################
dt = DecisionTreeClassifier(labelCol="_c0", featuresCol="pca")
model = dt.fit(train_pca_result)

predict = model.transform(test_pca_result)
predict.select("_c0","prediction" ).show(5)

result_list = predict.select("_c0","prediction" ).collect()
count = 0
for i in range(10000):
    if result_list[i][0] == result_list[i][1]:
        count +=1
print(count/100)
