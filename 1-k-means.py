import numpy as np
from math import sqrt
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

conf  = SparkConf().setMaster("local[*]").setAppName("length") 
sc = SparkContext(conf=conf)

spark = SparkSession.builder.appName('hw4').getOrCreate()
# Load and parse the data
data = sc.textFile("kmeans_data.txt")



data = data.map(lambda x: x.split(" "))

columns = ["index","x","y"]
df = spark.createDataFrame(data=data, schema = columns)

print(df.show())

feature_columns = ['x','y']

for col in df.columns:
    if col in feature_columns:
        df = df.withColumn(col,df[col].cast('float'))

vecAssembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_kmeans = vecAssembler.transform(df).select('index', 'features')
df_kmeans.show()

kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(df_kmeans)

predictions = model.transform(df_kmeans)

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
