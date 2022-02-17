import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

conf  = SparkConf().setMaster("local[*]").setAppName("length") 
sc = SparkContext(conf=conf)

spark = SparkSession.builder.appName('hw3').getOrCreate()

# Reads in the csv file
readin = sc.textFile("Amazon_Comments.csv")

# Creates list for each column
# [0] = ProductID
# [1] = ReviewID
# [2] = Review Title
# [3] = Review Time
# [4] = Verified 
# [5] = Review
# [6] = Rating
readin_sep = readin.map(lambda x: x.split("^"))

# Turn to (Key, Value) where it is (Rating, Review) where review is all lower cased 
readin_clean = readin_sep.map(lambda x: (x[6], re.sub('\W+',' ', x[5]).strip().lower()))

comments_columns = ["Rating","Review"]
comments_df = spark.createDataFrame(data=readin_clean.collect(), schema = comments_columns)

def splitAndCountUdf(x):
    return len(x.split(" "))

countWords = F.udf(splitAndCountUdf, 'int')

comments_df = comments_df.withColumn("wordCount", countWords(comments_df.Review))

comments_df.groupBy('Rating').mean("wordCount").withColumnRenamed("avg(wordCount)","Average Length of Comments").show()
