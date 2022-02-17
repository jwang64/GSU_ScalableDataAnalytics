import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
from pyspark.ml.classification import LogisticRegression

from nltk.corpus import stopwords
from pyspark.ml.feature import  Tokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import  IDF

from pyspark.ml.classification import LogisticRegression

stopword_list = set(stopwords.words("english"))


conf  = SparkConf().setMaster("local[*]").setAppName("length") 
conf.set("spark.sql.csv.parser.columnPruning.enabled",False)
sc = SparkContext(conf=conf)

spark = SparkSession.builder.appName('checkin').getOrCreate()

# Reads in the csv file
df=spark.read.csv('yelp_review_subset.csv', mode="DROPMALFORMED",inferSchema=True, header = True)
df = df.drop('review_id','user_id','business_id')


def stopWordRemover(text):
    text = ' '.join([word for word in text.split() if word not in stopword_list])
    return text
    
stopWords = F.udf(stopWordRemover)
df = df.withColumn("text", stopWords(df.text))

tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(df)

count = CountVectorizer (inputCol="words", outputCol="rawFeatures")
model = count.fit(wordsData)
featurizedData = model.transform(wordsData)
featurizedData.show()

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData = rescaledData.withColumnRenamed('stars','label')
rescaledData.select("label", "features").show()

lr = LogisticRegression(maxIter = 10)

lr_model = lr.fit(rescaledData)

predictions_lr = lr_model.transform(rescaledData)

predictions_lr.show(10)