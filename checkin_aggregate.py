import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd


conf  = SparkConf().setMaster("local[*]").setAppName("length") 
sc = SparkContext(conf=conf)

spark = SparkSession.builder.appName('checkin').getOrCreate()

# Reads in the csv file
df=spark.read.csv('yelp_checkin.csv', mode="DROPMALFORMED",inferSchema=True, header = True)

aggregated_df = df.groupBy('business_id').sum('checkins').withColumnRenamed("sum(checkins)","totalCheckin")

aggregated_df.show()

aggregated_df.toPandas().to_csv('aggregated_checkin.csv')