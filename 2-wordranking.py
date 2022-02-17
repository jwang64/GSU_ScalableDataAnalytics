import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import nltk

from nltk.corpus import stopwords
stopword_list = set(stopwords.words("english"))


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

def stopWordRemover(text):
    text = ' '.join([word for word in text.split() if word not in stopword_list])
    return text

stopWords = F.udf(stopWordRemover)

comments_df = comments_df.withColumn("updatedReviews", stopWords(comments_df.Review))

comments_df.show()

rating1_comments_df = comments_df.filter(comments_df.Rating == '1.00')

rating1_comments = rating1_comments_df.select("updatedReviews").rdd.flatMap(lambda x: x).collect()
rating1_word_list = ' '.join(rating1_comments).split(' ')
rating1_freqs = nltk.FreqDist(rating1_word_list)
print('1 Star Rating Top 10 Most Common Words: ' + str(rating1_freqs.most_common(10)))

rating2_comments_df = comments_df.filter(comments_df.Rating == '2.00')

rating2_comments = rating2_comments_df.select("updatedReviews").rdd.flatMap(lambda x: x).collect()
rating2_word_list = ' '.join(rating2_comments).split(' ')
rating2_freqs = nltk.FreqDist(rating2_word_list)
print('2 Star Rating Top 10 Most Common Words: ' + str(rating2_freqs.most_common(10)))

rating3_comments_df = comments_df.filter(comments_df.Rating == '3.00')

rating3_comments = rating3_comments_df.select("updatedReviews").rdd.flatMap(lambda x: x).collect()
rating3_word_list = ' '.join(rating3_comments).split(' ')
rating3_freqs = nltk.FreqDist(rating3_word_list)
print('3 Star Rating Top 10 Most Common Words: ' + str(rating3_freqs.most_common(10)))

rating4_comments_df = comments_df.filter(comments_df.Rating == '4.00')

rating4_comments = rating4_comments_df.select("updatedReviews").rdd.flatMap(lambda x: x).collect()
rating4_word_list = ' '.join(rating4_comments).split(' ')
rating4_freqs = nltk.FreqDist(rating4_word_list)
print('4 Star Rating Top 10 Most Common Words: ' + str(rating4_freqs.most_common(10)))

rating5_comments_df = comments_df.filter(comments_df.Rating == '5.00')

rating5_comments = rating5_comments_df.select("updatedReviews").rdd.flatMap(lambda x: x).collect()
rating5_word_list = ' '.join(rating5_comments).split(' ')
rating5_freqs = nltk.FreqDist(rating5_word_list)
print('5 Star Rating Top 10 Most Common Words: ' + str(rating5_freqs.most_common(10)))
