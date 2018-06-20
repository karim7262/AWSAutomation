import os
import sys
import boto3

from awsglue.job import Job
from awsglue.transforms import *
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions

import pyspark.sql.functions as F
from pyspark.sql import Row, Window, SparkSession
from pyspark.sql.types import *
from pyspark.conf import SparkConf
from pyspark.context import SparkContext


args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
#spark = glueContext.spark_session
spark = SparkSession.builder.config("spark.sql.broadcastTimeout", "600").getOrCreate()
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

spark._jsc.hadoopConfiguration().set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
spark._jsc.hadoopConfiguration().set("parquet.enable.summary-metadata", "false")

AWS_REGION = 'us-east-1'
MIN_SENTENCE_LENGTH_IN_CHARS = 10
MAX_SENTENCE_LENGTH_IN_CHARS = 5000
COMPREHEND_BATCH_SIZE = 50  ## This batch size results in groups no larger than 25 items
NUMBER_OF_BATCHES = 4
ROW_LIMIT = 10000

## Each task handles 25*4 records, there should be 10 partitions overall to process 1000 records.

SentimentRow = Row("review_id", "sentiment")
def getBatchSentiment(input_list):
  ## You can import the ratelimit module if you want to further rate limit API calls to Comprehend
  ## https://pypi.org/project/ratelimit/
  #from ratelimit import rate_limited
  arr = []
  bodies = [i[1] for i in input_list]
  client = boto3.client('comprehend',region_name = AWS_REGION)

  #@rate_limited(1) 
  def callApi(text_list):
    response = client.batch_detect_sentiment(TextList = text_list, LanguageCode = 'en')
    return response
  
  for i in range(NUMBER_OF_BATCHES-1):
    text_list = bodies[COMPREHEND_BATCH_SIZE * i : COMPREHEND_BATCH_SIZE * (i+1)]
    response = callApi(text_list)
    for r in response['ResultList']:
      idx = COMPREHEND_BATCH_SIZE * i + r['Index']
      arr.append(SentimentRow(input_list[idx][0], r['Sentiment']))
  
  return arr
  
### Main data processing

## Read source Amazon review dataset
print '*************************************************************START REVIEW*******************************************************************'
reviews = spark.read.parquet("s3://amazon-review-dataset-sample1/part-0000.snappy.parquet").distinct()
#reviews = spark.read.parquet("s3://amazon-review-dataset-sample1/sample_us.tsv").distinct()
#reviews = spark.read.
#reviews = df = sc.load("s3://amazon-review-dataset-sample1/sample_us.tsv")
#reviews  = sc.csvFile("s3://amazon-review-dataset-sample1/sample_us.csv")
print '***************************************************************END REVIEW*****************************************************************'
print reviews
## Filter down dataset to only those reviews that we expect will return a meaningful result
df = reviews \
  .filter("marketplace = 'US'") \
  .withColumn('body_len', F.length('review_body')) \
  .filter(F.col('body_len') > MIN_SENTENCE_LENGTH_IN_CHARS) \
  .filter(F.col('body_len') < MAX_SENTENCE_LENGTH_IN_CHARS) \
  .limit(ROW_LIMIT)

record_count = df.count()

## Group together rows and sort them
df2 = df \
  .repartition(record_count / (NUMBER_OF_BATCHES * COMPREHEND_BATCH_SIZE)) \
  .sortWithinPartitions(['review_id'], ascending=True)
  
## Concatenate review id and body tuples into arrays of similar size
group_rdd = df2.rdd.map(lambda l: (l.review_id, l.review_body)).glom()
## iterate over the groups or rows and call Comprehend API
sentiment = group_rdd.coalesce(10).map(lambda l: getBatchSentiment(l)).flatMap(lambda x: x).toDF().repartition('review_id').cache()
print '************************************************************************START SENTIMENT********************************************************'
print sentiment
print '************************************************************************END SENTIMENT**********************************************************'
## Join sentiment results with the review dataset
joined = reviews \
  .drop('review_body') \
  .join(sentiment, sentiment.review_id == reviews.review_id) \
  .drop(sentiment.review_id)

## Write out result set to S3 in Parquet format
joined.write.partitionBy('product_category').mode('overwrite').parquet('s3://amazon-review-dataset-sample1/')

job.commit()
