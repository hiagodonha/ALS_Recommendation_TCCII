!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://archive.apache.org/dist/spark/spark-3.3.2/spark-3.3.2-bin-hadoop2.tgz
!tar xf spark-3.3.2-bin-hadoop2.tgz
!pip install -q findspark

# configurar as variáveis de ambiente
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.3.2-bin-hadoop2"

# tornar o pyspark "importável"
import findspark
findspark.init('spark-3.3.2-bin-hadoop2')

from __future__ import print_function

import sys
if sys.version >= '3':
    long = int

from pyspark.sql import SparkSession

from pyspark.ml.evaluation import RegressionEvaluator #evaluation é a biblioteca para verificação da qualidade do modelo
from pyspark.ml.recommendation import ALS # ALS é o modelo de recomendação que será utilizadp
from pyspark.sql import Row #row é o formato que o ALS trabalha, row conterá o id do usuario, id filme, nota e timestamp

spark = SparkSession.builder.master('local[*]').getOrCreate() #criar/iniciar a sessão spark

# Importando bibliotecas
import numpy as np
import pandas as pd
import seaborn as sns

movies = spark.read.options(header='True', inferSchema='True', delimiter=',').csv("movies.csv")
movies.show()

ratings = spark.read.options(header='True', inferSchema='True', delimiter=',').csv("ratingsNew.csv")
ratings.show()

merged_df = ratings.unionByName(movies, allowMissingColumns=True)
merged_df.show()

merged_df = merged_df.drop("title", "genres")
merged_df.show()

df_result = merged_df.dropna()

df_result.count()

(training, test) = df_result.randomSplit([0.7, 0.3])

als = ALS(maxIter=5, rank=4, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

model = als.fit(training) #treina o modelo com o dataset de treinamento

predictions = model.transform(test) #aplica o modelo no conjunto de teste para fazer predições
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                               predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Erro médio quadrático = " + str(rmse))

predictions = model.transform(test) #aplica o modelo no conjunto de teste para fazer predições
evaluator = RegressionEvaluator(metricName="mae", labelCol="rating",
                               predictionCol="prediction")
mae = evaluator.evaluate(predictions)
print("Mean absolute error (MAE) on test data = " + str(mae))

userRec = model.recommendForAllUsers(10) #pegar todos os usuários e gerar 10 recomendações

name = movies.select("title", "genres").where(movies['movieId'] == 2314).show()
print(name)

movieRecs = model.recommendForAllItems(10) #faz a transposta da matriz de ratings, a fim de recomendar usuários em potencial

movieRecs.show()

name = movies.select("title").where(movies['movieId'] == 12).show()
print(name)

UserRecsOnlyItemId = userRec.select(userRec['userId'], userRec['recommendations']['movieId'])
UserRecsOnlyItemId.show(10, False)