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
from pyspark.sql.functions import *
from pyspark.sql.functions import col
from pyspark.sql import functions as F


spark = SparkSession.builder.master('local[*]').getOrCreate() #criar/iniciar a sessão spark

# Importando bibliotecas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

movies = spark.read.options(header='True', inferSchema='True', delimiter=',').csv("movies.csv")
movies.show()

movies.count()

ratings = spark.read.options(header='True', inferSchema='True', delimiter=',').csv("ratingsNew.csv")
ratings.show()

ratings.count()

merged_df = ratings.unionByName(movies, allowMissingColumns=True)
merged_df.show()

merged_df = merged_df.drop("title", "genres")
merged_df.show()

df_result = merged_df.dropna()
df_result

df_result.count()

df_result.show()

df_result.select(df_result['rating'] > 5.0).show()

(training, test) = df_result.randomSplit([0.7, 0.3])

als = ALS(maxIter=10, rank=4, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

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

userRec.show(10, False)

movieRecs = model.recommendForAllItems(10) #faz a transposta da matriz de ratings, a fim de recomendar usuários em potencial

movieRecs.show(10, False)

**FUNÇÕES ÚTEIS UTILIZADAS PELO SISTEMA**



def info_movie(id_movie):
  movies.select("title", "genres").where(movies['movieId'] == id_movie).show()
  df_result.select("userId", "rating").where(df_result['movieId'] == id_movie).show()
  df_result.groupBy("movieId").count().where(df_result["movieId"] == id_movie).show()

def info_user(id_user):
  ratings.select("userId", "movieId","rating").where( (ratings['userId'] == id_user)  ).show()
  df_result.groupBy("userId").count().where(df_result["userId"] == id_user).show()
  df_result.groupBy("userId").max("rating").where(df_result["userId"] == id_user).show()
  df_result.groupBy("userId").min("rating").where(df_result["userId"] == id_user).show()


**==============================================================**

info_user(336)

info_movie(8621)

UserRecsOnlyItemId = userRec.select(userRec['userId'], userRec['recommendations']['movieId'])
UserRecsOnlyItemId.show(10, False)

# df = UserRecsOnlyItemId.toPandas()
# for ind in df.index:
#     print(df['userId'][ind], df['recommendations.movieId'][ind])

df['userId'][0]
df['recommendations.movieId'][0]

# movieId|               title|              genres
def buscaFilme(movieId):
    movies.select("title", "genres").where(movies['moveiId'] == movieId).show()

**RMSE x MAE para cada usuário**


(training, test) = df_result.randomSplit([0.7, 0.3])

als = ALS(maxIter=5, rank=4, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

model = als.fit(training) #treina o modelo com o dataset de treinamento

predictions = model.transform(test) #aplica o modelo no conjunto de teste para fazer predições
evaluator = RegressionEvaluator(metricName="mae", labelCol="rating",
                               predictionCol="prediction")
mae = evaluator.evaluate(predictions)
print("Mean absolute error (MAE) on test data = " + str(mae))


predictions = model.transform(test) #aplica o modelo no conjunto de teste para fazer predições
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                               predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Erro médio quadrático = " + str(rmse))


# Especificar o ID do usuário para o qual você deseja fazer recomendações
usuario_especifico = 12

# Criar um DataFrame com os itens para os quais você deseja fazer recomendações
itens_para_recomendar = df_result.select("movieId").distinct()

# Adicionar a coluna do ID do usuário
itens_para_recomendar = itens_para_recomendar.withColumn("userId", lit(usuario_especifico))

# Gerar recomendações para o usuário específico
recomendacoes = model.transform(itens_para_recomendar)

# Ordenar as recomendações pela pontuação prevista em ordem decrescente
recomendacoes = recomendacoes.orderBy("prediction", ascending=False)

# Mostrar as recomendações
recomendacoes.select("movieId", "prediction").show()

user_metrics = predictions.groupBy("userId").agg(
    F.sqrt(F.mean(F.pow(F.col("rating") - F.col("prediction"), 2))).alias("user_rmse"),
    F.mean(F.abs(F.col("rating") - F.col("prediction"))).alias("user_mae")
)

# user_metrics.show()

user_metrics.select("userId", "user_rmse", "user_mae").filter(user_metrics["userId"] == 5).show()


# user2 = user_metrics.select("userId", "user_rmse", "user_mae").where(user_metrics["userId"] == 3).show()
# print('user2', user2)

# user3 = user_metrics.select("userId", "user_rmse", "user_mae").where(user_metrics["userId"] == 5).show()
# print('user3', user3)

PLOTAR MÉTRICAS DE USUÁRIO EM GRÁFICO DE BARRAS

# Coletar dados em listas
user_ids = user_metrics.select("userId").rdd.flatMap(lambda x: x).collect()
rmse_values = user_metrics.select("user_rmse").rdd.flatMap(lambda x: x).collect()
mae_values = user_metrics.select("user_mae").rdd.flatMap(lambda x: x).collect()

# Filtrar métricas para o usuário específico
user_data = user_metrics.filter(user_metrics["userId"] == 5).select("user_rmse", "user_mae").collect()[0]

rmse_value = user_data["user_rmse"]
mae_value = user_data["user_mae"]

# Criar gráfico de barras
labels = ['RMSE', 'MAE']
values = [rmse_value, mae_value]

fig, ax = plt.subplots()
ax.bar(labels, values, color=['blue', 'orange'])
ax.set_title(f'Comparação RMSE e MAE para Usuário {5}')
ax.set_ylabel('Valor')

plt.show()

# Suponha que você já tenha calculado as métricas por usuário (usando o DataFrame 'user_metrics')
# e que 'user_metrics' seja o DataFrame que contém as métricas de RMSE e MAE por usuário.

# Substitua 1 pelo ID do usuário específico que você deseja visualizar
#plot_rmse_mae_comparison(1, user_metrics)

# Criar gráficos individuais para cada usuário
# user_ids = [1]
# fig, ax = plt.subplots()
# bar_width = 0.35
# index = np.arange(len(user_ids))
    
# labels = []
# bar1 = ax.bar(index, [rmse_values[i]], bar_width, label='RMSE')
# bar2 = ax.bar(index + bar_width, [mae_values[i]], bar_width, label='MAE')

# ax.set_xlabel('Métrica')
# ax.set_ylabel('Valor')
# ax.set_title(f'RMSE e MAE para Usuário {user_id}')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(labels, fontdict=None, minor=False)
# ax.legend()
# plt.show()