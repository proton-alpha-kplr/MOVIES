from pyspark.sql.types import *
from pyspark.sql.functions import explode, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext

class RecommendationEngine:
    def __init__(self, sc, movies_set_path, ratings_set_path):
    # Méthode d'initialisation pour charger les ensembles de données et entraîner le modèle
    # Cette méthode d'initialisation est appelée lors de la création d'une instance de la classe RecommendationEngine.
    # Elle prend en paramètres le contexte Spark (sc), le chemin vers l'ensemble de données de films (movies_set_path) et le chemin vers l'ensemble de données d'évaluations (ratings_set_path).

    # La méthode initialise le contexte SQL à partir du contexte Spark sc
    sqlContext = SQLContext(self.sc)
    
    # Charge les données des ensembles de films et d'évaluations à partir des fichiers CSV spécifiés

    #Movies

    # définit le schéma des données
    moviesStruct = [StructField("movieId", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("genres", StringType(), True)]

    moviesSchema = StructType(moviesStruct)

    moviesDF = spark.read.format("csv").option("header", "true").option("delimiter", ",").schema(moviesSchema).load(movies_set_path)
    moviesDF.write.parquet("hdfs:///user/root/data/MOV/PARQUET/movies.parquet")

    self.moviesDF = spark.read.parquet("hdfs:///user/root/data/MOV/PARQUET/movies.parquet")
    self.moviesDF.cache()

    #Ratings
    # définit le schéma des données
    ratingsStruct = [StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", DoubleType(), True),
        StructField("timestamp", IntegerType(), True)]

    ratingsSchema = StructType(ratingsStruct)

    # Read ratings from HDFS (FIRST TIME ONLY)
    ratingsDF = spark.read.format("csv").option("header", "true").option("delimiter", ",").schema(ratingsSchema).load("hdfs:///user/root/data/MOV/CSV/ratings.csv")
    ratingsDF.write.parquet("hdfs:///user/root/data/MOV/PARQUET/ratings.parquet")

    self.ratingsDF = spark.read.parquet("hdfs:///user/root/data/MOV/PARQUET/ratings.parquet").drop("timestamp")
    self.ratingsDF.cache()

    # Unique Users Id :
    self.usersDF = ratingsDF.select("userId").distinct()

    # effectue diverses opérations de traitement des données

    # entraîne le modèle en utilisant la méthode privée __train_model()
    __train_model(self)



    def create_user(self, user_id):
        # Méthode pour créer un nouvel utilisateur
            # Cette méthode permet de créer un nouvel utilisateur.
            # Elle prend en paramètre un user_id facultatif pour spécifier l'identifiant de l'utilisateur. Si user_id est None, un nouvel identifiant est généré automatiquement.
            # Si user_id est supérieur à max_user_identifier, max_user_identifier est mis à jour avec la valeur de user_id.
            # La méthode retourne l'identifiant de l'utilisateur créé ou mis à jour.
        max_user_identifier = self.ratingsDF.agg({"userId": "max"}).collect()[0][0]
        if user_id == None:
            return max_user_identifier + 1
        elif is_user_known(user_id):
            return(user_id)
        else:
            return max_user_identifier + 1
        

    def is_user_known(self, user_id):
        # Méthode pour vérifier si un utilisateur est connu
        # Cette méthode permet de vérifier si un utilisateur est connu.
        # Elle prend en paramètre un user_id et retourne True si l'utilisateur est connu (c'est-à-dire si user_id est différent de None et inférieur ou égal à max_user_identifier), sinon elle retourne False.
    existing_count = self.ratingsDF.filter(self.ratingsDF.userID == user_id).count()

    # Vérifier le résultat
    if existing_count > 0:
        return True
    else:
        return False
 
    def get_movie(self, movie_id):
        # Méthode pour obtenir un film
        # Cette méthode permet d'obtenir un film.
        # Elle prend en paramètre un movie_id facultatif pour spécifier l'identifiant du film. Si movie_id est None, la méthode retourne un échantillon aléatoire d'un film à partir du dataframe best_movies_df. Sinon, elle filtre le dataframe movies_df pour obtenir le film correspondant à movie_id.
        # La méthode retourne un dataframe contenant les informations du film (colonne "movieId" et "title").
    
    if movie_id == None:
        random_movie_id = [row.movieId for row in self.moviesDF.select("movieId").collect()]
        return self.moviesDF.filter(self.moviesDF.movieId == random_movie_id).select("movieId","title")
    else:
        return self.moviesDF.filter(self.moviesDF.movieId == movie_id).select("movieId","title")


    def get_ratings_for_user(self, user_id):
        # Méthode pour obtenir les évaluations d'un utilisateur
        # Cette méthode permet d'obtenir les évaluations d'un utilisateur.
        # Elle prend en paramètre un user_id et filtre le dataframe ratings_df pour obtenir les évaluations correspondantes à l'utilisateur.
        # La méthode retourne un dataframe contenant les évaluations de l'utilisateur (colonnes "movieId", "userId" et "rating").
        return self.ratingsDF.filter(self.ratingsDF.userID == user_id).select("movieId", "userId", "rating")
        

    def add_ratings(self, user_id, ratings):
        # Méthode pour ajouter de nouvelles évaluations et re-entraîner le modèle
        # Cette méthode permet d'ajouter de nouvelles évaluations au modèle et de re-entraîner le modèle.
        # Elle prend en paramètres un user_id et une liste de ratings contenant les nouvelles évaluations.
        # La méthode crée un nouveau dataframe new_ratings_df à partir de la liste de ratings et l'ajoute au dataframe existant ratings_df en utilisant l'opération union().
        # Ensuite, les données sont divisées en ensembles d'entraînement (training) et de test (test) en utilisant la méthode randomSplit().
        # Enfin, la méthode privée __train_model() est appelée pour re-entraîner le modèle.

        
    new_ratings_schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", DoubleType(), True)
    ])

    new_ratings_df = spark.createDataFrame(ratings, new_ratings_schema)

    self.ratings_df = self.ratings_df.union(newRatingsDF)

    self.trainingDF, self.testDF = self.ratings_df.randomSplit([0.8, 0.2], seed=12345)

    __train_model(self)



    def predict_rating(self, user_id, movie_id):
        # Méthode pour prédire une évaluation pour un utilisateur et un film donnés
        # Cette méthode permet de prédire une évaluation pour un utilisateur et un film donnés.
        # Elle prend en paramètres un user_id et un movie_id.
        # La méthode crée un dataframe rating_df à partir des données (user_id, movie_id) et le transforme en utilisant le modèle pour obtenir les prédictions.
        # Si le dataframe de prédiction est vide, la méthode retourne -1, sinon elle retourne la valeur de prédiction.

        ratingsStruct = [StructField("userId", IntegerType(), True),
            StructField("movieId", IntegerType(), True),
            StructField("rating", DoubleType(), True)]
        
        row = [user_id, movie_id, None]

        rating_df = spark.createDataFrame(row, ratingsStruct)

        predictions = model.transform(rating_df)




    def recommend_for_user(self, user_id, nb_movies):
        # Méthode pour obtenir les meilleures recommandations pour un utilisateur donné
        ...

    def __train_model(self):
        # Méthode privée pour entraîner le modèle avec ALS
        ...

    def __evaluate(self):
        # Méthode privée pour évaluer le modèle en calculant l'erreur quadratique moyenne
        ...

