from pyspark.sql.types import *
from pyspark.sql.functions import explode, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext



class RecommendationEngine:



    def is_user_known(self, user_id):
        # Méthode pour vérifier si un utilisateur est connu
        # Cette méthode permet de vérifier si un utilisateur est connu.
        # Elle prend en paramètre un user_id et retourne True si l'utilisateur est connu (c'est-à-dire si user_id est différent de None et inférieur ou égal à max_user_identifier), sinon elle retourne False.
        existing_count = self.ratingsDF.filter(self.ratingsDF.userId == user_id).count()

    # Vérifier le résultat
        return existing_count > 0 #Vrai s'il y en a Faux sinon


    def create_user(self, user_id):
        # Méthode pour créer un nouvel utilisateur
            # Cette méthode permet de créer un nouvel utilisateur.
            # Elle prend en paramètre un user_id facultatif pour spécifier l'identifiant de l'utilisateur. Si user_id est None, un nouvel identifiant est généré automatiquement.
            # Si user_id est supérieur à max_user_identifier, max_user_identifier est mis à jour avec la valeur de user_id.
            # La méthode retourne l'identifiant de l'utilisateur créé ou mis à jour.
        max_user_identifier = self.ratingsDF.agg({"userId": "max"}).collect()[0][0]
        if user_id == None:
            return max_user_identifier + 1
        elif self.is_user_known(user_id):
            return(user_id)
        else:
            return max_user_identifier + 1
        
 
    def get_movie(self, movie_id):
        # Méthode pour obtenir un film
        # Cette méthode permet d'obtenir un film.
        # Elle prend en paramètre un movie_id facultatif pour spécifier l'identifiant du film. Si movie_id est None, la méthode retourne un échantillon aléatoire d'un film à partir du dataframe best_movies_df. Sinon, elle filtre le dataframe movies_df pour obtenir le film correspondant à movie_id.
        # La méthode retourne un dataframe contenant les informations du film (colonne "movieId" et "title").
    
        if movie_id == None:
            return self.moviesDF.sample(1, seed=42).first().select("movieId","title")

        else:
            return self.moviesDF.filter(self.moviesDF.movieId == movie_id).select("movieId","title")


    def get_ratings_for_user(self, user_id):
        # Méthode pour obtenir les évaluations d'un utilisateur
        # Cette méthode permet d'obtenir les évaluations d'un utilisateur.
        # Elle prend en paramètre un user_id et filtre le dataframe ratings_df pour obtenir les évaluations correspondantes à l'utilisateur.
        # La méthode retourne un dataframe contenant les évaluations de l'utilisateur (colonnes "movieId", "userId" et "rating").
        return self.ratingsDF.filter(self.ratingsDF["userId"] == user_id).select("movieId", "userId", "rating")
        

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

        new_ratings_df = self.sc.createDataFrame(ratings, new_ratings_schema)

        self.ratings_df = self.ratings_df.union(newRatingsDF)

        self.trainingDF, self.testDF = self.ratings_df.randomSplit([0.8, 0.2], seed=12345)

        self.__train_model()



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

        rating_df = self.sc.createDataFrame(row, ratingsStruct)

        predictions_df = self.model.transform(rating_df)

        if predictions_df.count() > 0:
            return predictions_df.filter(self.ratingsDF["userId"] == user_id).filter(self.ratingsDF["movie_id"] == movie_id).select("prediction").collect()[0][0]
        else :
            return -1



    def recommend_for_user(self, user_id, nb_movies):
        # Méthode pour obtenir les meilleures recommandations pour un utilisateur donné
        # Cette méthode permet d'obtenir les meilleures recommandations pour un utilisateur donné.
        # Elle prend en paramètres un user_id et un nombre de films nb_movies à recommander.
        # La méthode crée un dataframe user_df contenant l'identifiant de l'utilisateur et utilise la méthode recommendForUserSubset() du modèle pour obtenir les recommandations pour cet utilisateur.
        # Les recommandations sont ensuite jointes avec le dataframe movies_df pour obtenir les détails des films recommandés.
        # Le dataframe résultant est retourné avec les colonnes "title" et d'autres colonnes du dataframe movies_df.

        user_df = self.sc.createDataFrame([user_id])
        userSubsetRecs_df = model.recommendForUserSubset(user_df, nb_movies)
        result_df = self.movies_df.join(userSubsetRecs_df, userSubsetRecs_df["movieId"] ==  movies_df["movieId"] , "inner")
        result_df = result_df.select(self.movies_df["movieId"], self.movies_df["title"], self.movies_df["genres"])
        print(display(result_df))
        return result_df



    def __train_model(self):
        # Méthode privée pour entraîner le modèle avec ALS
        # Cette méthode privée permet d'entraîner le modèle avec l'algorithme ALS (Alternating Least Squares).
        # Elle utilise les paramètres maxIter et regParam définis dans l'initialisation de la classe pour créer une instance de l'algorithme ALS.
        # Ensuite, le modèle est entraîné en utilisant le dataframe training.
        # La méthode privée __evaluate() est appelée pour évaluer les performances du modèle.
        
        als = ALS(maxIter=self.maxIter_class,
          regParam=self.regParam_class, 
          implicitPrefs=False, 
          userCol="userId", 
          itemCol="movieId", 
          ratingCol="rating", 
          coldStartStrategy="drop")

        self.model = als.fit(self.trainingDF)

        self.__evaluate()

    def __evaluate(self):
        # Méthode privée pour évaluer le modèle en calculant l'erreur quadratique moyenne
        # Cette méthode privée permet d'évaluer le modèle en calculant l'erreur quadratique moyenne (RMSE - Root-mean-square error).
        # Elle utilise le modèle pour prédire les évaluations sur le dataframe test.
        # Ensuite, elle utilise l'évaluateur de régression pour calculer le RMSE en comparant les prédictions avec les vraies évaluations.
        # La valeur de RMSE est stockée dans la variable rmse de la classe et affichée à l'écran.

        evaluator = RegressionEvaluator(
           metricName="rmse", 
           labelCol="rating", 
           predictionCol="prediction") 
        
        self.trainingDF, self.testDF = self.ratingsDF.randomSplit([0.8, 0.2], seed=12345)
        
        predictions = self.model.transform(self.testDF)
        
        self.rmse = evaluator.evaluate(predictions)
        print("Root-mean-square error = " + str(self.rmse))


    def __init__(self, sc, movies_set_path, ratings_set_path):
        # Méthode d'initialisation pour charger les ensembles de données et entraîner le modèle
        # Cette méthode d'initialisation est appelée lors de la création d'une instance de la classe RecommendationEngine.
        # Elle prend en paramètres le contexte Spark (sc), le chemin vers l'ensemble de données de films (movies_set_path) et le chemin vers l'ensemble de données d'évaluations (ratings_set_path).

        self.sc = sc
        self.movies_set_path = movies_set_path
        self.ratings_set_path = ratings_set_path

        # La méthode initialise le contexte SQL à partir du contexte Spark sc
        sqlContext = SQLContext(self.sc)
        
        # Charge les données des ensembles de films et d'évaluations à partir des fichiers CSV spécifiés

        #Movies

        # définit le schéma des données
        moviesStruct = [StructField("movieId", IntegerType(), True),
            StructField("title", StringType(), True),
            StructField("genres", StringType(), True)]

        moviesSchema = StructType(moviesStruct)

        moviesDF = self.sc.read.format("csv").option("header", "true").option("delimiter", ",").schema(moviesSchema).load(movies_set_path)
        moviesDF.write.mode("overwrite").format("parquet").save("data_movies.parquet")

        self.moviesDF = self.sc.read.parquet("data_movies.parquet")
        self.moviesDF.cache()

        #Ratings
        # définit le schéma des données
        ratingsStruct = [StructField("userId", IntegerType(), True),
            StructField("movieId", IntegerType(), True),
            StructField("rating", DoubleType(), True),
            StructField("timestamp", IntegerType(), True)]

        ratingsSchema = StructType(ratingsStruct)

        # Read ratings from HDFS (FIRST TIME ONLY)
        ratingsDF = self.sc.read.format("csv").option("header", "true").option("delimiter", ",").schema(ratingsSchema).load(ratings_set_path)
        ratingsDF.write.mode("overwrite").format("parquet").save("data_ratings.parquet")
        

        self.ratingsDF = self.sc.read.parquet("data_ratings.parquet").drop("timestamp")
        self.ratingsDF.cache()

        self.trainingDF, self.testDF = self.ratingsDF.randomSplit([0.8, 0.2], seed=12345)

        # Unique Users Id :
        # self.usersDF = ratingsDF.select("userId").distinct()

        # effectue diverses opérations de traitement des données

        self.maxIter_class= 5
        self.regParam_class=0.01
        self.rmse = None

        # entraîne le modèle en utilisant la méthode privée __train_model()
        self.__train_model()





# from pyspark import SparkContext
from pyspark.sql import SparkSession

# sc = SparkContext.getOrCreate()
sc = SparkSession.builder.getOrCreate()

engine = RecommendationEngine(sc, "Spark-movie-recommendation-main/app/ml-latest/movies.csv", "Spark-movie-recommendation-main/app/ml-latest/ratings.csv")


# Exemple d'utilisation des méthodes de la classe RecommendationEngine
# user_id = engine.create_user(None)

user_id = engine.create_user(200)


if engine.is_user_known(user_id):
    movie = engine.get_movie(None)
    print("1", movie)
    print("2", display(movie))
    ratings = engine.get_ratings_for_user(user_id)
    print("3",ratings)
    print("4", display(ratings))
    engine.add_ratings(user_id, ratings)
    print("5", movie.movieId)
    print("6", type(movie.movieId))
    id_movie = movie.select("movieId").first()[0]
    print("user_id pour prédictions",user_id)
    print("id_movie pour prédictions",id_movie)
    prediction = engine.predict_rating(user_id, id_movie)
    recommendations = engine.recommend_for_user(user_id, 10)

