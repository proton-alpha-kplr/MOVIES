# Utilisez l'image de base officielle de Spark
FROM docker.io/apache/spark:3.4.0

# Installez Python et les dépendances nécessaires
RUN apt-get update && apt-get install -y python3 python3-pip

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez le code source de l'application Flask dans le conteneur
COPY /app /app

# Copiez le fichier requirements.txt dans le conteneur
COPY requirements.txt /app

# COPY /app/ml-latest /ml-latest

# Installez les dépendances Python
RUN pip3 install -r requirements.txt

# Exposez le port utilisé par votre application Flask
# EXPOSE 5432

# Lancement du CMD
CMD ["spark-submit", "server.py", "ml-latest/movies.csv", "ml-latest/ratings.csv"]
# CMD ["python3", "server.py"]
