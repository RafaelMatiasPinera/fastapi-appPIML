# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)` Rafael Piñera**</h1>

-Steam pide que te encargues de crear un sistema de recomendación de videojuegos para usuarios-
-Se requieren funciones de filtrado, EDA, modelo de sentient analysis NLP y modelo de ML. 

##PRIMERO:
Comienzo cargando los datasets correspondientes.
-Los Data Sets originales se encuentran en archivos JSON, que por su peso no van a ser cargados a la web.-
Estos son: a- output_steam_games; b- australian_users; c- australian_items.
Procedo a la limpieza de los datos (borrar duplicados, eliminar los Null correspondientes)
Generando así un DataFrame y luego un CSV para cada set.
user_items.csv - user_reviews.csv - output.csv
A user_reviews.csv, le agregaré también la columna Sentiment Analysis, aplicando análisis de sentimiento con NLP con la siguiente escala: debe tomar el valor '0' si es malo, '1' si es neutral y '2' si es positivo.

##SEGUNDO
Defino las funciones:

1 def developer( desarrollador: str ):
Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.
2 def userdata( User_id: str ):
Cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
3 def UserForGenre( genero: str ):
El usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
4 def best_developer_year( año: int ):
Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado.
5 def developer_reviews_analysis( desarrolladora: str ):
Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.
También investigo en YouTube sobre el algoritmo que puedo usar para el modelo de ML.

##TERCERO
Realizo un EDA (Exploratory Data Analysis).
Que será guardado en un archivo aparte llamado "EDA.ipynb".

##CUARTO
Para el modelo de Machine Learning,
Usaré el modelo de user-item, aplicando un filtro colaborativo.
Realizaré el preprocesamiento de los datos y definiré algunas métricas.
Dividiré en Train-y-Test y entrenaré el modelo.
Una vez entrenado, guardaré las variables del modelo usando una librería llamada pickle.
Esto último para no tener que entrenarlo cada vez y ahorrar memoria para la parte del deploy.
Además, para cuando se ingrese un usuario que no figura en la base de datos, se tomará como si fuese un usuario nuevo.
A esto último se le llama cold-start, y definí para esto que se le recomienden los juegos más populares.
6 def **recomendacion_usuario( *`id de usuario`* )**:
    Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.

##QUINTO
Paso las funciones en limpio y las defino junto con la de ML en el archivo "funciones.py".
Instalo FastAPI.
Creo una cuenta en Render.
Creo el archivo main que hace las consultas con FastAPI.

POR ULTIMO
-Hice el deploy con Render. Se puede observar ingresando a este link:
https://fast-api-pi1-ml-a41b.onrender.com/docs#/
-video explicando el proyecto:
