import sklearn
from sklearn.model_selection import train_test_split
import sklearn.metrics.pairwise
import numpy as np
import ast

def recomendacion_usuario(id_de_usuario):
    # Cargar los datos desde los CSVs
    user_items = pd.read_csv('user_items.csv')
    user_reviews = pd.read_csv('user_reviews.csv')
    output = pd.read_csv('output.csv')

    # Unir las tablas user_items y user_reviews basadas en la columna 'user_id'
    merged_data = pd.merge(user_items, user_reviews, how="inner")

    columnas_a_eliminar = ['funny', 'posted', 'last_edited', 'helpful', 'recommend', 'review', 'user_url']

    df = merged_data.drop(columnas_a_eliminar, axis=1)
    promedio_playtime = df['playtime_forever'].mean()


    if id_de_usuario not in user_items['user_id'].values:
        popular_items = user_items['item_id'].value_counts().index[:6]
        popular_items_details = pd.merge(pd.DataFrame({'item_id': popular_items}),
                                             output[['item_id', 'app_name']],
                                             on='item_id', how='inner')
        return popular_items_details

    else:
        distancia_chica = promedio_playtime / 3
        distancia_media = promedio_playtime - (distancia_chica * 2)

        def asignar_valor(playtime):
            if playtime <= distancia_chica:
                return 0
            elif distancia_chica < playtime <= distancia_media:
                return 1
            else:
                return 2

        # Aplica la función a la columna 'playtime forever' para crear la nueva columna 'nueva_columna'
        df['playtime'] = df['playtime_forever'].apply(asignar_valor)


        user_id_mapping = {user_id: i for i, user_id in enumerate(df['user_id'].unique())}
        df['user_id_numeric'] = df['user_id'].map(user_id_mapping)

        # Mapear item_id a identificadores numéricos únicos
        item_id_mapping = {item_id: i for i, item_id in enumerate(df['item_id'].unique())}
        df['item_id_numeric'] = df['item_id'].map(item_id_mapping)

        # Find the corresponding user_id_numeric using the mapping
        user_id_num = user_id_mapping.get(id_de_usuario, None)

        

        # Suma la columna 'playtime_forever' al 'Sentiment_analysis'
        df['combined_rating'] = df['Sentiment_analysis'] + df['playtime']
        
        # Utiliza pivot para crear una matriz de recomendación
        matriz_recomendacion = df.pivot(index='user_id_numeric', columns='item_id_numeric', values='combined_rating')

        # Llenar los valores NaN con 0
        matriz_recomendacion = matriz_recomendacion.fillna(0)

        # Se obtienen los valores de la matriz como un array
        ratings = matriz_recomendacion.values

        # Se calcula la cantidad de elementos no cero en la matriz
        sparsity = float(len(ratings.nonzero()[0]))

        # Se calcula la sparsity dividiendo la cantidad de elementos no cero entre el total de elementos en la matriz
        sparsity /= (ratings.shape[0] * ratings.shape[1])

        # Se convierte el resultado a porcentaje multiplicando por 100
        sparsity *= 100

        # División en conjuntos de entrenamiento y prueba
        ratings_train, ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)

        # Imprime las formas de los conjuntos resultantes
        print('Shape of ratings_train:', ratings_train.shape)
        print('Shape of ratings_test:', ratings_test.shape)

        sim_matrix = 1 - sklearn.metrics.pairwise.cosine_distances(ratings)

        # Número de usuarios y elementos
        num_users, num_items = ratings_train.shape

        # Número de usuarios en el conjunto de prueba
        num_test_users = ratings_test.shape[0]

        # Selecciona una parte de la matriz de similitud para el conjunto de entrenamiento
        sim_matrix_train = sim_matrix[:num_users, :num_users]

        # Selecciona una parte de la matriz de similitud para el conjunto de prueba
        sim_matrix_test = sim_matrix[num_users:num_users+num_test_users, :num_users]

        # Realiza las predicciones para el conjunto de entrenamiento
        users_predictions_train = sim_matrix_train.dot(ratings_train) / np.array([np.abs(sim_matrix_train).sum(axis=1)]).T

        # Realiza las predicciones para el conjunto de prueba
        users_predictions_test = sim_matrix_test.dot(ratings_train) / np.array([np.abs(sim_matrix_test).sum(axis=1)]).T
        
        def recommendation_for_user(user_id, n=6):
            # Obtén la fila correspondiente al usuario en la matriz de similitud
            user_similarities = sim_matrix[user_id]

            # Encuentra los índices de los usuarios más similares (excluyendo al propio usuario)
            similar_users = np.argsort(user_similarities)[::-1][1:]

            # Suma ponderada de las calificaciones de juegos de usuarios similares
            weighted_sum = np.zeros(ratings.shape[1])
            total_similarity = 0

            for similar_user in similar_users:
                if not np.isnan(sim_matrix[user_id, similar_user]):
                    weighted_sum += sim_matrix[user_id, similar_user] * ratings[similar_user]
                    total_similarity += np.abs(sim_matrix[user_id, similar_user])

            # Avoid division by zero
            if total_similarity != 0:
                # Calcula las predicciones dividiendo por la suma total de similitudes
                user_predictions = weighted_sum / total_similarity
            else:
                user_predictions = np.zeros(ratings.shape[1])


            # Encuentra los índices de los juegos no jugados por el usuario
            games_played = np.where(ratings[user_id] > 0)[0]
            games_not_played = np.where(ratings[user_id] == 0)[0]

            # Filtra los juegos ya jugados
            user_predictions = np.delete(user_predictions, games_played)

            # Obtén los índices de los juegos recomendados (los de mayor predicción)
            recommended_indices = np.argsort(user_predictions)[::-1][:n]

            # Obtén los identificadores reales de los juegos recomendados
            recommended_games = games_not_played[recommended_indices]

            # lista_de_numeros es la lista de valores de item_id_numeric que estás buscando
            lista_de_numeros = recommended_games
            # Filtrar el DataFrame para obtener las filas donde item_id_numeric está en la lista
            resultados = df[df['item_id_numeric'].isin(lista_de_numeros)]
            # Mostrar solo las columnas 'item_id_numeric' y 'user_id'
            valores_correspondientes = resultados[['item_id_numeric', 'item_id']]
            # Eliminar duplicados basados en 'item_id_numeric'
            valores_correspondientes = valores_correspondientes.drop_duplicates(subset='item_id_numeric')
    
            merged_df = pd.merge(valores_correspondientes, output[['item_id', 'app_name', 'genres']], on='item_id', how='inner')

            # Muestra los resultados
            merged_df[['item_id', 'app_name', 'genres']]

            return merged_df
            

        return recommendation_for_user (user_id_num)