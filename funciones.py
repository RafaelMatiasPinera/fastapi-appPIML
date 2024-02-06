import pandas as pd
import numpy as np
import ast


def developer(desarrollador: str):
    # obtener los datos del csv
    dataframe = pd.read_csv('output.csv')
    
    # Filtra el DataFrame para el desarrollador específico
    filtered_df = dataframe[dataframe['publisher'] == desarrollador]
    
    # Convierte la columna 'release_date' a tipo datetime
    filtered_df['release_date'] = pd.to_datetime(filtered_df['release_date'], errors='coerce')
    
    # Rellena los valores NaN en la columna 'price' con 0
    filtered_df['price'].fillna(0, inplace=True)
    
    # Crea una nueva columna 'Años' basada en la fecha de lanzamiento
    filtered_df['Años'] = filtered_df['release_date'].dt.year
    
    # Agrupa por año y cuenta la cantidad total de elementos y la cantidad de elementos gratuitos para cada año
    grouped_df = filtered_df.groupby('Años').agg({'item_id': 'count', 'price': lambda x: (x == 0).sum()})
    
    # Renombra las columnas
    grouped_df.rename(columns={'item_id': 'Cantidad de Items', 'price': 'Contenido'}, inplace=True)
    
    # Calcula el porcentaje de contenido gratuito
    grouped_df['Contenido Free'] = (grouped_df['Contenido'] / grouped_df['Cantidad de Items']) * 100
    
    # Elimina la columna 'Contenido'
    grouped_df.drop(columns=['Contenido'], inplace=True)
    
    # Restablece el índice para obtener 'Año' como una columna
    grouped_df.reset_index(inplace=True)
    
    # Verifica si hay valores NaN en la columna 'Cantidad de Items'
    if grouped_df['Cantidad de Items'].isnull().any():
        raise ValueError("Cantidad de Items no puede ser NaN")
    
    # Convierte el DataFrame a un dict
    data = grouped_df.to_dict(orient='records')
    
    return data



def userdata(User_id):
    # Cargar los Datos
    df_output = pd.read_csv('output.csv')
    df_items = pd.read_csv('user_items.csv')
    df_reviews = pd.read_csv('user_reviews.csv')

    # Fusión de Datos
    df_merged = pd.merge(df_output, df_items, on='item_id')

    # Filtrar datos por el User_id proporcionado
    user_data = df_merged[df_merged['user_id'] == User_id]

    # Calcular la cantidad de dinero gastado por el usuario
    total_spent = (user_data['playtime_forever'] * user_data['price']).sum()

    # Calcular el porcentaje de recomendación en base a reviews.recommend
    recommend_percentage = df_reviews[df_reviews['user_id'] == User_id]['recommend'].mean() * 100

    # Calcular la cantidad de items
    num_items = len(user_data)

    # Construir el Resultado en el Formato Especificado
    resultado = {
        "Usuario": User_id,
        "Dinero gastado": f"{total_spent:.2f} USD",
        "porcentaje de recomendación": f"{recommend_percentage:.2f}%",
        "Cantidad de items": num_items
    }
    return resultado




def UserForGenre(genero_especificado: str):
    # Cargar los Datos
    df_games = pd.read_csv('output.csv')
    df_items = pd.read_csv('user_items.csv')

    # Verificar y eliminar filas con valores NaN en la columna 'genres'
    df_games = df_games.dropna(subset=['genres'])

    # Convertir las listas de texto a listas de Python usando ast.literal_eval
    df_games['genres'] = df_games['genres'].apply(ast.literal_eval)

    # Filtrar por Género
    df_filtered_games = df_games[df_games['genres'].apply(lambda genres: genero_especificado in genres)]

    # Fusión de Datos
    df_merged = pd.merge(df_filtered_games, df_items, on='item_id')

    # Agrupación y Cálculos
    usuario_mas_horas = df_merged.groupby('user_id')['playtime_forever'].sum().idxmax()
    
    # Convertir la columna 'release_date' a objetos de fecha
    df_merged['release_date'] = pd.to_datetime(df_merged['release_date'], errors='coerce', infer_datetime_format=True)
    
    # Filtrar filas con fechas no válidas
    df_merged = df_merged.dropna(subset=['release_date'])

    # Agrupar por año y calcular las horas jugadas
    acumulacion_por_anio = df_merged.groupby(df_merged['release_date'].dt.year)['playtime_forever'].sum().reset_index()

    # Obtener el usuario con más horas jugadas para el género dado
    top_user = df_merged[df_merged['playtime_forever'] == df_merged.groupby('user_id')['playtime_forever'].transform('max')]['user_id'].iloc[0]

    # Construir el Resultado en el Formato Especificado
    resultado = {
        "Usuario con más horas jugadas para el género dado": top_user,
        "Acumulación de horas jugadas por año": [{"Año": int(anio), "Horas": int(horas)} for anio, horas in zip(acumulacion_por_anio['release_date'], acumulacion_por_anio['playtime_forever'])]
    }

    return resultado



def best_developer_year(anio: int):
    # Cargar los DataFrames
    df_reviews = pd.read_csv('user_reviews.csv', dtype={'Sentiment_analysis': str})
    df_output = pd.read_csv('output.csv')

    # Convertir 'recommend' y 'Sentiment_analysis' a tipo booleano después de la lectura
    df_reviews['recommend'] = df_reviews['recommend'].astype(bool)
    
    # Fusionar DataFrames usando 'item_id'
    df_merged = pd.merge(df_reviews, df_output[['item_id', 'publisher', 'release_date']], on='item_id')

    # Convertir 'release_date' a datetime
    df_merged['release_date'] = pd.to_datetime(df_merged['release_date'], errors='coerce')

    # Filtrar por el año dado
    df_filtered = df_merged[df_merged['release_date'].dt.year == anio]

    # Filtrar por reviews positivas y recomendadas
    df_filtered = df_filtered[(df_filtered['recommend'] == True) & (df_filtered['Sentiment_analysis'] == '2')]

    # Contar el número de juegos recomendados por cada desarrollador
    developer_counts = df_filtered.groupby('publisher')['item_id'].nunique()

    # Obtener el top 3 de desarrolladores
    top_developers = developer_counts.nlargest(3)

    # Construir el resultado en el formato especificado
    resultado = [{"Puesto {}".format(i+1): developer} for i, developer in enumerate(top_developers.index)]

    return resultado



def developer_reviews_analysis(desarrolladora):
    # Cargar los DataFrames
    df_reviews = pd.read_csv('user_reviews.csv', dtype={'Sentiment_analysis': str})
    df_output = pd.read_csv('output.csv')

    # Fusionar DataFrames usando 'item_id'
    df_merged = pd.merge(df_reviews, df_output[['item_id', 'developer']], on='item_id')

    # Filtrar por el desarrollador dado
    df_filtered = df_merged[df_merged['developer'] == desarrolladora]

    # Contar la cantidad de registros con análisis de sentimiento positivo y negativo
    sentiment_counts = df_filtered['Sentiment_analysis'].value_counts()

    # Construir el resultado en el formato especificado
    resultado = {desarrolladora: {'Negative': sentiment_counts.get('0', 0), 'Positive': sentiment_counts.get('2', 0)}}

    return resultado

