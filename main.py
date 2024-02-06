import funciones
from funciones import *
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get(path = "/developer",
    description = """
    Esta función toma como parámetro de entrada el nombre de un desarrollador y devuelve
    la Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora 
""",
    tags = ["Función Developer"])
def developer(desarrollador: str):
    resultadodeveloper = funciones.developer(desarrollador)
    return resultadodeveloper

Developer = developer('Valve')
print(Developer)


@app.get("/userdata",
         description="Esta función toma como parámetro de entrada el ID de usuario y devuelve información sobre su actividad en la plataforma.",
         tags=["Función Userdata"])
def userdata_endpoint(User_id: str):
    return userdata(User_id)


@app.get("/user_for_genre",
         description="Esta función toma como parámetro de entrada el género específico y devuelve información sobre el usuario con más horas jugadas en ese género y la acumulación de horas jugadas por año.",
         tags=["Función UserForGenre"])
def user_for_genre_endpoint(genero_especificado: str):
    return UserForGenre(genero_especificado)


@app.get("/best_developer_year",
         description="Esta función toma como parámetro de entrada el año y devuelve el top 3 de desarrolladores con la mayor cantidad de juegos recomendados en ese año.",
         tags=["Función BestDeveloperYear"])
def best_developer_year_endpoint(anio: int):
    return best_developer_year(anio)


@app.get("/developer_reviews_analysis",
         description="Esta función toma como parámetro de entrada el nombre del desarrollador y devuelve el análisis de sentimiento de las reviews de sus juegos.",
         tags=["Función DeveloperReviewsAnalysis"])
def developer_reviews_analysis_endpoint(desarrolladora: str):
    return developer_reviews_analysis(desarrolladora)
