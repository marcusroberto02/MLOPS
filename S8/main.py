import numpy as np
from sklearn import datasets
from http import HTTPStatus
from fastapi import FastAPI, BackgroundTasks
import pickle
import time
import pandas as pd
import os

classes =  ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def add_to_database(sepal_length: float, 
                    sepal_width: float, 
                    petal_length: float, 
                    petal_width: float, 
                    prediction: int, 
                    time=time.time(), 
                    message=""):
    with open("database.csv", mode="a") as file:
        if os.stat('database.csv').st_size == 0:
            file.write("time, sepal_length, sepal_width, petal_length, petal_width, prediction\n")
        content = f"{time}, {sepal_length}, {sepal_width}, {petal_length}, {petal_width}, {prediction}\n"
        file.write(content)

app = FastAPI()


@app.get("/")
def read_root():
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.post("/iris_v1/")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    input_data = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = model.predict([input_data]).item()
    return {"prediction": classes[prediction], "prediction_int": prediction}

@app.post("/iris_v2/")
async def predict_v2(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float, background_tasks: BackgroundTasks):
    input_data = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = model.predict([input_data]).item()

    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    background_tasks.add_task(add_to_database,sepal_length,sepal_width,petal_length,petal_width,prediction,time=t)

    return {"prediction": classes[prediction], "prediction_int": prediction}
