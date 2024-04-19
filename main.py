from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi import File, UploadFile
from sklearn.model_selection import train_test_split

from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import pandas as pd
from sklearn.metrics import  precision_score, recall_score, f1_score
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload")
def read_item(file: UploadFile = File(...)):
    try:
       contents = file.file.read()
       with open("file.csv", "wb") as f:
           f.write(contents)
    except Exception:
       return {"message": "Error al subir el archivo"}
    finally:
       file.file.close()
    # procesar todo
    df_og = pd.read_csv("training.csv")
    model = load("assets/model.joblib")
    df = pd.read_csv("file.csv")
    X_train, X_test, y_train, y_test = train_test_split(df_og[["Review"]], df_og["Class"], test_size=0.3, stratify=df_og["Class"], random_state=1)
    result = model.best_estimator_.predict(df["Review"])
    y_test, result = cortar(y_test, result)
    df['Class'] = result
    df.to_csv("file_predicted.csv")
    ps = precision_score(y_test, result, average="weighted")
    rs = recall_score(y_test, result, average="weighted")
    f1 = f1_score(y_test, result, average="weighted")
    return {"message": f"Archivo subido correctamente {file.filename}", "ps":ps, "rs":rs, "f1":f1}

@app.get("/download")
def download_file():
    return FileResponse(path ="file_predicted.csv", media_type="text/csv")


def cortar(arr1, arr2):
    min_length = min(len(arr1), len(arr2))
    truncated_arr1 = arr1[:min_length]
    truncated_arr2 = arr2[:min_length]
    return truncated_arr1, truncated_arr2
