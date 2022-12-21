from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import pandas as pd
import sklearn

# from kaggle.api.kaggle_api_extended import KaggleApi
import kaggle
import subprocess

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/score_history")
async def history(version: str = "**", model: str = "*"):
    # scores = pd.concat([pd.read_csv(score) for score in Path("/kaggle_data").glob(f"{version}/*_score.csv")])
    # return HTMLResponse(scores.to_html())
    rep = ""
    for score in Path("/kaggle_data").glob(f"{version}/{model}_score.csv"):
        rep += f"{score.parent.name} {score.stem}</br>{pd.read_csv(score).to_html()}</br></br>"
    return HTMLResponse(rep)


@app.get("/predict/{version}/{model}")
async def predict(version: str, model: str):
    test_sentence = ["git head main"]
    pipeline = joblib.load(Path(f"/kaggle_data/{version}/{model}_model.joblib"))
    print(pipeline)
    # print(pipeline.predict(["git head main"]))
    target_names = [
        "java",
        "c#",
        "javascript",
        "python",
        "android",
        "c++",
        "ios",
        "html",
        "php",
        ".net",
        "jquery",
        "css",
        "objective-c",
        "c",
        "sql",
        "iphone",
        "asp.net",
        "mysql",
        "linux",
        "node.js",
        "git",
    ]
    preds = pipeline.predict(test_sentence)
    # res = dict(zip(target_names, preds[0]))
    # print(res)
    return {
        sentence: {
            tag_name: int(pred) for tag_name, pred in zip(target_names, pred_row)
        }
        for sentence, pred_row in zip(test_sentence, preds)
    }


@app.get("/download_history/{name}")
async def download_history(name: str):
    # @TODO controle sur name
    process = subprocess.Popen(
        f"kaggle kernels output waechter/p5-nlp-tfidf-onevsrest -p /kaggle_data/{name}".split(),
        stdout=subprocess.PIPE,
    )

    output, error = process.communicate()
    return {"error": error, "output": output}
