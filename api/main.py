from pathlib import Path
from datetime import datetime
import json
import subprocess
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import pandas as pd
import sklearn

from api.ml_class.keras_class import KerasEmbedTransformer

# from api.ml_class.keras_class import KerasEmbedTransformer as KerasEmbedTransformerClass

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/score_history")
async def history(version: str = "**", model: str = "*"):
    rep = "<style>table{border-collapse: collapse;}</style>"
    for score in Path("/kaggle_data").glob(f"{version}/{model}_score.csv"):
        version_name = score.parent.name
        model_name = score.stem.split("_")[0]
        rep += f"Version: {version_name} Model: <a href='/predict/{version_name}/{model_name}'>{model_name}</a>{datetime.fromtimestamp(score.stat().st_mtime).strftime('%A %d %B %Y %H:%M')}</br>{pd.read_csv(score).to_html()}</br></br>"
    return HTMLResponse(rep)


@app.get("/predict/{version}/{model}")
async def predict(version: str, model: str):
    test_sentence = [
        "git head main merge rebase",  # git
        "java object class main",  # java
        "python pandas numpy",  # python
        "request database join merge",  # sql
        # "",
    ]
    path = Path(f"/kaggle_data/{version}").resolve()
    with open(path / "description.json", "r") as f:
        description = json.loads(f.read())
    print("ici")
    # KerasEmbedTransformer = KerasEmbedTransformerClass().init(description["Word2Vec"])
    pipeline = joblib.load(path / f"{model}_model.joblib")
    print("la")
    print(pipeline)

    if "keras_embed_transformer" in pipeline.named_steps:
        pipeline.named_steps["keras_embed_transformer"].load(
            path / "keras", description["Word2Vec"]
        )
    preds = pipeline.predict(test_sentence)
    return {
        sentence: {
            tag_name: int(pred)
            for tag_name, pred in zip(description["target_names"], pred_row)
        }
        for sentence, pred_row in zip(test_sentence, preds)
    }


@app.get("/download_history/{name}")
async def download_history(name: str):
    # @TODO controle sur name
    # os.mkdir(f"/kaggle_data/{name}", mode=0o755)
    process = subprocess.Popen(
        f"kaggle kernels output waechter/p5-nlp-tfidf-onevsrest -p /kaggle_data/{name}".split(),
        stdout=subprocess.PIPE,
    )

    output, error = process.communicate()
    return {"error": error, "output": output}
