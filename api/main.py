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

# from tensorflow.keras import preprocessing as keras_preprocessing
from tensorflow import keras
import tensorflow as tf

# from api.ml_class import kerasembedtransformerclass
# from api.kerasembedtransformerclass.kerasembedtransformerclass import (
#     KerasEmbedTransformer,
# )
from .kerasembedtransformerclass import kerasembedtransformerclass

# from api.ml_class.keras_class import KerasEmbedTransformer as KerasEmbedTransformerClass

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


CSS = "table{border-collapse: collapse;}"
TEMPLATE_ROOT = (
    "<html><head><style>{css}</style></head><body>{body}<body></html>"  # body
)
TEMPLATE_SCORES_ALL = "<div>Synthese f1-scores par modéle</div>{synthese}</br>{scores_html}"  # scores_syntheses, X*TEMPLATE_SCORES
TEMPLATE_SCORES = "Version: {version} Model: {model_link} {date}</br>{scores_html}</br></br>"  # version_name, link_model, datetime.fromtimestamp(score.stat().st_mtime).strftime('%A %d %B %Y %H:%M'), score_df.to_html()
TEMPLATE_LINK = "<a href='/predict/{version_name}/{model_name}'>{model_name}</a>"


@app.get("/score_history")
async def history(version: str = "**", model: str = "*"):
    rep = ""
    scoresf1 = {}
    for score in Path("/kaggle_data").glob(f"{version}/{model}_score.csv"):
        version_name = score.parent.name
        model_name = score.stem.split("_")[0]
        # TODO verif si le model exist & si joblib sklearn ou tf keras
        if (
            Path(f"/kaggle_data/{version_name}/{model_name}_model.joblib")
            .resolve()
            .exists()
            or Path(f"/kaggle_data/{version_name}/{model_name}/keras_metadata.pb")
            .resolve()
            .exists()
        ):
            model_link = TEMPLATE_LINK.format(
                version_name=version_name, model_name=model_name
            )

        else:
            model_link = model_name
        score_df = pd.read_csv(score, index_col=0)
        # @TODO filter naive identique
        scoresf1[f"{version_name}_{model_name}"] = score_df["f1-score"][
            ["micro avg", "macro avg", "weighted avg", "samples avg"]
        ]
        rep += TEMPLATE_SCORES.format(
            version=version_name,
            model_link=model_link,
            date=datetime.fromtimestamp(score.stat().st_mtime).strftime(
                "%A %d %B %Y %H:%M"
            ),
            scores_html=score_df.to_html(),
        )
    scoresf1 = pd.DataFrame(scoresf1).T.sort_values(by="micro avg", ascending=False)
    return HTMLResponse(
        TEMPLATE_ROOT.format(
            css=CSS,
            body=TEMPLATE_SCORES_ALL.format(
                synthese=scoresf1.to_html(), scores_html=rep
            ),
        )
    )


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
    actifs = {
        k: v
        for k, v in description.items()
        if isinstance(v, dict) and v.get("actif", False)
    }
    print(actifs)  #
    # TODO read type sklearn/keras from description ?
    if model in ["TfidfOvRSVC", "LogisticRegression"]:
        pipeline = joblib.load(path / f"{model}_model.joblib")
    elif model in ["kerasPipeline", "BERT"]:
        pipeline = tf.keras.models.load_model(
            path / model,
            options=tf.saved_model.LoadOptions(
                allow_partial_checkpoint=False,
                experimental_io_device=None,
                experimental_skip_checkpoint=True,
                experimental_variable_policy=None,
            ),
        )
    print(pipeline)

    # if "keras_embed_transformer" in model.named_steps:
    #     load = kerasembedtransformerclass.KerasEmbedTransformer().load(
    #         path / "keras", description["Word2Vec"]
    #     )
    #     print(load)
    #     # pipeline.named_steps[
    #     #     "keras_embed_transformer"
    #     # ] = kerasembedtransformerclass.KerasEmbedTransformer().load(
    #     #     path / "keras", description["Word2Vec"]
    #     # )
    preds = pipeline.predict(test_sentence)
    return HTMLResponse(
        pd.DataFrame(
            {
                sentence: {
                    tag_name: int(pred)
                    for tag_name, pred in zip(description["target_names"], pred_row)
                }
                for sentence, pred_row in zip(test_sentence, preds)
            }
        ).T.to_html()
    )


@app.get("/download_output/{name}")
async def download_history(name: str):
    current_time = datetime.now()
    name = f"{name}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}"
    # print(name)
    process = subprocess.Popen(
        f"kaggle kernels output waechter/p5-nlp-tfidf-onevsrest -p /kaggle_data/{name}".split(),
        stdout=subprocess.PIPE,
    )

    output, error = process.communicate()
    return {"error": error, "output": output}
