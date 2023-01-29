from random import randrange
from pathlib import Path
from datetime import datetime
import json
import subprocess
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import gradio as gr

from api.transformerclass.pred_pipeline import apply_model, get_history

app = FastAPI()


CSS = "table{border-collapse: collapse;}"
TEMPLATE_ROOT = (
    "<html><head><style>{css}</style></head><body>{body}<body></html>"  # body
)
TEMPLATE_SCORES_ALL = "<div>Synthese f1-scores par modéle</div>{synthese}</br>{scores_html}"  # scores_syntheses, X*TEMPLATE_SCORES


@app.get("/score_history")
async def history(version: str = "**", model: str = "*"):
    scoresf1, rep = get_history(version, model)
    return HTMLResponse(
        TEMPLATE_ROOT.format(
            css=CSS,
            body=TEMPLATE_SCORES_ALL.format(
                synthese=scoresf1.to_html(), scores_html=rep
            ),
        )
    )


@app.get("/download_output/{name}")
async def download_history(name: str):
    current_time = datetime.now()
    name = f"{name}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}"
    # print(name)
    process = subprocess.Popen(
        f"kaggle kernels output waechter/p5-nlp-tfidf-onevsrest -p /data/{name}".split(),
        stdout=subprocess.PIPE,
    )

    output, error = process.communicate()
    return {"error": error, "output": output}


_MODEL = [
    f"{model.parent.stem}/{model.stem.split('_')[0]}"
    for model in Path("/data").glob(f"**/*_score.csv")
    if model.stem.split("_")[0]
    in [
        "kerasUSE",
        "BERT",
        "kerasWord2Vec",
        "LogisticRegression",
        "TfidfOvRSVC",
        "TfidfOvRestSvc",
        "Word2Vec",
    ]
]

examples = [
    ["git head main merge rebase", _MODEL[randrange(len(_MODEL) - 1)]],
    ["java object class main", _MODEL[randrange(len(_MODEL) - 1)]],
    ["python pandas numpy", _MODEL[randrange(len(_MODEL) - 1)]],
    ["request database join merge", _MODEL[randrange(len(_MODEL) - 1)]],
]
by_text = gr.Interface(
    fn=apply_model,
    inputs=["textbox", gr.Dropdown(_MODEL, label="Model")],
    outputs=[gr.Dataframe(label="Tag prédit")],
    examples=examples,
    title="From text",
)

by_idStackOverFlow = gr.Interface(
    fn=apply_model,
    inputs=[
        gr.Number(precision=0, label="StackOverFlow ID"),
        gr.Dropdown(_MODEL, label="Model"),
    ],
    outputs=[gr.Dataframe(label="Tag prédit")],
    examples=[[74611350, _MODEL[randrange(len(_MODEL) - 1)]]],
    title="From StackOverFlow",
    description="NLP to predict tags from StackOverFlow questions",
)

historyInterface = gr.Interface(
    fn=get_history,
    inputs=[
        gr.Textbox(value="**", label="Version"),
        gr.Textbox(value="*", label="Model"),
    ],
    outputs=[gr.Dataframe(label="synthese scores"), gr.HTML(label="scores par tag")],
    examples=[["**", "*"], ["**", "kerasUSE"]],
    title="Scores",
)

app = gr.mount_gradio_app(
    app,
    gr.TabbedInterface(
        [by_text, by_idStackOverFlow, historyInterface],
        ["From text", "From StackOverFlow", "Scores"],
        title="p5 NLP Openclassrooms",
    ),
    path="/",
)
