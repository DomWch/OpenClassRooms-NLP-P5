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

from api.transformerclass.pred_pipeline import (
    apply_model,
    get_history,
    apply_model_by_id,
)

app = FastAPI()


CSS = "table{border-collapse: collapse;}"
TEMPLATE_ROOT = (
    "<html><head><style>{css}</style></head><body>{body}<body></html>"  # body
)
TEMPLATE_SCORES_ALL = "<div>Synthese f1-scores par modéle</div>{synthese}</br>{scores_html}"  # scores_syntheses, X*TEMPLATE_SCORES


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/merge_notebook")
async def merge_notebook():
    tfidf = read("/notebooks/p5-nlp-tfidf-onevsrest.ipynb")
    word2vec = read("/notebooks/p5-nlp-word2veckeras.ipynb")
    bert = read("/notebooks/p5-nlp-bert.ipynb")
    use = read("/notebooks/p5-nlp-use.ipynb")
    lda = read("/notebooks/p5-nlp-lda.ipynb")

    def pick_cell_generator():
        part = 0
        for tfidf_cell, word2vec_cell, bert_cell, use_cell, lda_cell in zip(
            tfidf["cells"], word2vec["cells"], bert["cells"], use["cells"], lda["cells"]
        ):
            if tfidf_cell["cell_type"] == "code":
                max_output = max(
                    [
                        len(tfidf_cell.get("outputs", [])),
                        len(word2vec_cell.get("outputs", [])),
                        len(bert_cell.get("outputs", [])),
                        len(use_cell.get("outputs", [])),
                        len(lda_cell.get("outputs", [])),
                    ]
                )
                if max_output > 1:
                    if max_output == len(word2vec_cell.get("outputs", [])):
                        res = word2vec_cell
                        part = 1
                    elif max_output == len(bert_cell.get("outputs", [])):
                        res = bert_cell
                        part = 2
                    elif max_output == len(use_cell.get("outputs", [])):
                        res = use_cell
                        part = 3
                    elif max_output == len(lda_cell.get("outputs", [])):
                        res = lda_cell
                        part = 4
                else:
                    res = [tfidf_cell, word2vec_cell, bert_cell, use_cell, lda_cell][
                        part
                    ]
            else:
                res = [tfidf_cell, word2vec_cell, bert_cell, use_cell, lda_cell][part]
            yield res

    tfidf["cells"] = [test for test in pick_cell_generator()]
    with open(
        "/notebooks/Waechter_Dominique_2_notebook_test_122022.ipynb", "w"
    ) as outfile:
        json.dump(tfidf, outfile)
    return 200, "Ok"


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
exemples_ids = [
    [75275230, "modelsOvR_1/LogisticRegression", "python"],  # python #good
    [75275227, "modelsOvR_1/LogisticRegression", "python"],  # python
    [75275209, _MODEL[randrange(len(_MODEL) - 1)], "python"],  # python
    [75275263, _MODEL[randrange(len(_MODEL) - 1)], "java"],  # java #good
    [75275245, _MODEL[randrange(len(_MODEL) - 1)], "javascript"],  # javascript
    [74611350, _MODEL[randrange(len(_MODEL) - 1)], "docker"],  # docker
]
by_idStackOverFlow = gr.Interface(
    fn=apply_model_by_id,
    inputs=[
        gr.Number(precision=0, label="StackOverFlow ID"),
        gr.Dropdown(_MODEL, label="Model"),
        gr.Textbox(interactive=False, label="Tag"),
    ],
    outputs=[
        gr.HTML(label="Question"),
        gr.Dataframe(label="Tags prédit"),
        gr.JSON(label="Tags réels"),
    ],
    examples=exemples_ids,
    title="From StackOverFlow",
    description="Predict tags from StackOverFlow questions</br>\
    [newest python questions](https://stackoverflow.com/questions/tagged/python?tab=Newest)",
)

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
        [by_idStackOverFlow, by_text, historyInterface],
        ["From StackOverFlow id", "From text", "Scores"],
        title="p5 NLP Openclassrooms",
    ),
    path="/",
)
