from random import randrange
from pathlib import Path
from datetime import datetime
import json
import subprocess
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import gradio as gr
from stackapi import StackAPI
import pandas as pd

from .transformerclass.pred_pipeline import (
    apply_model,
    get_history,
    apply_model_by_id,
)

app = FastAPI()

_MODEL = [
    f"{model.parent.stem}/{model.stem.split('_')[0]}"
    for model in Path("/data").glob(f"**/*_score.csv")
    if model.stem.split("_")[0]
    in [
        "kerasUSE",
        # "BERT",
        # "kerasWord2Vec",
        "LogisticRegression",
        "TfidfOvRSVC",
        "TfidfOvRestSvc",
        "Word2Vec",
        "kerasWord2Vec",
    ]
]

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
    if not Path("/notebooks").exists():
        return 0
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


# @app.get("/secrets/test")
# async def test():
#     return json.loads(os.environ.get("KAGGLE_JSON"))["username"]


@app.post("/webhook/github")
async def webhook(request: Request):
    body = await request.json()
    print(body)
    if (
        body.get("head_commit", {})
        .get("message", "")
        .startswith("Kaggle Notebook | p5-nlp-Tfidf-OneVsRest |")
    ):
        print("New kaggle output data", "Starting download...", sep="\n")
        # on commit download kaggle output
        current_time = datetime.now()
        name = f"{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}"
        process = subprocess.Popen(
            f"kaggle kernels output waechter/p5-nlp-tfidf-onevsrest -p /data/{name}".split(),
            stdout=subprocess.PIPE,
        )
        output, error = process.communicate()
        if error:
            print("Error", error)
        print(output)

    if os.environ.get("GITHUB_KEY"):
        print("got sshkey")
        commits = body.get("commits", [])
        modified = [modif for commit in commits for modif in commit.get("modified", [])]
        print(modified)
        modified = [
            modif
            for modif in modified
            if modif
            in [
                "api/main.py",
                "api/transformerclass/pred_pipeline.py",
                "api/transformerclass/p5_nlp_utils.py",
            ]
        ]
        if len(modified) > 0:
            process = subprocess.Popen(
                [
                    "bash",
                    "/code/api/majapi.sh",
                    body.get("head_commit", {}).get("message", ""),
                ],
                stdout=subprocess.PIPE,
            )
            output, error = process.communicate()
            if error:
                print("Error", error)
            print(output)
    return 200


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


@app.get("/questions/{tag}")
async def get_question_by_tag(
    tag: str, version_model: str = "modelsOvR_1/LogisticRegression", get_nb: int = 5
):
    """
    Return `get_nb` (default 5) newest question id correctly tagged by model `version_model`for given `tag`
    Ex: https://domw-p5-nlp.hf.space/questions/git?version_model=USE_21tags/kerasUSE
    """
    SITE = StackAPI("stackoverflow")
    resp = SITE.fetch(
        "questions",
        order="desc",
        sort="creation",
        tagged=tag,
        site="stackoverflow",
        filter="withbody",
    )
    # print(resp)

    def chunk_emitter():
        good_res = []
        for i, question in enumerate(resp["items"], start=1):
            prediction = apply_model(
                title=question["title"],
                version_model=version_model,
                body=question["body"],
            )
            if prediction[prediction["tag"] == tag]["pred"][0]:
                good_res.append(question["question_id"])
                yield f'{len(good_res)} sur {i} {question["question_id"]} '
                # TODO compare prediction & tags
            if len(good_res) >= get_nb:
                break
        yield f"{(len(good_res)/(i)):.2%} {str(good_res)}"

    # return [
    #     {
    #         "question_id": question["question_id"],
    #         "title": question["title"],
    #         "tags": question["tags"],
    #     }
    #     for question in resp["items"]
    # ]
    return StreamingResponse(chunk_emitter())


ids_python = [
    75276194,
    75275997,
    75275646,
    75275639,
    75275543,
    75275500,
    75275434,
    75275407,
    75275321,
    75275256,
    75275230,
    75275158,
    75274973,
    75274791,
    75274704,
    75274608,
    75274505,
]
ids_javascript = [75278877, 75278828, 75278824, 75278802, 75278777]
ids_git = [75288389, 75288329, 75288143, 75287879, 75287761]
ids_java = [
    75277724,
    75277601,
    75277450,
    75277341,
    75276995,
    75276919,
    75276766,
    75276726,
    75276696,
    75276324,
    75276050,
    75275934,
    75275798,
    75275673,
    75275263,
    75274887,
    75274872,
    75274602,
    75274384,
    75274317,
    75274144,
    75273207,
    75272580,
    75272317,
    75272274,
    75272260,
    75272029,
    75271958,
    75271675,
    75271629,
    75271327,
    75270749,
    75270691,
    75270554,
    75269963,
    75269602,
    75269434,
    75269417,
    75269315,
    75269041,
    75268687,
    75268454,
    75268295,
    75267856,
    75267694,
    75267521,
    75267042,
    75266475,
    75266256,
    75265283,
    75264954,
    75264716,
    75264354,
    75263983,
    75262593,
    75262375,
    75262368,
    75262218,
    75261925,
    75261898,
    75261725,
    75261251,
    75260868,
    75260539,
    75259708,
    75259194,
    75258989,
    75258594,
    75258443,
]
exemples_ids = (
    [
        [questions_id, "modelsOvR_1/LogisticRegression", "python"]
        for questions_id in ids_python
    ]
    + [
        [questions_id, "USE_21tags/kerasUSE", "javascript"]
        for questions_id in ids_javascript
    ]
    + [
        [questions_id, "Word2Vec_1_30_8_20/kerasWord2Vec", "git"]
        for questions_id in ids_git
    ]
    + [
        [questions_id, "modelsOvR_1/LogisticRegression", "java"]
        for questions_id in ids_java
    ]
    + [
        [75275245, _MODEL[randrange(len(_MODEL) - 1)], "javascript"],  # javascript
        [74611350, _MODEL[randrange(len(_MODEL) - 1)], "docker"],  # docker
    ]
)
FLAG_OPTIONS = ["Wrong", "Incorrect", "Good"]
FLAG_DIR = Path(os.environ.get("FLAG_DIR"))
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
    examples_per_page=25,
    allow_flagging="manual",
    flagging_options=FLAG_OPTIONS,
    flagging_callback=gr.CSVLogger(),
    flagging_dir=FLAG_DIR,
    # cache_examples=True,
    live=True,
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
    allow_flagging="never",
    #     allow_flagging="manual",
    #     flagging_options=FLAG_OPTIONS,
    #     flagging_callback=gr.CSVLogger(),
    #     flagging_dir=FLAG_DIR / "text",
)

historyInterface = gr.Interface(
    fn=get_history,
    inputs=[
        gr.Textbox(value="**", label="Version"),
        gr.Textbox(value="*", label="Model"),
    ],
    outputs=[gr.Dataframe(label="synthese scores"), gr.HTML(label="scores par tag")],
    examples=[["**", "*"], ["**", "kerasUSE"]],
    allow_flagging="never",
    title="Scores",
)


def get_flags():
    # for path in FLAG_DIR.glob("Tag prédit/*"):
    #     print(path)
    return (
        pd.read_csv(FLAG_DIR / "log.csv").drop(
            ["Tag", "Question", "Tags prédit", "Tags réels"], axis="columns"
        )
        if Path(FLAG_DIR / "log.csv").is_file()
        else pd.DataFrame()
    )


flagsInterface = gr.Interface(
    fn=get_flags,
    inputs=None,
    outputs=[
        gr.Dataframe(label="Flags"),
    ],
    # examples=[[]],
    allow_flagging="never",
    title="Flags",
)

app = gr.mount_gradio_app(
    app,
    gr.TabbedInterface(
        [by_idStackOverFlow, by_text, historyInterface, flagsInterface],
        ["From StackOverFlow id", "From text", "Scores", "Flags"],
        title="p5 NLP Openclassrooms",
    ),
    path="/",
)
