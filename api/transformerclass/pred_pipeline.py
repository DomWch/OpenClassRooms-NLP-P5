from typing import Literal
from pathlib import Path
import requests
import json
from datetime import datetime

import joblib
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from bs4 import BeautifulSoup

from stackapi import StackAPI

from . import p5_nlp_utils


def is_the_only_string_within_a_tag(s):
    """Return True if this string is the only child of its parent tag."""
    return s == s.parent.string


def apply_model(
    text: str | int,
    version_model: str
    # model: Literal["kerasUSE", "BERT", "kerasWord2Vec", "LogisticRegression"],
    # version: str = "**",
):
    if isinstance(text, int):
        SITE = StackAPI("stackoverflow")
        resp = SITE.fetch("questions/{ids}", ids=[text], filter="withbody")
        print(resp)
        text_clean = p5_nlp_utils.TextCleaning.cleaning_text(
            title=resp["items"][0]["title"], body=resp["items"][0]["body"]
        )
    else:
        text_clean = p5_nlp_utils.TextCleaning.cleaning_text(title=text)
    print(text_clean)
    version_model = version_model.split("/")  # [version, model]
    # print(version_model)
    path = Path(f"/data/{version_model[0]}").resolve()
    print(path)
    with open(path / "description.json", "r") as f:
        description = json.loads(f.read())
    # description = {
    #     k: v
    #     for k, v in description.items()
    #     if not isinstance(v, dict)
    #     or (isinstance(v, dict) and v.get("actif", False))
    # }
    # print("Model paramétre:", description, sep="\n")
    match version_model[1]:  # TODO Word2Vec, bert, LDA?
        case "kerasUSE" | "BERT" | "Word2Vec":
            if version_model[1] == "kerasUSE":
                with open(path / "best_limits_use.json", "r") as f:
                    best_limits = json.loads(f.read())
                text_lemma = text_clean[0]
                encoder = hub.load(
                    "https://tfhub.dev/google/universal-sentence-encoder/4"
                )
                test_sentences_encoded = encoder([text_lemma])
            elif version_model[1] == "BERT":
                with open(path / "best_limits_bert.json", "r") as f:
                    best_limits = json.loads(f.read())
                encoder = tf.keras.models.load_model(
                    path / "bert_base_uncased",
                    options=tf.saved_model.LoadOptions(
                        allow_partial_checkpoint=False,
                        experimental_io_device=None,
                        experimental_skip_checkpoint=True,
                        experimental_variable_policy=None,
                    ),
                )
                test_sentences_encoded = encoder([text_clean[1]])
            elif version_model[1] == "Word2Vec":
                version_model[1] = "kerasWord2Vec"
                with open(path / "best_limits_kerasword2vec.json", "r") as f:
                    best_limits = json.loads(f.read())
                test_sentences_encoded = [text_clean[0]]

            pipeline = tf.keras.models.load_model(
                path / version_model[1],
                options=tf.saved_model.LoadOptions(
                    allow_partial_checkpoint=False,
                    experimental_io_device=None,
                    experimental_skip_checkpoint=True,
                    experimental_variable_policy=None,
                ),
            )
            preds = pipeline.predict(test_sentences_encoded)
            return pd.DataFrame(
                {
                    tag_name: {
                        "tag": tag_name,
                        "pred": pred > limit[0],
                        "pred_proba": pred,
                        "limit": limit[0],
                    }
                    for pred, tag_name, limit in zip(
                        preds[0], best_limits.keys(), best_limits.values()
                    )
                }
            ).T.sort_values(by="pred_proba", ascending=False)
        case "TfidfOvRSVC" | "LogisticRegression" | "TfidfOvRestSvc":
            pipeline = joblib.load(path / f"{version_model[1]}_model.joblib")
            preds = pipeline.predict(text_clean[0].split())
            return pd.DataFrame(
                {
                    tag_name: {"tag": tag_name, "pred": pred}
                    for pred, tag_name in zip(preds[0], description["target_names"])
                }
            ).T.sort_values(by="pred", ascending=False)
        case _:
            return 404


TEMPLATE_SCORES = "Version: {version} Model: {model_link} {date}</br>{scores_html}</br></br>"  # version_name, link_model, datetime.fromtimestamp(score.stat().st_mtime).strftime('%A %d %B %Y %H:%M'), score_df.to_html()
TEMPLATE_LINK = "<a href='/predict/{version_name}/{model_name}'>{model_name}</a>"


def get_history(version: str = "**", model: str = "*") -> tuple:
    """return (synthese scores: pd.DataFrame, tableaux html score pour chaque tag: str)"""
    rep = ""
    scoresf1 = {}
    for score in Path("/data").glob(f"{version}/{model}_score.csv"):
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
    return scoresf1.reset_index(names="Model"), rep
