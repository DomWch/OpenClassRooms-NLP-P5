from typing import Literal
from pathlib import Path
import requests
import json

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from bs4 import BeautifulSoup

from . import p5_nlp_utils


def is_the_only_string_within_a_tag(s):
    """Return True if this string is the only child of its parent tag."""
    return s == s.parent.string


def apply_model(text: str | int, model: Literal["USE"]):
    if isinstance(text, int):
        r = requests.get("https://stackoverflow.com/questions/")
        r.raise_for_status()
        soup = BeautifulSoup(r.text, features="html.parser")
        title = soup.find(id="question-header")
        body = " ".join(
            [
                elem.string
                for elem in soup.find(id="mainbar").find_all(
                    string=is_the_only_string_within_a_tag
                )
                if elem
            ]
        )
        text = f'{title.string if title else ""} {body}'.strip()
    match model:
        case "USE":
            model = "kerasUSE"
            # https://stackoverflow.com/questions/2138873/cleanest-way-to-get-last-item-from-python-iterator
            *_, path = Path("/data").glob(f"USE_*/{model}_score.csv")
            path = path.parent
            print(path)
            # with open(path / "description.json", "r") as f:
            #     description = json.loads(f.read())
            # description = {
            #     k: v
            #     for k, v in description.items()
            #     if not isinstance(v, dict)
            #     or (isinstance(v, dict) and v.get("actif", False))
            # }
            # print("Model paramétre:", description, sep="\n")
            with open(path / "best_limits_use.json", "r") as f:
                best_limits = json.loads(f.read())
            text_lemma = p5_nlp_utils.TextCleaning.cleaning_text(text)
            encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            test_sentences_encoded = encoder([text_lemma])
            pipeline = tf.keras.models.load_model(
                path / model,
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
        case _:
            return 404
