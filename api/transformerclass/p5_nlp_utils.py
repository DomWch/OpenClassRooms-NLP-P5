import os
import joblib
import progressbar
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import gensim
import pyLDAvis.gensim_models as pyLDA
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow import keras
from sklearn import (
    metrics,
    # feature_extraction,
    # linear_model,
    # model_selection,
    # preprocessing,
    # cluster,
    # decomposition,
    # multiclass,
    # svm,
    # pipeline,
    # exceptions,
    manifold,
)
from transformers import (
    AutoTokenizer,
    # TFAutoModel,
    AutoModel,
    BertTokenizer,
    TFBertModel,
    # BertConfig,
)  # BertModel
from tokenizers import BertWordPieceTokenizer
import tensorflow as tf
import tensorflow_hub as hub

##nettoyage
import spacy
from bs4 import BeautifulSoup


class TextCleaning:
    @staticmethod
    def cleaning_v1(df: pd.DataFrame()):
        """before 25/01"""
        nlp = spacy.load("en_core_web_sm")
        df = df[["Id", "Title", "Body", 0, 1, 2, 3, 4, 5]].copy()
        df["Title_clean"] = ""
        df["Code"] = ""
        df["Body_clean"] = ""
        start = time.time()

        bar = progressbar.ProgressBar(
            max_value=len(df),
            widgets=[
                progressbar.Bar("=", "[", "]"),
                " ",
                progressbar.Percentage(),
                " ",
                Commun.time_e(start),
            ],
        )
        bar.start()
        for index, row in df.iterrows():
            bar.update(index)
            #     print(index, row["Title"])
            title = nlp(row["Title"])
            row["Title_clean"] = " ".join(
                [
                    token.lemma_
                    for token in title
                    if token.pos_ in ["VERB", "NOUN", "PROPN", "ADP"]
                    and not token.is_stop
                ]
            ).lower()
            # print(row["Title_clean"])
            soup = BeautifulSoup(row["Body"], features="lxml")
            row["Code"] = " ".join([code.get_text() for code in soup.find_all("code")])
            row["Body_clean"] = " ".join(
                [
                    token.lemma_
                    for token in nlp(
                        " ".join([p.get_text() for p in soup.find_all("p")])
                    )
                    if token.pos_ in ["VERB", "NOUN", "PROPN", "ADP"]
                    and not token.is_stop
                ]
            ).lower()
            df.iloc[index] = row
        bar.finish()
        return df

    @staticmethod
    def cleaning_text(text: str, for_bert: bool = False):
        if for_bert:
            return text.strip().lower()
        nlp = spacy.load("en_core_web_sm")
        return " ".join(
            [
                token.lemma_
                for token in nlp(text)
                if token.pos_ in ["VERB", "NOUN", "PROPN", "ADP"] and not token.is_stop
            ]
        ).lower()

    @staticmethod
    def cleaning_v2(df: pd.DataFrame()):
        """return separate cleaning for BERT"""
        nlp = spacy.load("en_core_web_sm")
        df = df[["Id", "Title", "Body", 0, 1, 2, 3, 4, 5]].copy()
        df["Token_lemma"] = ""
        df["Token_BERT"] = ""
        df["code"] = ""
        start = time.time()

        bar = progressbar.ProgressBar(
            max_value=len(df),
            widgets=[
                progressbar.Bar("=", "[", "]"),
                " ",
                progressbar.Percentage(),
                " ",
                Commun.time_e(start),
            ],
        )
        bar.start()
        for index, row in df.iterrows():
            bar.update(index)
            #     print(index, row["Title"])
            title = nlp(row["Title"])
            title_clean = " ".join(
                [
                    token.lemma_
                    for token in title
                    if token.pos_ in ["VERB", "NOUN", "PROPN", "ADP"]
                    and not token.is_stop
                ]
            ).lower()
            # print(row["Title_clean"])
            soup = BeautifulSoup(row["Body"], features="lxml")
            row["code"] = " ".join([code.get_text() for code in soup.find_all("code")])
            body_clean = " ".join(
                [
                    token.lemma_
                    for token in nlp(
                        " ".join([p.get_text() for p in soup.find_all("p")])
                    )
                    if token.pos_ in ["VERB", "NOUN", "PROPN", "ADP"]
                    and not token.is_stop
                ]
            ).lower()
            row["Token_lemma"] = f"{title_clean} {body_clean}"  # test avec code?
            row[
                "Token_BERT"
            ] = f'{row["Title"].strip()} {" ".join([p.get_text().strip() for p in soup.find_all("p")])}'.lower()
            df.iloc[index] = row

        bar.finish()
        return df


class Commun:
    @staticmethod
    def time_e(start: float):
        return f"{time.time()-start:_.0f}s"

    @staticmethod
    def calcul_tsne(X, perplexity=50, n_iter=500, random_state=42) -> np.ndarray:
        tsne = manifold.TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            n_iter_without_progress=100,  # used after 250 initial iterations
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        return tsne.fit_transform(X)

    @staticmethod
    def tag_is_in(df: pd.DataFrame(), tag: str, nb_cols: int = 6):
        return eval("|".join(f'(df["{i}"] == "{tag}")' for i in range(0, nb_cols)))

    @staticmethod
    def tags_are_in(df: pd.DataFrame(), tags: list):
        return pd.DataFrame({tag: Commun.tag_is_in(df, tag) for tag in tags})

    @staticmethod
    def save_score(
        y_true, y_pred, target_names: list, name=None, zero_division=0
    ) -> pd.DataFrame():
        scores = pd.DataFrame(
            metrics.classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                zero_division=zero_division,
                output_dict=True,
            )
        ).T
        if name is not None:
            scores.to_csv(f"/kaggle/working/{name}_score.csv")
        return scores

    @staticmethod
    def visu_tsne(
        X_tsne,
        y_train_monodim,
        y_train,
        target_names,
        tag_list=["java", "python", "android", "c#"],
        maxcols=4,
    ) -> None:
        #     X_tsne : (len,2), y_train_monodim
        #     y_train : (len,nb_tags)
        #     target_names : (nb_tags)
        #     tag_list : list tags to plot in addition too all tags
        #     maxcols : max cols per row (4 is nice because with figsize shape its squared)
        nb_col = min(maxcols, max(1, len(tag_list)))
        nb_row = int(len(tag_list) / nb_col) + 1
        fig = plt.figure(figsize=(30, (30 / nb_col) * nb_row))
        ax1 = fig.add_subplot(nb_row, 1, 1)
        scatter = ax1.scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=y_train_monodim.apply(
                lambda v: (target_names + ["Non Suivi"]).index(v)
            ).values,
            cmap="coolwarm",
            alpha=0.8,
        )
        legend = ax1.legend(
            handles=scatter.legend_elements()[0],
            labels=(target_names + ["Non Suivi"]),
            loc="best",
            title="Tags",
        )
        ax1.add_artist(legend)
        ax1.set_title("Représentation tags (moins présent d'abord) tsne")
        axs = {}
        for i, tag in enumerate(tag_list):
            # print(i, 1 + nb_row, nb_col, nb_col + i + 1)
            axs[i] = fig.add_subplot(1 + nb_row, nb_col, nb_col + i + 1)
            scatter_ = axs[i].scatter(
                X_tsne[:, 0],
                X_tsne[:, 1],
                c=y_train[tag],
                cmap="coolwarm",
                alpha=0.8,
            )
            legend = axs[i].legend(
                handles=scatter_.legend_elements()[0],
                labels=["False", tag],
                loc="best",
                title="Tags",
            )
            axs[i].add_artist(legend)
            axs[i].set_title(f"tag:{tag}")
        plt.show()
        return None

    @staticmethod
    def convert_pred_to_bool(preds, limit: float = 0.5):
        return pd.DataFrame(preds).apply(lambda x: x > limit)

    @staticmethod
    def convert_pred_to_bool_by_tags(preds, limit_by_tags: dict):
        return pd.DataFrame(
            [
                {
                    tag_name: (tag_pred > tag_limit)
                    for tag_pred, tag_name, tag_limit in zip(
                        pred, limit_by_tags.keys(), limit_by_tags.values()
                    )
                }
                for pred in preds
            ]
        )

    @staticmethod
    def find_best_limit(
        X_pred,
        y_true,
        target_names,
        limits=np.linspace(0, 1, 101),
        average="micro",
        full_print=False,
    ) -> tuple:
        f1_scores = [
            Commun.save_score(
                y_true=y_true,
                y_pred=Commun.convert_pred_to_bool(X_pred, limit=limit),
                target_names=target_names,
                name=None,
            ).loc[f"{average} avg", "f1-score"]
            for limit in limits
        ]
        if full_print:
            for limit in limits:
                pred_bood = Commun.convert_pred_to_bool(X_pred, limit=limit)
                score_temp = Commun.save_score(
                    y_true=y_true,
                    y_pred=pred_bood,
                    target_names=target_names,
                    name=None,
                ).loc[["micro avg", "macro avg"], "f1-score"]
                roc_auc = metrics.roc_auc_score(
                    y_true,
                    pred_bood,
                    multi_class="ovr",
                    average="micro",
                )
                print(
                    f'{limit} f1-score: {score_temp["micro avg"]} micro {score_temp["macro avg"]} macro {roc_auc} ROC AUC score'
                )
        best = limits[np.argmax(f1_scores)]
        print(f"Meilleur f1-score {max(f1_scores):.2%} pour limit {best}")
        plt.figure(figsize=(30, 15))
        plt.plot(f1_scores)
        plt.show()
        return best, f1_scores

    @staticmethod
    def find_best_limit_by_tags(
        X_pred,
        y_true,
        target_names,
        limits=np.linspace(0, 1, 101),
    ) -> dict:
        f1_scores = pd.DataFrame(
            [
                Commun.save_score(
                    y_true=y_true,
                    y_pred=Commun.convert_pred_to_bool(X_pred, limit=limit),
                    target_names=target_names,
                    name=None,
                ).loc[target_names, "f1-score"]
                for limit in limits
            ],
            index=limits,
        )

        best = {}
        plt.figure(figsize=(30, 15))
        for target_tag in target_names:
            f1_score = f1_scores[target_tag]
            print(
                f"{target_tag} : meilleur f1-score {f1_score.max():.2%} pour limit {f1_score.idxmax()}"
            )
            best[target_tag] = (f1_score.idxmax(), f1_score.max())
            plt.plot(f1_score, label=target_tag)
        plt.legend()
        plt.show()
        return best


class Word2Vec:
    @staticmethod
    def build_Word2Vec(X_train, params):
        print("Build & train Word2Vec model ...")
        X_train_token = X_train.str.split()
        w2v_model = gensim.models.Word2Vec(
            min_count=params["Word2Vec__min_count"],
            window=params["Word2Vec__window"],
            vector_size=params["Embedding__output_dim"],
            seed=42,
            workers=1,
        )
        w2v_model.build_vocab(X_train_token)
        w2v_model.train(
            X_train_token,
            total_examples=w2v_model.corpus_count,
            epochs=params["Word2Vec__epochs"],
        )
        model_vectors = w2v_model.wv
        print("Vocabulary size: %i" % len(model_vectors.index_to_key))
        print("Word2Vec trained")
        return model_vectors

    @staticmethod
    def get_embedding(model_vectors, vocabulary, params) -> np.ndarray:
        print("Create Embedding matrix ...")
        embedding_matrix = np.asarray(
            [
                model_vectors[word]
                if word in model_vectors.index_to_key
                else np.zeros(params["Embedding__output_dim"])
                for word in vocabulary
            ]
        )
        print(embedding_matrix.shape)
        return embedding_matrix


class LDA:
    def __init__(
        self,
        tokens,
        num_topics,
        no_below=100,
        no_above=0.5,
        max_tokens=100_000,
        iterations=50,
        per_word_topics=True,
    ):
        start = time.time()
        print("Creation dictionnaire et corpus", end=" ")
        self.dictionary = gensim.corpora.dictionary.Dictionary(tokens)
        self.dictionary.filter_extremes(
            no_below=no_below, no_above=no_above, keep_n=max_tokens
        )
        self.corpus = [self.dictionary.doc2bow(text) for text in tokens]
        self.num_topics = num_topics
        print("terminé en {}".format(Commun.time_e(start)))
        print("Entrainement du model", end=" ")
        # https://radimrehurek.com/gensim/models/ldamodel.html
        self.model = gensim.models.ldamodel.LdaModel(
            self.corpus,
            num_topics=num_topics,
            random_state=42,
            id2word=self.dictionary,
            per_word_topics=per_word_topics,
            iterations=iterations,
        )
        print("terminé en {}".format(Commun.time_e(start)))

    def word_cloud_by_topics(self, nb_words=200, maxcols=3) -> None:
        start = time.time()
        nb_col = min(maxcols, max(1, self.num_topics))
        nb_row = int(self.num_topics / nb_col) + 1
        fig = plt.figure(figsize=(30, (30 / nb_col) * nb_row))
        axs = {}
        #         bar = progressbar.ProgressBar(max_value=self.num_topics,
        #             widgets=[
        #                 progressbar.Bar("=", "[", "]"),
        #                 " ",
        #                 progressbar.Percentage(),
        #                 Commun.time_e(start),
        #             ],
        #         )
        #         bar.start()
        for num_topic in range(self.num_topics):
            #             bar.update(num_topic)
            #             print("Creation WordCloud")
            axs[num_topic] = fig.add_subplot(nb_row, nb_col, num_topic + 1)
            axs[num_topic].imshow(
                WordCloud().fit_words(dict(self.model.show_topic(num_topic, nb_words)))
            )
            axs[num_topic].axis("off")
            axs[num_topic].set_title(f"#{num_topic} {self.topics_names[num_topic]}")
        #         bar.finish()
        plt.show()

    def prepare_display(self, sort_topics=False):
        # https://pyldavis.readthedocs.io/en/latest/modules/API.html#pyLDAvis.prepare
        self.display_data = pyLDA.prepare(
            self.model, self.corpus, self.dictionary, sort_topics=sort_topics
        )
        return self

    def name_topics(self, target_names):
        self.topics_names = [
            "_".join(
                [
                    f"{word}({freq:.2%})"
                    for word, freq in self.model.show_topic(num_topic, 100)
                    if word in target_names
                ]
            )
            for num_topic in range(self.num_topics)
        ]
        return self

    def predict(self, X_tokens, limit=5):
        corpus_pred = [self.dictionary.doc2bow(text) for text in X_tokens]
        pred_lda = self.model.inference(corpus_pred, collect_sstats=False)[0]
        return [
            {name: prob for name, prob in zip(self.topics_names, pred) if prob > limit}
            for pred in pred_lda
        ]

    @staticmethod
    def convert_pred(preds: list, target_names: list):
        return pd.DataFrame(
            {
                index: {
                    target_tag: sum([target_tag in topic for topic in pred.keys()]) > 0
                    for target_tag in target_names
                }
                for index, pred in enumerate(preds)
            }
        ).T


class Bert:
    @staticmethod
    def get_tokenizer(model_max_length, save_path="bert_base_uncased/"):
        if not os.path.exists(save_path):
            slow_tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", model_max_length=model_max_length
            )
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            slow_tokenizer.save_pretrained(save_path)
        # from https://keras.io/examples/nlp/text_extraction_with_bert/
        # Load the fast tokenizer from saved file
        # return BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)
        return BertWordPieceTokenizer().from_file(
            "bert_base_uncased/vocab.txt", lowercase=True
        )

    @staticmethod
    def create_bert_input(sentence: str, tokenizer, max_len: int):
        x_encoded = tokenizer.encode(sentence)
        x_encoded.truncate(max_len)
        x_encoded.pad(max_len)
        return (
            np.array(x_encoded.ids),
            np.array(x_encoded.attention_mask),
            np.array(x_encoded.type_ids),
        )

    @staticmethod
    def create_bert_inputs(sentences: list, max_len: int) -> np.ndarray:
        tokenizer = Bert.get_tokenizer(model_max_length=max_len)
        x_ids = []
        x_masks = []
        x_types = []
        for sentence in sentences:
            x_encoded = Bert.create_bert_input(sentence, tokenizer, max_len)
            x_ids.append(x_encoded[0])
            x_masks.append(x_encoded[1])
            x_types.append(x_encoded[2])
        return np.array(x_ids), np.array(x_masks), np.array(x_types)

    def build_bert_model(
        max_len: int,
        target_names: list,
        module_url="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
    ):
        bert_layer = hub.KerasLayer(module_url)
        input_word_ids = tf.keras.Input(
            shape=(max_len,), dtype=tf.int32, name="input_word_ids"
        )
        input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.Input(
            shape=(max_len,), dtype=tf.int32, name="input_type_ids"
        )

        pooled_output, sequence_output = bert_layer(
            [input_word_ids, input_mask, segment_ids]
        )
        clf_output = sequence_output[:, 0, :]
        net = tf.keras.layers.Dense(512, activation="relu")(clf_output)
        net = tf.keras.layers.Dropout(0.2)(net)
        net = tf.keras.layers.Dense(128, activation="relu")(net)
        net = tf.keras.layers.Dropout(0.2)(net)
        out = tf.keras.layers.Dense(len(target_names), activation="sigmoid")(net)

        model = tf.keras.models.Model(
            inputs=[input_word_ids, input_mask, segment_ids], outputs=out
        )
        model.compile(
            tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.BinaryCrossentropy(),  # "categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @staticmethod
    def create_bert_model(params: dict, target_names: list):
        ## BERT encoder
        encoder = TFBertModel.from_pretrained("bert-base-uncased")

        inputs = tf.keras.layers.Input(
            shape=(params["max_length"],),
            dtype=tf.int32,
        )
        embedding = encoder(inputs)[0]

        #     layerDense = tf.keras.layers.Dense(
        #         len(target_names), name="start_logit", use_bias=False
        #     )(embedding)
        #     end_layer = tf.keras.layers.Activation(tf.keras.activations.softmax)(layerDense)

        layerFlat = tf.keras.layers.Flatten()(embedding)
        #     layerGAvg = tf.keras.layers.GlobalAveragePooling1D()(embedding)
        #     layerDense = tf.keras.layers.Dense(16 * len(target_names), activation="relu")(
        #         layerGAvg
        #     )
        #     layerEnd = tf.keras.layers.Dense(
        #         len(target_names)  # , name="start_logit", use_bias=False
        #     )(layerDense)
        #     layerActivation = tf.keras.layers.Activation(tf.keras.activations.softmax)(
        #         layerGAvg
        #     )
        layerReduc = tf.keras.layers.Dense(2**10, activation="relu")(layerFlat)
        layerEnd = tf.keras.layers.Dense(len(target_names), activation="sigmoid")(
            layerReduc
        )
        model = tf.keras.Model(
            inputs=[inputs],
            outputs=[layerEnd],
        )
        #     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        #     model.compile(optimizer=optimizer, loss=[loss, loss])
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                "accuracy"
            ],  # @TODO https://keras.io/api/metrics/classification_metrics/
        )
        return model

    # autre façon: comme dans le notebook exemple des tweets:
    def BertTransformer(sentences: list, params):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")  # ou TFAutoModel
        ## (input_ids,attention_mask,token_type_ids)
        encoded_input = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=params["max_length"],
        )
        b_size = params["batch_size"]
        output = [
            ## (last_hidden_state,pooler_output)
            model(
                encoded_input["input_ids"][step : step + b_size],
                attention_mask=encoded_input["attention_mask"][step : step + b_size],
            )
            .last_hidden_state.detach()
            .numpy()
            for step in range(0, len(sentences), b_size)
        ]
        print(model.summary())
        return np.concatenate(output)


class KerasEmbedTransformer(BaseEstimator, TransformerMixin):
    def init(self, params):
        self.params = params
        return self

    def fit(self, X, y=None):
        self.embed_model, self.tokenizer = self.create_keras_model(X)
        return self

    def transform(self, X, y=None):
        x_sentences = keras.preprocessing.sequence.pad_sequences(
            self.tokenizer.texts_to_sequences(X),
            maxlen=self.params["maxlen"],
            padding="post",
        )
        embeddings = self.embed_model.predict(x_sentences)
        print("embedings shape ", embeddings.shape)
        return embeddings

    def save(self, filename):
        joblib.dump(self.tokenizer, f"{filename}.tokenizer")
        self.embed_model.save(
            f"{filename}.model"
        )  # This hack allows us to save the sklearn pipeline
        self.embed_model = None
        return self

    def load(self, filename, params):
        self.tokenizer = joblib.load(f"{filename}.tokenizer")
        self.embed_model = keras.models.load_model(f"{filename}.model")
        self.init(params)
        return self

    def create_embeding(self, word_index, model_vectors):
        vocab_size = len(word_index) + 1
        print(f"Number of unique words: {vocab_size}")
        print("Create Embedding matrix ...")
        embedding_matrix = np.zeros((vocab_size, self.params["size"]))
        i = 0
        j = 0

        for word, idx in word_index.items():
            i += 1
            if word in model_vectors.index_to_key:
                j += 1
                embedding_vector = model_vectors[word]
                if embedding_vector is not None:
                    embedding_matrix[idx] = model_vectors[word]

        word_rate = np.round(j / i, 4)
        print("Word embedding rate : ", word_rate)
        print("Embedding matrix: %s" % str(embedding_matrix.shape))
        return (embedding_matrix, vocab_size)

    def create_keras_model(self, X_train):
        print("Build & train Word2Vec model ...")
        X_train_token = X_train.str.split()
        w2v_model = gensim.models.Word2Vec(
            min_count=self.params["min_count"],
            window=self.params["window"],
            vector_size=self.params["size"],
            seed=42,
            workers=1,
        )
        w2v_model.build_vocab(X_train_token)
        w2v_model.train(
            X_train_token,
            total_examples=w2v_model.corpus_count,
            epochs=self.params["epochs"],
        )
        model_vectors = w2v_model.wv
        print("Vocabulary size: %i" % len(model_vectors.index_to_key))
        print("Word2Vec trained")

        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(X_train_token)
        embedding_matrix, vocab_size = self.create_embeding(
            tokenizer.word_index, model_vectors
        )

        word_input = keras.layers.Input(shape=(self.params["maxlen"],), dtype="float64")
        word_embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=self.params["size"],
            weights=[embedding_matrix],
            input_length=self.params["maxlen"],
        )(word_input)
        word_vec = keras.layers.GlobalAveragePooling1D()(word_embedding)
        embed_model = keras.models.Model([word_input], word_vec)
        print(embed_model.summary())

        return (embed_model, tokenizer)
