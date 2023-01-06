# %% [code]
import os
import joblib
import progressbar
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import gensim
import pyLDAvis.gensim as pyLDA
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
    TFAutoModel,
    AutoModel,
    BertTokenizer,
    TFBertModel,
    BertConfig,
)  # BertModel
from tokenizers import BertWordPieceTokenizer


class Commun:
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
        fig = plt.figure(figsize=(40, 10 * nb_row))
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
    def time_e(start: float):
        return f"{time.time()-start:_.0f}s"


class Word2Vec:
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
    def __init__(self, tokens, num_topics):
        start = time.time()
        print("Creation dictionnaire et corpus", end=' ')
        self.dictionary = gensim.corpora.dictionary.Dictionary(tokens)
        self.corpus = [self.dictionary.doc2bow(text) for text in tokens]
        self.num_topics = num_topics
        print("terminé en {}".format(Commun.time_e(start)))
        print("Entrainement du model", end=' ')
        self.model = gensim.models.ldamodel.LdaModel(
            self.corpus,
            num_topics=num_topics,
            random_state=42,
            id2word=self.dictionary,
        )
        print("terminé en {}".format(Commun.time_e(start)))

    def word_cloud_by_topics(self, nb_words=200, maxcols=4) -> None:
        start = time.time()
        nb_col = min(maxcols, max(1, self.num_topics))
        nb_row = int(self.num_topics / nb_col) + 1
        fig = plt.figure(figsize=(40, 10 * nb_row))
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
            axs[num_topic].set_title(f"Topic #{num_topic}")
#         bar.finish()
        plt.show()

    def prepare_display(self, sort_topics=False):
        #https://pyldavis.readthedocs.io/en/latest/modules/API.html#pyLDAvis.prepare
        self.display_data=pyLDA.prepare(
            self.model, self.corpus, self.dictionary, sort_topics=sort_topics
        )
        return self


# class Bert():
#     @staticmethod
#     def get_tokenizer(model_max_length,save_path = "bert_base_uncased/"):
#         if not os.path.exists(save_path):
#             slow_tokenizer = BertTokenizer.from_pretrained(
#                 "bert-base-uncased", model_max_length=model_max_length
#             )
#             try:
#                 os.makedirs(save_path)
#             except OSError:
#                 pass
#         slow_tokenizer.save_pretrained(save_path)
#         # from https://keras.io/examples/nlp/text_extraction_with_bert/
#         # Load the fast tokenizer from saved file
#         return BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)


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
