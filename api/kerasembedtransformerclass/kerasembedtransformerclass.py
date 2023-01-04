import joblib
import numpy as np
import gensim
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


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
        print(f"{filename}.tokenizer")
        self.embed_model = keras.models.load_model(f"{filename}.model")
        self.tokenizer = joblib.load(f"{filename}.tokenizer")
        # self.tokenizer = joblib.load(1f"{filename}.tokenizer")
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
