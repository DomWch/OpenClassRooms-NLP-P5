FROM python:3.10-bullseye
RUN pip install --upgrade pip
# RUN pip install 
# TODO RUN pip install -r requirements.txt
RUN pip install fastapi "uvicorn[standard]" joblib

COPY ./kaggle/.cred/ /root/.kaggle
WORKDIR /code
COPY ./api /code/api
# separate line to keep cache
RUN pip install --upgrade pip
RUN pip install fastapi "uvicorn[standard]" joblib
RUN pip install pandas scikit-learn==1.0.2 tensorflow gensim joblib kaggle "tensorflow>=2.0.0" 
RUN pip install --upgrade tensorflow-hub
RUN pip install gradio
RUN pip install spicy beautifulsoup4
RUN pip install progressbar wordcloud
RUN pip install transformers tokenizers
RUN pip install bs4
RUN pip install spacy 
RUN python -m spacy download en_core_web_sm
RUN export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN pip install pyLDAvis
RUN pip install stackapi
# RUN uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

