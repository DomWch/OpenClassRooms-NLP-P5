FROM python:3.10-bullseye
RUN pip install --upgrade pip
# RUN pip install 
# TODO RUN pip install -r requirements.txt
RUN pip install fastapi "uvicorn[standard]" joblib

COPY ./kaggle/.cred/ /root/.kaggle
WORKDIR /code
COPY ./api /code/api
# todo separate install to keep cache
RUN pip install pandas scikit-learn==1.0.2 tensorflow gensim joblib kaggle "tensorflow>=2.0.0" 
RUN pip install --upgrade tensorflow-hub
# RUN uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

