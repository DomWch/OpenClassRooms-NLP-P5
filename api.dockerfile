FROM python:3.10-bullseye
RUN pip install --upgrade pip
# RUN pip install 
# RUN pip install -r requirements.txt
RUN pip install fastapi "uvicorn[standard]"


WORKDIR /code
COPY ./api /code/api
# RUN pip install -e .

# RUN uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

