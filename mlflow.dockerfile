FROM python:3.10-bullseye
# RUN set -eux; \
#     groupadd -r user -g 2012; \
#     useradd -r -g 2012 -u 2012 --home-dir=/home/user --shell=/bin/bash user; \
# mkdir -p /home/user; chown user:user /home/user; \
# mkdir /mlruns_data; chmod 750 /mlruns_data; \
# mkdir /kaggle_data; chmod 750 /kaggle_data;
# COPY --chown=user ./kaggle/cred/ /home/user/.kaggle
COPY ./kaggle/.cred/ /root/.kaggle
RUN pip install --upgrade pip
# Install python packages
RUN pip install mlflow pymysql kaggle