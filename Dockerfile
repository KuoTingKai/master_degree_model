# FROM ubuntu:22.10
FROM deeplabcut/deeplabcut:2.2.1.1-core-cuda11.7.0-runtime-ubuntu20.04

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip &&\
    pip install flask &&\
    pip install joblib==1.2.0 &&\
    pip install scikit-learn==0.22.1 &&\
    apt-get update
    # sed -i '43s#C:/Users/Kevin/Desktop/app/pose_model/0611_depth_2021_06_12#/app/pose_model/0611_depth_2021_06_12#' /app/app.py &&\
    # sed -i '9s#C:/Users/Kevin/Desktop/app/pose_model/0611_depth_2021_06_12#/app/pose_model/0611_depth_2021_06_12#' /app/pose_model/0611_depth_2021_06_12/config.yaml

CMD python3 app.py