FROM python:3.7.13-slim

ENV PATH /usr/local/bin:$PATH

WORKDIR /usr/src/app




#COPY . /usr/src/app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install tensorflow==1.13.1 protobuf==3.7.0 scipy==1.2.1 matplotlib==3.5.3 opencv-python==4.7.0.68  \
    numpy==1.21.6 flask==2.2.2 flask-cors==3.0.10 keras==2.0.8 pandas

# Creating the non-root user

RUN useradd -ms /bin/bash appuser

USER appuser
WORKDIR /home/appuser
