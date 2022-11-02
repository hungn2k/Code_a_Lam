FROM python:3.6.15

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/code/store_manager

RUN mkdir /code
WORKDIR /code/store_manager

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt .
RUN pip install -r requirements.txt && rm -f requirements.txt
COPY ./docker_code /code/store_manager
 
CMD ["python", "main_process.py"]