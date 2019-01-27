FROM tensorflow/tensorflow
RUN pip install keras
RUN pip install click

WORKDIR /app
COPY . /app


ENTRYPOINT ["python", "/app/main.py"]
