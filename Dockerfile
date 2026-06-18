FROM python:3.10.6-buster

COPY requirements.txt requirements.txt
COPY api api
COPY retention_agent retention_agent
COPY notebooks/models notebooks/models

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
