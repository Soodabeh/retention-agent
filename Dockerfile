FROM python:3.10.6-buster

COPY requirements.txt requirements.txt
COPY api api
COPY frontend frontend
COPY retention_agent retention_agent
COPY notebooks/models notebooks/models

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose ports for both FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Run both processes simultaneously in the background
CMD uvicorn api.fast:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0
