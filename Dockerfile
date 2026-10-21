FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY config ./config
RUN mkdir -p models

# Model weights are not in git. Either mount models/csrnet.onnx, or set SS_MODEL_URL
# to a download link (for example a GitHub Release asset) and the app fetches it on start.
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
