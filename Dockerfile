# Serves the Streamlit app. Train first so models/ contains the .keras file.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY plant_disease/ plant_disease/
COPY .streamlit/ .streamlit/
COPY app.py .
COPY models/ models/

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
