# API lista para desplegar (opcional, cuenta para alpha). Construir y correr:
#   docker build -t proyecto-api .
#   docker run -p 8000:8000 --env-file .env proyecto-api
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api ./api
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
