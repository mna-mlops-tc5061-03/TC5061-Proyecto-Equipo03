FROM python:3.13-slim

WORKDIR /app

# MLFLOW_TRACKING_URI, MODELO_URI y MODELO_URI_P1 los lee mi_bici/config.py del entorno al correr
# el contenedor (-e MLFLOW_TRACKING_URI=... -e MODELO_URI=...), sin fijarlos en la imagen
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Primero las dependencias de la API, para aprovechar la caché de capas si solo cambia el código
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Después el código del proyecto
COPY mi_bici/ ./mi_bici/

# Usuario sin privilegios para no correr el servidor como root
RUN useradd --create-home --shell /sbin/nologin appuser && chown -R appuser:appuser /app
USER appuser

# Si el servidor de MLflow no responde, por defecto MLflow reintenta 7 veces con espera creciente
# y la API tarda varios minutos en arrancar. Con 2 reintentos y 10 s por petición arranca en
# segundos y responde 503. Se pueden cambiar con -e al correr el contenedor
ENV MLFLOW_HTTP_REQUEST_MAX_RETRIES=2 \
    MLFLOW_HTTP_REQUEST_TIMEOUT=10

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]

CMD ["uvicorn", "mi_bici.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

