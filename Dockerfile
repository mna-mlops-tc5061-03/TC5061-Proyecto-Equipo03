FROM python:3.13-slim

WORKDIR /app

# MLFLOW_TRACKING_URI y MODELO_URI los lee mi_bici/config.py del entorno al correr
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

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]

CMD ["uvicorn", "mi_bici.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

