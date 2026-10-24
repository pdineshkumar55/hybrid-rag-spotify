# One image for both services: the API (FastAPI) and the UI (Streamlit).
# docker-compose.yml runs it twice with different commands.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code (see .dockerignore for what is left out: venv, .env, data, vectorstore...)
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY ui/ ./ui/

# Run as a non-root user
RUN useradd --create-home appuser \
    && mkdir -p /app/data /app/vectorstore \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# Default: the API. The UI service overrides this in docker-compose.yml.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
