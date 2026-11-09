# Small, reproducible image that serves the API (and can also run experiments).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Copy only dependency metadata first => Docker caches the slow `pip install` layer
# until pyproject.toml changes (not on every source edit).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY data/registry.yaml ./data/registry.yaml
COPY experiments ./experiments

# Never run as root inside the container.
RUN useradd -m app && mkdir -p /app/results /app/data/raw && chown -R app /app
USER app

ENV ADA_REGISTRY=/app/data/registry.yaml ADA_RAW_DIR=/app/data/raw
EXPOSE 8000
HEALTHCHECK CMD python -c "import httpx;httpx.get('http://localhost:8000/health').raise_for_status()"
CMD ["uvicorn", "ada.api:app", "--host", "0.0.0.0", "--port", "8000"]
