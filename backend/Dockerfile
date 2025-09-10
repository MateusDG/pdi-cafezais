FROM python:3.11-slim

ENV POETRY_VERSION=1.8.3 PIP_DISABLE_PIP_VERSION_CHECK=on PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# System deps (psql for health/wait)
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential curl gnupg libpq-dev postgresql-client     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Entrypoint: wait DB, run migrations, start API
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
