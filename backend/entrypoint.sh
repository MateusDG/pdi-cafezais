#!/usr/bin/env bash
set -e

# Wait for Postgres
echo ">> Waiting for Postgres at ${POSTGRES_HOST}:${POSTGRES_PORT}..."
for i in {1..60}; do
  if PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "${POSTGRES_HOST}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "SELECT 1" >/dev/null 2>&1; then
    echo ">> Postgres is up!"
    break
  fi
  echo "   still waiting..."; sleep 2
done

# Alembic migrations (create PostGIS extension if needed)
echo ">> Running Alembic migrations"
alembic upgrade head

# Start API
echo ">> Starting FastAPI"
exec uvicorn app.main:app --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-8000} --reload
