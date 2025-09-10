# PDI Cafezais — Project Skeleton

Monorepo de referência (MVP robusto) para mapeamento de cafezais com **FastAPI**, **PostGIS**, **Celery/Redis**, **MinIO (S3)**, **MLflow** e **Frontend React+Leaflet**.  
Feito para rodar **via Docker Compose** (offline-friendly).

## Stack (segundo o plano anexado)
- **Backend**: FastAPI + SQLAlchemy + GeoAlchemy2 + Alembic
- **DB**: Postgres **com PostGIS**
- **Mensageria**: Celery + Redis
- **Armazenamento de objetos**: MinIO (S3-compatible)
- **MLOps**: MLflow (tracking + artifacts S3/MinIO)
- **Frontend**: React + TypeScript + Vite + Leaflet
- **Observabilidade**: logs estruturados + health checks
- **Infra**: Docker Compose

## Rodando
1) Copie variáveis de ambiente:
```bash
cp .env.example .env
```
2) Suba tudo:
```bash
docker compose up -d --build
```
3) Aguarde o backend aplicar migrações Alembic e criar extensões PostGIS automaticamente.
4) Acesse:
- API: http://localhost:8000/docs
- Frontend: http://localhost:5173  (proxy/preview dev) ou http://localhost:8080 (build via Nginx)
- MLflow: http://localhost:5000
- MinIO: http://localhost:9001 (console) | http://localhost:9000 (S3 API)

## Estrutura
```
backend/           # FastAPI + SQLAlchemy + Alembic + Celery
frontend/          # React + TS + Vite + Leaflet
docker-compose.yml
.env.example
```

## Tabelas iniciais
- `farm`, `plot`, `image`, `mosaic`, `label`, `model_version`, `inference_run`, `mask`

## Comandos úteis
```bash
# logs
docker compose logs -f backend

# aplicar migrações manualmente
docker compose exec backend alembic upgrade head

# criar usuário/bucket no MinIO (re-run)
docker compose run --rm createbuckets
```

> **Observação**: Este é um esqueleto funcional com _stubs_ de rotas/serviços. Substitua os TODOs conforme evoluir.


## Perfis de execução (profiles)
Para manter o backend leve e adicionar capacidades conforme a necessidade, use perfis do Compose:

- **Geo (leve)**: somente PDI/Geo
```bash
docker compose --profile geo up -d --build ml-worker-geo
```

- **ML ONNX (inferência leve)**:
```bash
docker compose --profile ml up -d --build ml-worker-onnx
```

- **Torch/Ultralytics (treino+inferência — pesado)**:
```bash
docker compose --profile train up -d --build ml-worker-torch
```

Endpoints de teste:
- `POST /inference/geo`
- `POST /inference/onnx`
- `POST /inference/torch`
