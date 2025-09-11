# Café Mapper — Esqueleto do Projeto

Sistema web simples para **mapeamento de cafezais (Conilon)** a partir de **fotos RGB** feitas por drone. MVP focado em **usabilidade** para pequenos produtores: upload de imagem, processamento básico (ervas daninhas, vigor/cores das folhas, falhas) e visualização em mapa.

## Stack
- **Backend**: Python 3.11 + FastAPI, Jinja2 (templates), OpenCV/NumPy/scikit-image
- **Frontend**: Vite + React + TypeScript + Leaflet
- **Container**: Docker (opcional)

## Como rodar Backend (sem Docker)
```bash
cd backend
python -m venv .venv
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Acesse: http://localhost:8000

## Como rodar Frontend (sem Docker)
```bash
cd frontend
npm install
npm run dev
```
Abra: http://localhost:5173 (proxy configurado para /api -> http://localhost:8000)

## Como rodar com Docker (api + web)
```bash
docker compose up -d --build
```
Acesse frontend: http://localhost:5173

## Estrutura
```
backend/           # API FastAPI + processamento
frontend/          # SPA React (Vite) + Leaflet
data/              # amostras e saídas
docs/              # documentação
scripts/           # scripts dev
tests/             # testes
```
