# Plano de Execução — PDI para Mapeamento de Cafezais (Somente Imagens de Satélite)

**Data de início:** 10 Sep 2025 • **Horizonte:** 8 semanas • **Fuso:** America/Sao_Paulo  
**Premissa:** *Sem drone neste momento* — usar **imagens de satélite** (p.ex., Sentinel‑2 L2A ~10 m, Landsat 8/9 OLI ~30 m) para MVP e validação.  
**Monorepo base:** `pdi-cafezais-skeleton` (backend FastAPI + PostGIS, Celery/Redis, MinIO/S3, MLflow, frontend Leaflet) com **perfis** para PDI/Geo, ONNX e Torch.

---

## 1) Objetivo e escopo
- **Objetivo:** mapear cafezais e **destacar áreas problemáticas** (falhas, ervas daninhas/heterogeneidade, vigor baixo) a partir de **satélite**, gerando **camadas GIS** (GeoJSON/Shapefile) e um **dashboard simples**.
- **Escopo (MVP):**
  - Importar cenas (COG/GeoTIFF) → **pré-processar** → **NDVI/EVI** → **segmentar** talhões/áreas de interesse → **detectar anomalias** simples → **mapa final** e **export GIS**.
  - **Sem treino** no MVP; foco em PDI clássico + regras (threshold/adaptive/k‑means) e estrutura pronta para plugar ML.

## 2) Requisitos funcionais (RF)
1. **Upload/registro** de cena/s (URI S3/MinIO ou caminho local) e metadados básicos.
2. **Pré-processamento**: recorte por AOI, reprojeção para EPSG:4326, mascaramento de nuvem/sombra (quando disponível), filtros (mediana/gauss), equalização/CLAHE.
3. **Índices**: cálculo de **NDVI** (e **EVI** quando houver NIR/BLUE), estatísticas por AOI/talhão.
4. **Segmentação** ROI: threshold (Otsu/adaptativo), **watershed** ou **k‑means** (k=2..4) com heurísticas.
5. **Detecção**: gaps (falhas) por morfologia/área mínima; hotspots de baixa vegetação via score NDVI/EVI; **mapa de classes** simples.
6. **Exportação**: GeoJSON e Shapefile (polígonos de falha/baixa saúde + grade/heatmap).
7. **Visualização**: frontend Leaflet com camadas lig/desl, paletas simples e popup com estatísticas.
8. **Jobs assíncronos** via Celery e **rastreamento** no MLflow (parâmetros e métricas).

## 3) Não funcionais (RNF)
- **Reprodutibilidade** (seed e parâmetros gravados), **execução offline** (MinIO local), **tempo**: cena média processada em ≤ 5–10 min (CPU), **observabilidade** (logs estruturados).

## 4) Arquitetura e perfis
- **Backend leve** (CRUD/rotas) + **workers** específicos por perfil:
  - `geo` → `opencv-python-headless`, `rasterio`, `shapely`
  - `ml` → `onnxruntime`
  - `train` → `torch` + `ultralytics` (opcional/pesado)
- **Rotas disponíveis**: `/inference/geo`, `/inference/onnx`, `/inference/torch` (stubs prontos para evoluir).

## 5) Dados e preparação
- **Fontes**: Sentinel‑2 (L2A) ou Landsat 8/9 (OLI). **Preferir** cenas com baixa cobertura de nuvem para a AOI.
- **Formato**: GeoTIFF/COG; **bandas** mínimas: **NIR e RED** (NDVI); opcional **BLUE** (EVI).
- **AOI**: polígono GeoJSON do(s) talhão(ões) ou bounding box de estudo.
- **Organização** (MinIO): `s3://raw-images/<projeto>/<data>/<cena>.tif`

## 6) Critérios de aceite do MVP
- **NDVI** gerado para AOI; **mapa binário** “vegetação vs não‑vegetação” com *F1 ≥ 0,80* em amostras manuais.
- Detecção de **falhas** (polígonos com área ≥ X m²) exportada em **GeoJSON**.
- **Frontend** carrega base OSM + camadas GeoJSON; popup com estatísticas por polígono (NDVI médio, área).
- Pipeline roda via **POST `/inference/geo`** com parâmetros (AOI, URIs, thresholds).

---

## 7) Cronograma detalhado (8 semanas)

### Semana 1 — Fundamentos & dados (10 Sep 2025–16 Sep 2025)
- Ajustar `.env` e subir stack base (`docker compose up -d`).
- Ativar **profile geo**: `docker compose --profile geo up -d --build ml-worker-geo`.
- Definir **AOI** (GeoJSON) e **cenas** alvo (2–3 datas) e organizar no MinIO.
- Implementar **reader Rasterio** + recorte por AOI; utilitário de **reprojeção** e **stats**.

**Entrega:** função `read_and_crop()` + rota inicial `/inference/geo` com parâmetros de AOI/URI.  

### Semana 2 — Pré-processamento (17 Sep 2025–23 Sep 2025)
- **Máscara de nuvem** quando disponível; fallback por thresholds de brilho.
- **Filtros**: mediana/gauss + **CLAHE** opcional.
- Normalização e checagem de **no‑data** / *transform*. Logs com parâmetros.

**Entrega:** módulo `preprocessing/` com pipeline encadeado + métricas de tempo no MLflow.

### Semana 3 — Índices de vegetação (24 Sep 2025–30 Sep 2025)
- Implementar **NDVI** (NIR/RED) e **EVI** (se BLUE disponível).
- Geração de **rasters** e **stats** (média, p5/p95, histograma por AOI).

**Entrega:** endpoints para calcular/exportar NDVI/EVI (GeoTIFF) e salvar stats no MLflow.

### Semana 4 — Segmentação ROI (01 Oct 2025–07 Oct 2025)
- **Threshold** (Otsu/adaptativo) → **mask vegetação**.
- Alternativa **k‑means** (k=2..4) sobre [NDVI, RED, NIR].
- **Morfologia** (open/close) para limpar ruído; **watershed** em casos complexos.

**Entrega:** `mask_veg.tif` + **GeoJSON** vetorizado (contornos).

### Semana 5 — Detecção de problemas (08 Oct 2025–14 Oct 2025)
- **Falhas**: detectar gaps por *connected components* + área mínima (paramétrica).
- **Baixo vigor**: “hotspots” com NDVI < T (por percentil).
- **Ervas daninhas** (MVP): heterogeneidade de textura/cor (GLCM simples/entropia) — *heurística*.

**Entrega:** `problems.geojson` com classes {{falha, baixo_vigor, heterog}} + atributos (área, score).

### Semana 6 — Mapa & frontend (15 Oct 2025–21 Oct 2025)
- Camadas no **Leaflet** (NDVI como *tile* básico + **GeoJSON** de problemas).
- **Popup** com estatísticas; legenda simples; *toggle* de camadas.

**Entrega:** frontend com camadas; rota de **download** (GeoJSON/Shapefile).

### Semana 7 — Qualidade, export e automação (22 Oct 2025–28 Oct 2025)
- **Exportadores** (GeoJSON, Shapefile). Validação no QGIS.
- **Jobs Celery** parametrizados; *retry/backoff*; logs/erros clareados.
- **MLflow**: parâmetros (thresholds, janela CLAHE, k) + métricas (F1 vegetação, #gaps, área).

**Entrega:** processamento idempotente por cena; reprocessamento com versionamento leve.

### Semana 8 — Piloto & ajustes (29 Oct 2025–04 Nov 2025)
- Rodar **piloto** em 1–2 áreas; coletar **amostras manuais** p/ validação.
- Ajustar thresholds e heurísticas; consolidar documentação **README** + exemplos.

**Entrega final:** pacote com **NDVI**, **problems.geojson**, prints de mapa e parâmetros usados.

---

## 8) Backlog (épicos → histórias)
- **Dados/Infra**: upload/URI, AOI, MinIO buckets, validação de metadados.
- **PDI**: pré-processamento, NDVI/EVI, segmentação, morfologia, vetorização.
- **Detecção**: gaps, baixo vigor, heterogeneidade.
- **Export/Visual**: GeoJSON/Shapefile, frontend Leaflet, legenda/popup.
- **Ops**: Celery (retry), MLflow (tracking), logs e testes.

> Cada história deve ter **DoD**: entrada/saída definidas, parâmetros registrados, teste de unidade, exemplo reproduzível e registro no MLflow.

## 9) Riscos e mitigação
- **Nuvem/sombra** alta → escolher datas alternativas; mascarar nuvem; interpolar temporalmente (futuro).
- **Resolução** (10–30 m) limita micro‑falhas → trabalhar com **área mínima** e validação por amostragem.
- **Registro/CRS** → reprojeção automática e checagem de *transform* antes de operar.

## 10) Como executar agora
```bash
# subir serviços base
cp .env.example .env
docker compose up -d --build

# ativar worker PDI/Geo (leve)
docker compose --profile geo up -d --build ml-worker-geo

# testar stub e depois plugar pipeline real em backend/app/modules/
curl -X POST http://localhost:8000/inference/geo
```

---

## 11) Próximas melhorias (opcionais pós-MVP)
- **ONNX Runtime** para inferência de modelos pré‑treinados (classe/segmentação) → profile `ml`.
- **YOLO/Ultralytics** para detecção de “falhas” ou pragas em imagens de maior resolução (quando houver) → profile `train`.
- **Camada temporal** (séries NDVI): tendência e alertas por talhão.
