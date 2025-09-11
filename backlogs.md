# Cronograma detalhado (passo a passo)

Abaixo um roteiro **prático** para você (desenvolvedor solo) evoluir do MVP até uma versão usável em campo. Sugestão de  **6–8 semanas** , adaptável.

## Semana 0 — Preparação do ambiente

* [ ] Clonar/extrair o projeto e validar build local e com Docker (`docker compose up -d`).
* [ ] Configurar virtualenv do backend, `pip install -r requirements.txt`.
* [ ] Subir frontend (Vite) e confirmar proxy `/api` → `8000`.
* [ ] Criar branch `feat/mvp-pipeline`.

  **Entregáveis:** app abre em `http://localhost:5173` e `http://localhost:8000/api/health` retorna `{"status":"ok"}`.

## Semana 1 — MVP Pipeline (ervas daninhas “simples”)

* [ ] Implementar segmentação HSV básica em `services/processing/weed.py`:

  * Converter para HSV, máscaras de verdes “claros/amarelados” (mato) vs “verde escuro/brilhante” (café).
  * Remoção de ruído (morfologia) e contorno de blobs (áreas mínimas).
* [ ] Endpoint `/api/process`: retornar também um **resumo** (qtd. de áreas, área aproximada).
* [ ] UI: mostrar imagem anotada no React (ou manter na página do backend por enquanto).

  **Critério de aceite:** imagem de teste gera contornos plausíveis de invasoras sem muitos falsos positivos.

## Semana 2 — Vigor (cor de folhas) e Falhas

* [ ] `vigor.py`: índice **Excess Green (ExG)** ou variações RGB para mapear vigor (heatmap).
* [ ] `gaps.py`: heurística de  **falhas de plantio** :

  * Detectar padrão/linhas (básico) ou grid + threshold de “solo exposto”.
  * Marcar “buracos” acima de área mínima.
* [ ] Backend: retornar **GeoJSON** (polígonos/pontos) além da imagem.
* [ ] Frontend: renderizar **camadas no Leaflet** (polígonos de daninhas, pontos de falhas, raster/heatmap opcional).

  **Critério de aceite:** visualizar no mapa as camadas separadas (ligar/desligar) e resumo numérico no painel.

## Semana 3 — UX, robustez e testes

* [ ] UI: tela única com cards:  **Upload → Resultado → Mapa → Sumário** .
* [ ] Tratamento de erros (tamanho/formatos de imagem, timeouts).
* [ ] Testes básicos (unitários) de utilitários (ex.: máscaras e limiares).
* [ ] Perfil de performance: otimizar leitura/escrita de imagens, evitar cópias desnecessárias.
* [ ] Doc: atualizar `USER_GUIDE.md` com fotos de exemplo e dicas de captura (altitude, iluminação).

  **Critério de aceite:** fluxo suave de upload; erros exibidos de forma amigável; testes passando.

## Semana 4 — Mosaico simples e ajustes de campo

* [ ] Permitir upload de **múltiplas imagens** e processar individualmente (por ora sem “stitch”).
* [ ] Implementar **equalização/normalização** simples de cor para reduzir efeito de sombra/iluminação.
* [ ] Flag de “sensibilidade” (slider) para o produtor calibrar o quão agressiva é a detecção.
* [ ] Logging básico e métricas de execução.

  **Critério de aceite:** usuário consegue ajustar sensibilidade e obter resultados mais estáveis.

## Semana 5 — Pré-piloto (refino e pacote)

* [ ] Gerar **relatório** simples (HTML/PDF) com miniaturas e contagens por tipo (mato, falhas, baixa cor).
* [ ] Script `scripts/export_report.py` (opcional) para lote.
* [ ] Configurar build **Docker** reprodutível e validar em outra máquina.
* [ ] Preparar conjunto de **imagens reais** (se possível) para ensaio.

  **Critério de aceite:** relatório gerado com sumário e links para imagens/camadas.

## Semana 6 — Piloto de campo

* [ ] Rodar com 1–2 talhões reais (se ainda sem drone, usar imagens públicas/bancos internos).
* [ ] Coletar feedback: falsos positivos, legibilidade do mapa, termos na UI.
* [ ] Ajustar thresholds, regras morfológicas e tamanho mínimo de blob.

  **Critério de aceite:** feedback de “utilidade prática” positivo; lista de melhorias priorizada.

## Semana 7 — Polimento e “v1”

* [ ] Guardar **projetos** (metadados por execução) em SQLite (data, notas, estatísticas).
* [ ] Tela “Histórico” (listar processamentos anteriores e reabrir resultados).
* [ ] Melhorar responsividade (mobile) e acessibilidade (contraste, fontes).
* [ ] Hardening: limites de upload, varredura de extensão, mensagens localizadas.

  **Critério de aceite:** histórico funcionando e UX estável.

## Semana 8 — Próximos passos (opcional)

* [ ] Acrescentar **filtros por iluminação** (sombreamento) e pós-processamento de bordas.
* [ ] “Comparar datas” (lado a lado) para ver evolução pós-intervenção.
* [ ] Início de um classificador leve (sem treinar ainda) e pipeline para anotação manual (futuro ML).

  **Critério de aceite:** comparador de datas básico e backlog de IA definido.

---

## Backlog técnico (resumo de tarefas)

**Backend**

* [ ] `/api/process` retornar: `result_image_url`, `summary`, `geojson_weeds`, `geojson_gaps`, `raster_vigor_url`.
* [ ] Implementar ExG e heatmap (salvar em `static/results/`).
* [ ] Normalização de cor (CLAHE/opencv) opcional por parâmetro.
* [ ] Limites de upload e validações.

**Frontend**

* [ ] Camadas Leaflet (toggle): Daninhas (polígono), Falhas (pontos), Vigor (tile/raster ou canvas overlay).
* [ ] Painel lateral com métricas: % área afetada, nº de falhas, etc.
* [ ] Upload arrasta-e-solta, barra de progresso.

**DevOps**

* [ ] Logs estruturados (nível INFO/ERROR).
* [ ] Script de seed com imagens exemplo em `data/samples/`.
* [ ] CI simples (lint + tests).
