# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Café Mapper is a web system for mapping coffee plantations (Conilon) from RGB drone photos. It's an MVP focused on usability for small producers, featuring image upload, basic processing (weeds, vigor/leaf colors, gaps), and map visualization.

## Development Commands

### Backend (Python FastAPI)
```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Backend runs on: http://localhost:8000

### Frontend (React + TypeScript + Vite)
```bash
cd frontend
npm install
npm run dev          # Development server
npm run build        # Production build
npm run preview      # Preview production build
```
Frontend runs on: http://localhost:5173 (proxy configured for /api -> http://localhost:8000)

### Docker (Full Stack) - RECOMMENDED
```bash
# Production (recommended)
docker-compose up -d                                    # Start production environment
docker-compose ps                                       # Check status
docker-compose logs -f                                  # View logs

# Development with hot-reload
docker-compose -f docker-compose.dev.yml up -d          # Start dev environment
docker-compose -f docker-compose.dev.yml logs -f        # View dev logs

# Automated scripts
./scripts/docker-prod.sh start                          # Production with automation
./scripts/docker-dev.sh start                           # Development with automation

# Access points
# Production: http://localhost (port 80)
# Development: http://localhost:5173
# API: http://localhost:8000
```

**Docker Features:**
- ✅ **Multi-stage builds** for optimized images
- ✅ **Security**: Non-root users, security headers
- ✅ **Health checks** and auto-restart
- ✅ **Volume persistence** for data and results  
- ✅ **Hot-reload** in development mode
- ✅ **Production-ready** Nginx configuration
- ✅ **Backup/restore** scripts for production data

## Architecture

### Backend Structure
- **FastAPI app** with CORS middleware for frontend integration
- **API endpoints**: `/api/health` and `/api/process` for image processing
- **Processing services**: Located in `app/services/processing/`
  - `gaps.py` - Detection of planting gaps/lacunas (TODO)
  - `vigor.py` - Plant vigor analysis 
  - `weed.py` - Weed detection
  - `utils.py` - Common processing utilities
- **Configuration**: Environment-based settings in `app/core/config.py`
- **Static files**: Processed results served from `app/static/results/`
- **Templates**: Jinja2 templates for basic HTML responses

### Frontend Structure
- **React 18** with TypeScript
- **Vite** as build tool and development server
- **Leaflet + react-leaflet** for map visualization
- **Main components**:
  - `Upload` - Image upload interface
  - `MapView` - Leaflet map display component

### Data Flow
1. Frontend uploads image via `/api/process` endpoint
2. Backend processes image using OpenCV/scikit-image
3. Results saved to static directory and URL returned
4. Frontend displays processed image and notes

### Key Dependencies
- **Backend**: FastAPI, OpenCV, NumPy, scikit-image, Pillow, Jinja2
- **Frontend**: React, TypeScript, Leaflet, Vite

### Environment Configuration
- Backend uses environment variables for paths (DATA_DIR, OUTPUTS_DIR, STATIC_RESULTS)
- Frontend configured with Vite proxy for API calls
- Docker setup available with separate containers for API and web

## Processing Pipeline
The image processing currently focuses on basic coffee plantation analysis:
- Image upload and validation
- Processing through specialized modules (vigor, weeds, gaps detection)
- Output generation with annotations
- Static file serving for results display