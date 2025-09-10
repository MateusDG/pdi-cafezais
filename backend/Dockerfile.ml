FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=on PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# System deps (GDAL for rasterio)
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    build-essential gdal-bin libgdal-dev libgl1-mesa-glx libglib2.0-0 \ 
    libsm6 libxext6 libxrender-dev libgomp1 curl \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Base deps shared with backend
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Extra tiered deps (copy as optional)
ARG GEO_REQ=requirements-geo.txt
ARG ML_REQ=
COPY requirements-geo.txt /app/requirements-geo.txt
COPY requirements-ml-onnx.txt /app/requirements-ml-onnx.txt
COPY requirements-ml-torch.txt /app/requirements-ml-torch.txt

# Install GEO (always for ML workers)
RUN pip install --no-cache-dir -r /app/${GEO_REQ}

# Install ML tier if provided
RUN if [ -n "${ML_REQ}" ]; then pip install --no-cache-dir -r /app/${ML_REQ}; fi

COPY . /app

CMD ["bash", "-lc", "celery -A app.celery_app:celery_app worker --loglevel=INFO -Q ml"]
