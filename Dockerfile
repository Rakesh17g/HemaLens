# ─────────────────────────────────────────────────────────────────────────────
# ALL Detection System — Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build:
#   Stage 1 (builder) — install Python deps in a venv
#   Stage 2 (runtime) — copy only the venv; minimal image size
#
# Build:
#   docker build -t all-detection:latest .
#
# Run (Streamlit dashboard):
#   docker run -p 8501:8501 -v $(pwd)/models:/app/models all-detection:latest
#
# Run (training):
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
#              all-detection:latest python train.py
#
# Run (evaluation):
#   docker run -v $(pwd)/models:/app/models -v $(pwd)/logs:/app/logs \
#              all-detection:latest python evaluate.py \
#              --checkpoint models/checkpoints/best.pth
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System deps needed to build some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps — leverages Docker layer cache
COPY requirements.txt .
RUN pip install --upgrade pip wheel \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="ALL Detection System" \
      description="EfficientNet-B0 Acute Lymphoblastic Leukaemia Screening" \
      version="1.0.0"

# Minimal runtime libs for OpenCV (headless) and ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy source code (respects .dockerignore)
COPY --chown=appuser:appuser . .

# Create writable directories for logs, models, reports
RUN mkdir -p logs models/checkpoints data reports \
 && chown -R appuser:appuser logs models data reports

USER appuser

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

# Health check — Streamlit responds on /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1

# Default: launch the Streamlit dashboard
CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
