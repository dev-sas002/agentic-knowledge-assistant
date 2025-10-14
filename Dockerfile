# Multi-stage. The build stage compiles wheels (torch and chromadb both need a
# toolchain for some transitive deps); the runtime stage gets the wheels and no
# compiler. That keeps build tooling out of the shipped image, which is both
# smaller and a smaller attack surface.

# ---- builder ----------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels

# Test dependencies stay in their own layer: requirements.txt pulls torch, and
# editing a test pin must not invalidate a multi-gigabyte layer.
COPY requirements.txt requirements-dev.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt \
 && /opt/venv/bin/pip install -r requirements-dev.txt

# Bake the embedding model into the image. Downloading it on first request
# makes a cold start look like a hang, and makes the container need network
# access to answer a question about local documents.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV HF_HOME=/opt/hf
RUN /opt/venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

# ---- runtime ----------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false

# curl only, for the healthchecks. No compiler in the runtime image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf

WORKDIR /app
COPY . .

# /data is the mount point for the named volume. Docker initialises a named
# volume from the image's directory, ownership included — so if /data does not
# exist in the image it is created root-owned and this non-root user cannot
# write to it.
RUN useradd --create-home --uid 1000 agent \
 && mkdir -p /app/data/raw /app/data/chroma /data/raw /data/chroma \
 && chown -R agent:agent /app /data /opt/hf
USER agent

EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=6 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "apps/ui/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
