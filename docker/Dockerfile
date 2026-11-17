# CanVec experiment image (CPU only).
# Experiments run inside this container, never directly on the host.
# Build:  docker build -t canvec/exp:core -f docker/Dockerfile docker/
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements-core.txt /opt/canvec/requirements-core.txt
RUN pip install -r /opt/canvec/requirements-core.txt \
    && pip freeze > /opt/canvec/requirements.freeze.txt

WORKDIR /work
