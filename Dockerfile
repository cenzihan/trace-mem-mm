FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir --only-binary=:all: --require-hashes -r requirements-docker.txt \
    && groupadd --gid 10001 tracemem \
    && useradd --uid 10001 --gid 10001 --no-create-home tracemem \
    && mkdir -p /data \
    && chown 10001:10001 /data

COPY agent_memory/ /app/agent_memory/
COPY deploy/ /app/deploy/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "/app/deploy/healthcheck.py"]
CMD ["gunicorn", "--config", "/app/deploy/gunicorn.conf.py", "agent_memory.container_app:create_app()"]
