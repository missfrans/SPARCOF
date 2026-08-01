FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .
COPY configs ./configs
COPY scripts/make_demo_dataset.py ./scripts/make_demo_dataset.py

CMD ["sparcof", "--help"]
