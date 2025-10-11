FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY configs/python/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Install pac_system as an editable package from the monorepo packages directory
RUN pip install --no-cache-dir -e packages/pac_system

CMD ["python", "-c", "import pac_system; print('PAC System', pac_system.__version__)"]


