FROM python:3.12-slim

ENV PIP_DEFAULT_TIMEOUT=60 \
    PIP_RETRIES=10

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY packages ./packages
COPY apps ./apps
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/input /data/artifacts \
    && chown -R appuser:appuser /data
USER appuser

EXPOSE 8000
CMD ["audio-platform-api"]
