FROM python:3.12-slim

ENV PIP_DEFAULT_TIMEOUT=60 \
    PIP_RETRIES=10

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
ARG AUDIO_PLATFORM_EXTRAS=""
ARG PYTORCH_VERSION="2.11.0"
ARG TORCHCODEC_VERSION="0.13.0"
ARG PYTORCH_CPU_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN --mount=type=cache,target=/root/.cache/pip \
    case ",$AUDIO_PLATFORM_EXTRAS," in \
        *,diarization-pyannote,*) \
            pip install \
                --index-url "$PYTORCH_CPU_INDEX_URL" \
                "torch==${PYTORCH_VERSION}+cpu" \
                "torchaudio==${PYTORCH_VERSION}+cpu" \
                "torchcodec==${TORCHCODEC_VERSION}+cpu" \
            ;; \
    esac

RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import os, subprocess, sys, tomllib; data=tomllib.load(open('pyproject.toml','rb'))['project']; deps=list(data['dependencies']); [deps.extend(data['optional-dependencies'][name]) for name in os.environ.get('AUDIO_PLATFORM_EXTRAS','').split(',') if name]; subprocess.check_call([sys.executable,'-m','pip','install',*deps])"

COPY packages ./packages
COPY apps ./apps
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation --no-deps .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/input /data/model-cache \
    && chown -R appuser:appuser /data
USER appuser

CMD ["audio-ingest-worker"]
