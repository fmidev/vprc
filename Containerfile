# Container image for vprc - VPR correction for weather radar data.
# Used by Airflow @task.docker tasks, which run `python` from PATH; Quay builds
# the image from Git tags.
#
# The build is reproducible: the base and uv images are pinned by digest, the
# Python dependencies by uv.lock and the build backend by
# [tool.uv] build-constraint-dependencies. Only the Debian libexpat1 package
# (needed by the rasterio wheels) follows Debian stable updates.
#
# To update the base image, pick a new tag and refresh its multi-arch digest:
#   skopeo inspect --raw docker://docker.io/library/python:<tag> | sha256sum

ARG PYTHON_IMAGE=docker.io/library/python:3.14.10-slim-trixie@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f

FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv


FROM ${PYTHON_IMAGE} AS builder

# git is needed by hatch-vcs to derive the version
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Locked runtime dependencies first, so this layer is reused when only the
# source changes
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md LICENSE.txt ./
COPY src/ src/
COPY .git/ .git/
RUN uv sync --locked --no-dev --no-editable


FROM ${PYTHON_IMAGE}

LABEL org.opencontainers.image.title="vprc"
LABEL org.opencontainers.image.description="FMI VPR correction for weather radar data"
LABEL org.opencontainers.image.source="https://github.com/fmidev/vprc"
LABEL org.opencontainers.image.licenses="MIT"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash vprc

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

USER vprc
WORKDIR /home/vprc

# Default command shows package info (useful for debugging)
CMD ["python", "-c", "from importlib.metadata import version; print(f'vprc {version(\"vprc\")}')"]
