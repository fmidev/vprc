# Dockerfile for vprc - VPR correction for weather radar data
# Designed for use with Airflow @task.docker decorator
#
# Example Airflow usage:
#   @task.docker(image="fmi/vprc:latest")
#   def correct_vpr(vvp_file: str, radar_config: dict) -> str:
#       from vprc import process_vvp
#       return process_vvp(vvp_file, radar_config)

FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install build tools
RUN pip install --no-cache-dir hatch hatch-vcs

# Copy source for building
COPY pyproject.toml README.md LICENSE.txt ./
COPY src/ src/
# Include .git for hatch-vcs versioning
COPY .git/ .git/

# Build wheel
RUN hatch build -t wheel


FROM python:3.12-slim

LABEL org.opencontainers.image.title="vprc"
LABEL org.opencontainers.image.description="FMI VPR correction for weather radar data"
LABEL org.opencontainers.image.source="https://github.com/fmidev/vprc"

# Install runtime dependencies
RUN apt-get update && apt-get install -y libexpat1 && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash vprc
WORKDIR /home/vprc

# Copy and install the built wheel
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Switch to non-root user
USER vprc

# Default command shows package info (useful for debugging)
CMD ["python", "-c", "import vprc; print(f'vprc {vprc.__version__}')"]
