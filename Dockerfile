# Build stage
FROM python:3.11-slim-bookworm AS builder

# Install uv (fast package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install dependencies (leverage caching)
COPY pyproject.toml uv.lock ./
# Sync dependencies only
ENV UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-install-project --no-dev

# Install project
COPY src ./src
COPY README.md ./
# Sync project (installs plutus into .venv)
RUN uv sync --frozen --no-dev

# Final stage
FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy configuration (runtime requirement)
COPY config /app/config

# Set environment
# - Add venv to PATH
# - Set PLUTUS_ROOT to /app (required for config loading)
# - Unbuffer python output for logs
ENV PATH="/app/.venv/bin:$PATH" \
    PLUTUS_ROOT="/app" \
    PYTHONUNBUFFERED=1

# Create data directory
RUN mkdir -p /app/data

# Default command
ENTRYPOINT ["plutus"]
CMD ["--help"]
