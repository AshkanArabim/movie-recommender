FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY gateway-service/pyproject.toml gateway-service/uv.lock* ./

# Install dependencies
RUN uv sync --frozen

# Copy service files
COPY gateway-service/ .

# Copy ranking-service proto file (needed for generating proto modules)
RUN mkdir -p ../ranking-service
COPY ranking-service/ranking.proto ../ranking-service/ranking.proto

# Copy feature-service proto file (needed for generating proto modules)
RUN mkdir -p ../feature-service
COPY feature-service/embeddings.proto ../feature-service/embeddings.proto

# Expose HTTP port
EXPOSE 8000

# Run the server
CMD ["uv", "run", "python", "server.py"]

