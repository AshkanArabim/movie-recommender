FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY ranking-service/pyproject.toml ranking-service/uv.lock* ./

# Install dependencies
RUN uv sync --frozen

# Copy service files
COPY ranking-service/ .

# Copy candidate-service proto file (needed for generating proto modules)
RUN mkdir -p ../candidate-service
COPY candidate-service/candidates.proto ../candidate-service/candidates.proto

# Copy feature-service proto file (needed for generating proto modules)
RUN mkdir -p ../feature-service
COPY feature-service/embeddings.proto ../feature-service/embeddings.proto

# Expose gRPC port
EXPOSE 50053

# Run the server
CMD ["uv", "run", "python", "server.py"]

