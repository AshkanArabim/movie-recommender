FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY candidate-service/pyproject.toml candidate-service/uv.lock* ./

# Install dependencies
RUN uv sync --frozen

# Copy service files
COPY candidate-service/ .

# Copy feature-service proto file (needed for generating proto modules)
# Create the directory structure to match the code's expectations
RUN mkdir -p ../feature-service
COPY feature-service/embeddings.proto ../feature-service/embeddings.proto

# Expose gRPC port
EXPOSE 50052

# Run the server
CMD ["uv", "run", "python", "server.py"]

