FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY feature-service/pyproject.toml feature-service/uv.lock* ./

# Install dependencies
RUN uv sync --frozen

# Copy service files
COPY feature-service/ .

# Create embedding-calc directory and copy embedding files (maintaining relative path structure)
# Since WORKDIR is /app, ../embedding-calc resolves to /embedding-calc
RUN mkdir -p /embedding-calc
COPY embedding-calc/user_embeddings.pkl /embedding-calc/user_embeddings.pkl
COPY embedding-calc/movie_embeddings.pkl /embedding-calc/movie_embeddings.pkl

# Expose gRPC port
EXPOSE 50051

# Run the server
CMD ["uv", "run", "python", "server.py"]

