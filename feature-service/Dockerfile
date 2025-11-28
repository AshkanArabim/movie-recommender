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

# Data directory will be mounted as a volume at /data
# No need to copy large CSV/pkl files into the image

# Expose gRPC port
EXPOSE 50051

# Run the server
CMD ["uv", "run", "python", "server.py"]

