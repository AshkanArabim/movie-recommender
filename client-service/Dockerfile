FROM python:3.13-slim

WORKDIR /app

# Copy client files
COPY client-service/ .

# Expose port
EXPOSE 8080

# Run simple HTTP server
CMD ["python", "-m", "http.server", "8080"]

