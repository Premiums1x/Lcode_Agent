FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions
COPY pyproject.toml README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[all]"

# Copy application code
COPY lcode/ ./lcode/
COPY examples/ ./examples/

# Create data directories
RUN mkdir -p /app/data /app/plugins

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Expose web UI port
EXPOSE 8000

# Default command: web server
CMD ["lcode", "server", "--host", "0.0.0.0", "--port", "8000"]
