# =============================================================================
# Production Dockerfile for Railway Deployment
# Security Hardened Configuration
# =============================================================================

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables for security and performance
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV PORT=5000

# Create non-root user for security (Railway best practice)
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copy application code
COPY . /app/

# Create necessary directories with correct permissions
RUN mkdir -p /app/data /app/models \
    && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Health check for Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/foods || exit 1

# Expose port (Railway uses PORT env variable)
EXPOSE ${PORT}

# Run with Gunicorn for production (NOT Flask dev server)
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 120 --access-logfile - --error-logfile - app_flask:app
