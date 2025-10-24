FROM python:3.12-slim

# Keep git for optional runtime pulls (lightweight)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy vectorizer-specific requirements (kept local to build context)
COPY requirements.txt /app/requirements.txt

# Install runtime deps for vectorizer (may be large)
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy vectorizer app
COPY . /app
RUN chmod +x /app/start.sh || true

EXPOSE 8080

CMD ["/app/start.sh"]

