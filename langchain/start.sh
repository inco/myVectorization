#!/bin/bash
set -e

# Simple entrypoint for the vectorizer service
# Run with: ./start.sh

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}

exec uvicorn app:app --host $HOST --port $PORT

