# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all requirements first for better caching
COPY backend/requirements.txt /app/backend/requirements.txt
COPY frontend/requirements.txt /app/frontend/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt -r /app/frontend/requirements.txt

# Copy the application code
COPY backend /app/backend
COPY frontend /app/frontend

# Create a script to run both services
RUN echo '#!/bin/bash\n\
    cd /app/backend && PYTHONPATH=/app uvicorn main:app --host 0.0.0.0 --port 8000 & \
    cd /app/frontend && PYTHONPATH=/app streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0\
    ' > /app/start.sh

RUN chmod +x /app/start.sh

EXPOSE 8000 8501

CMD ["/app/start.sh"]
