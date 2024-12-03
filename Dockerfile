FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_mlflow.txt .
RUN pip install --no-cache-dir -r requirements_mlflow.txt

# Copy application code into the container
COPY ./app ./app

# Expose the port MLFlow Service will run on
EXPOSE 8001

# Command to run the MLFlow Service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
