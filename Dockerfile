FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_yolo.txt .
RUN pip install --no-cache-dir -r requirements_yolo.txt

COPY ./app ./app

# Expose the port YOLO Service will run on
EXPOSE 8000

# Command to run the YOLO Service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
