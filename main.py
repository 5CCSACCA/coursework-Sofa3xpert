from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import io

# Initialize the FastAPI app
app = FastAPI()

# Load the YOLOv5 model
WEIGHTS_PATH = "weights/yolov5s.pt"
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=WEIGHTS_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Endpoint: /status
@app.get("/status")
async def status():
    if model:
        return {"status": "Model is loaded and ready for predictions."}
    else:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

# Endpoint: /version
@app.get("/version")
async def version():
    return {"model_version": "YOLOv5-custom", "api_version": "1.0"}

# Endpoint: /predict
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # Read the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Perform inference
        results = model(image)
        predictions = results.pandas().xyxy[0]  # Convert results to pandas DataFrame

        # Extract class names and confidences
        response = []
        for _, row in predictions.iterrows():
            response.append({"class": row["name"], "confidence": float(row["confidence"])})

        return {"predictions": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
