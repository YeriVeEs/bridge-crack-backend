from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

  
    image = cv2.imread(file_path)

  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

   
    edges = cv2.Canny(blurred, 50, 150)

    overlay = image.copy()
    overlay[edges != 0] = [0, 0, 255]

    annotated = cv2.addWeighted(image, 0.8, overlay, 0.8, 0)

    
    output_filename = "annotated_" + file.filename
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    cv2.imwrite(output_path, annotated)

    return JSONResponse({
        "status": "success",
        "annotated_image_url": f"https://crack-detection-on-bridges.onrender.com/outputs/{output_filename}"
    })
