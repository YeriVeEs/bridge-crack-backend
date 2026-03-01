from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
import numpy as np
from skimage.morphology import skeletonize

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

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read image
    image = cv2.imread(file_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        2
    )

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Skeleton of image
    binary_bool = binary > 0
    skeleton = skeletonize(binary_bool)
    skeleton = (skeleton * 255).astype(np.uint8)

    # Crack length (in pixels)
    crack_length_pixels = int(np.sum(skeleton == 255))

    # Width measurement
    distance_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    width_values = distance_map[skeleton == 255]

    if len(width_values) > 0:
        avg_width_pixels = float(np.mean(width_values) * 2)
        max_width_pixels = float(np.max(width_values) * 2)
    else:
        avg_width_pixels = 0
        max_width_pixels = 0

    #Sev class
    if max_width_pixels < 3:
        severity = "Low"
    elif max_width_pixels < 6:
        severity = "Moderate"
    else:
        severity = "Severe"

    #creates the ann image 
    annotated = image.copy()
    annotated[binary == 255] = [0, 0, 255]     # Red crack region
    annotated[skeleton == 255] = [0, 255, 0]   # Green centerline

    output_filename = "annotated_" + file.filename
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    cv2.imwrite(output_path, annotated)

    return JSONResponse({
        "status": "success",
        "crack_length_pixels": crack_length_pixels,
        "avg_width_pixels": avg_width_pixels,
        "max_width_pixels": max_width_pixels,
        "severity": severity,
        "annotated_image_url": f"https://crack-detection-on-bridges.onrender.com/outputs/{output_filename}"
    })