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

# Allow frontend communication
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

   
    # Save Uploaded Image
 
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(file_path)

    if image is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    
    # Convert to Grayscale
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

 
    # Adaptive Threshold
  
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        2
    )

    
    # Morphological Cleanup
  
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

   
    # Remove Small / Non-Crack Components
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    clean_mask = np.zeros(binary.shape, dtype=np.uint8)

    image_area = image.shape[0] * image.shape[1]
    min_area = int(0.0005 * image_area)  # auto scale threshold

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        if height == 0 or width == 0:
            continue

        aspect_ratio = max(width, height) / min(width, height)

        # Keep only large AND long-thin structures
        if area > min_area and aspect_ratio > 3:
            clean_mask[labels == i] = 255

    binary = clean_mask


    # Skeletonization
    
    binary_bool = binary > 0
    skeleton = skeletonize(binary_bool)
    skeleton = (skeleton * 255).astype(np.uint8)

    
    # Crack Length (pixels)
   
    crack_length_pixels = int(np.sum(skeleton == 255))

   
    # Crack Width Measurement
   
    distance_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    width_values = distance_map[skeleton == 255]

    if len(width_values) > 0:
        avg_width_pixels = float(np.mean(width_values) * 2)
        max_width_pixels = float(np.max(width_values) * 2)
    else:
        avg_width_pixels = 0.0
        max_width_pixels = 0.0

   
    # Severity Classification
    
    if max_width_pixels < 3:
        severity = "Low"
    elif max_width_pixels < 6:
        severity = "Moderate"
    else:
        severity = "Severe"

    
    # Create Annotated Image
  
    annotated = image.copy()

    # Red = crack region
    annotated[binary == 255] = [0, 0, 255]

    # Green = skeleton centerline
    annotated[skeleton == 255] = [0, 255, 0]

    output_filename = "annotated_" + file.filename
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    cv2.imwrite(output_path, annotated)

 
    # Return Results
   
    return JSONResponse({
        "status": "success",
        "crack_length_pixels": crack_length_pixels,
        "avg_width_pixels": round(avg_width_pixels, 2),
        "max_width_pixels": round(max_width_pixels, 2),
        "severity": severity,
        "annotated_image_url": f"https://crack-detection-on-bridges.onrender.com/outputs/{output_filename}"
    })