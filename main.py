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

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(file_path)

    if image is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(edges, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    clean_mask = np.zeros(binary.shape, dtype=np.uint8)

    image_area = image.shape[0] * image.shape[1]
    min_area = int(0.0001 * image_area)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            clean_mask[labels == i] = 255

    binary = clean_mask

    binary_bool = binary > 0
    skeleton = skeletonize(binary_bool)
    skeleton = (skeleton * 255).astype(np.uint8)

    crack_length_pixels = int(np.sum(skeleton == 255))

    distance_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    width_values = distance_map[skeleton == 255]

    if len(width_values) > 0:
        avg_width_pixels = float(np.mean(width_values) * 2)
        max_width_pixels = float(np.max(width_values) * 2)
    else:
        avg_width_pixels = 0.0
        max_width_pixels = 0.0

    mm_per_pixel = 0.2

    crack_length_mm = crack_length_pixels * mm_per_pixel
    avg_width_mm = avg_width_pixels * mm_per_pixel
    max_width_mm = max_width_pixels * mm_per_pixel

    if max_width_mm < 0.3:
        severity = "Low"
    elif max_width_mm < 1.0:
        severity = "Moderate"
    else:
        severity = "Severe"

    annotated = image.copy()
    annotated[binary == 255] = [0, 0, 255]
    annotated[skeleton == 255] = [0, 255, 0]

    output_filename = "annotated_" + file.filename
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    cv2.imwrite(output_path, annotated)

    return JSONResponse({
        "status": "success",
        "crack_length_mm": round(crack_length_mm, 2),
        "avg_width_mm": round(avg_width_mm, 3),
        "max_width_mm": round(max_width_mm, 3),
        "severity": severity,
        "annotated_image_url": f"https://crack-detection-on-bridges.onrender.com/outputs/{output_filename}"
    })