from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import numpy as np
import base64
from skimage.morphology import skeletonize
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load trained YOLO crack model
model = YOLO("runs/segment/train5/weights/best.pt")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(file_path)

    if image is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
   
    # YOLO DETECTION
    

    results = model.predict(file_path, conf=0.25, imgsz=640)

    binary = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    if results[0].masks is not None:

        masks = results[0].masks.data.cpu().numpy()

        for mask in masks:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask = (mask > 0.5).astype(np.uint8) * 255
            binary = cv2.bitwise_or(binary, mask)

    # -------------------
    # CLEAN MASK
    # -------------------

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    clean_mask = np.zeros(binary.shape, dtype=np.uint8)

    image_area = image.shape[0] * image.shape[1]
    min_area = int(0.0001 * image_area)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            clean_mask[labels == i] = 255

    binary = clean_mask

    # -------------------
    # SKELETONIZE CRACK
    # -------------------

    binary_bool = binary > 0
    skeleton = skeletonize(binary_bool)
    skeleton = (skeleton * 255).astype(np.uint8)

    crack_length_pixels = int(np.sum(skeleton == 255))

    # -------------------
    # WIDTH CALCULATION
    # -------------------

    distance_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    width_values = distance_map[skeleton == 255]

    if len(width_values) > 0:
        avg_width_pixels = float(np.mean(width_values) * 2)
        max_width_pixels = float(np.max(width_values) * 2)
    else:
        avg_width_pixels = 0.0
        max_width_pixels = 0.0

    # -------------------
    # PIXEL → MM
    # -------------------

    mm_per_pixel = 0.2

    crack_length_mm = crack_length_pixels * mm_per_pixel
    avg_width_mm = avg_width_pixels * mm_per_pixel
    max_width_mm = max_width_pixels * mm_per_pixel

    # -------------------
    # SEVERITY
    # -------------------

    if max_width_mm < 0.3:
        severity = "Low"
    elif max_width_mm < 1.0:
        severity = "Moderate"
    else:
        severity = "Severe"

    # -------------------
    # CREATE ANNOTATED IMAGE
    # -------------------

    annotated = image.copy()
    annotated[binary == 255] = [0, 0, 255]
    annotated[skeleton == 255] = [0, 255, 0]

    _, buffer = cv2.imencode(".png", annotated)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "status": "success",
        "crack_length_mm": round(crack_length_mm, 2),
        "avg_width_mm": round(avg_width_mm, 3),
        "max_width_mm": round(max_width_mm, 3),
        "severity": severity,
        "annotated_image": image_base64
    })