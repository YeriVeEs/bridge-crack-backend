import os
import cv2
import numpy as np
import random
import shutil

BASE_PATH = "dataset/DATASET 2020"
OUTPUT_PATH = "datasets/bridge_cracks"

FOLDERS = [
    "Steel crack images",
    "Non-steel crack images"
]

def convert_mask_to_yolo(mask_path, image_shape):
    mask = cv2.imread(mask_path, 0)

    if mask is None:
        return []

    mask = cv2.bitwise_not(mask)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = image_shape[:2]
    segments = []

    for contour in contours:
        if len(contour) < 10:
            continue

        contour = contour.squeeze()

        if len(contour.shape) != 2:
            continue

        segment = []

        for point in contour:
            x = point[0] / width
            y = point[1] / height
            segment.append(f"{x:.6f} {y:.6f}")

        segments.append("0 " + " ".join(segment))

    return segments


all_data = []

for folder in FOLDERS:
    image_dir = os.path.join(BASE_PATH, folder, "Image")
    mask_dir = os.path.join(BASE_PATH, folder, "Label")

    for filename in os.listdir(image_dir):
        base = os.path.splitext(filename)[0]

        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, base + ".png")

        if not os.path.exists(mask_path):
            continue

        all_data.append((image_path, mask_path, filename, base))

random.shuffle(all_data)

split_index = int(0.8 * len(all_data))
train_data = all_data[:split_index]
val_data = all_data[split_index:]


def save_data(data, split):
    for image_path, mask_path, filename, base in data:

        image = cv2.imread(image_path)
        segments = convert_mask_to_yolo(mask_path, image.shape)

        if len(segments) == 0:
            continue

        image_output = f"{OUTPUT_PATH}/images/{split}/{filename}"
        label_output = f"{OUTPUT_PATH}/labels/{split}/{base}.txt"

        shutil.copy(image_path, image_output)

        with open(label_output, "w") as f:
            for seg in segments:
                f.write(seg + "\n")


os.makedirs(f"{OUTPUT_PATH}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/images/val", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/labels/val", exist_ok=True)

save_data(train_data, "train")
save_data(val_data, "val")

print("Dataset conversion complete.")
