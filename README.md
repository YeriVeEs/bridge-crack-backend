This is a FastAPI backend project for Bridge Crack Detection using YOLOv8 segmentation.

## Getting Started

First, run the development server:

```bash
uvicorn main:app --reload
```

Make sure you have installed dependencies first:

```bash
pip install -r requirements.txt
```

Open http://127.0.0.1:8000 with your browser to see the result.

You can also view the interactive API documentation at:

http://127.0.0.1:8000/docs

You can start editing the backend by modifying `main.py`. The server auto-reloads as you make changes.

This project uses the Ultralytics YOLOv8 model for crack segmentation and FastAPI for building high-performance APIs.

## Learn More

To learn more about the technologies used, take a look at the following resources:

* FastAPI Documentation: https://fastapi.tiangolo.com/ - learn about FastAPI features and API development.
* Ultralytics YOLO: https://docs.ultralytics.com/ - learn about YOLOv8 and computer vision models.

You can check out the FastAPI GitHub repository:
https://github.com/tiangolo/fastapi

## Deploy on Render

This backend is deployed on Render:

https://crack-detection-on-bridges.onrender.com/

The easiest way to deploy this FastAPI app is to use the Render platform.

Check out Render’s documentation for more details:
https://render.com/docs
