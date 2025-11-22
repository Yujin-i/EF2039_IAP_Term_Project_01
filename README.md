# Crowd Density Visualizer

A simple web UI AI modeling based on YOLOv8 from ultralytics, detect **People** in a video frame, classfies crowd density level (low, medium, high) and show summary statistics about output video. 

# Requirements

- Python 3.10+
- Dependencies:
  - ultralytics (YOLOv8 required)
  - opencv-python
  - gradio

Install all with:

bash
pip install -r requirements.txt

# YOLO Model from Ultralytics

This project uses a pretrained YOLOv8n model.

Download yolov8n.pt from Ultralytics:
https://github.com/ultralytics/assets/releases 
Put yolov8n.pt in the same folder as Crowd_density_visualizer.py.

or 

pip install -r requirements.txt

# How to run

bash
python Crowd_density_visualizer.py

Open the gradio link shown in the terminal.
And in web UI, upload a short video and click "Run Crowd Analysis"
