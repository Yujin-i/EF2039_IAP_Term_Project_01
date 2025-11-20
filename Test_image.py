from ultralytics import YOLO
import cv2

def main():
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt") 

    image_path = "test_image.jpg"  
    print(f"Running inference on: {image_path}")

    results = model(image_path)  
    annotated_frame = results[0].plot()

    output_path = "result_test.jpg"
    cv2.imwrite(output_path, annotated_frame)

    print(f"Done! Result saved as {output_path}")

if __name__ == "__main__":
    main()
