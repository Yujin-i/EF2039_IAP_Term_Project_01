from ultralytics import YOLO
import cv2

def main():

    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    input_video_path = "test_video.mp4"   
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file: {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    output_video_path = "result_video.mp4"   
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Start processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  

        results = model(frame)
        annotated_frame = results[0].plot()  

        out.write(annotated_frame)

       
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Done! Result video saved as: {output_video_path}")

if __name__ == "__main__":
    main()
