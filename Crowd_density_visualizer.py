from ultralytics import YOLO
import cv2
import gradio as gr

# Load YOLO model
model = YOLO("yolov8n.pt")  # "yolov8n" : small & fast pretrained model


def analyze_crowd_final(video_path, low_threshold=5, medium_threshold=15):
    """
    Analyze crowd density per frame and create a new video with:
    - person bounding boxes for object recognition
    - text overlay: "People: N | Density: LEVEL (Message)"
    - summary statistics about number of people 

    Returns:
    - output_video_path: processed video file
    - summary_text: string containing global stats
    """

    # Video doesn't come over to Gradio
    if video_path is None:
        return None, "No video provided."

    # Open video by using Open CV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Cannot open video."

    # Information about FPS, screen resolution
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setting about result video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path = "crowd_result.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Resetting statistical variable.
    frame_index = 0
    total_person_count = 0
    max_person_count = 0
    high_frames = 0

    # Read video one frame at a time and process it 
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video (No remained frames)

#----------------------Yolo inference part------------------------------

        # COCO dataset criteria class 0 = "person" only detected
        results = model(frame, classes=[0])
        boxes = results[0].boxes
        person_count = len(boxes)

        # Update statistical value
        total_person_count += person_count
        if person_count > max_person_count:
            max_person_count = person_count

#----------------------Distribution about density logic------------------------------

        # Determine density level and message
        if person_count <= low_threshold:
            density_level = "LOW"
            message = "Comfortable"
            color = (0, 255, 0)      # green
        elif person_count <= medium_threshold:
            density_level = "MEDIUM"
            message = "Busy"
            color = (0, 255, 255)    # yellow
        else:
            density_level = "HIGH"
            message = "Crowded"
            color = (0, 0, 255)      # red
            high_frames += 1

        # Get annotated frame with boxes
        annotated_frame = results[0].plot()

        # Text about above info
        text = f"People: {person_count} | Density: {density_level} ({message})"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness  = 2

        # Calculate text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Text position
        x, y = 20, 40

        # Text background box position
        box_x1 = x - 10
        box_y1 = y - text_h - 10
        box_x2 = x + text_w + 10
        box_y2 = y + 10

        overlay = annotated_frame.copy()
        cv2.rectangle(
            overlay,
            (box_x1, box_y1),
            (box_x2, box_y2),
            (40, 40, 40),  # dark gray
            -1             # filled
        )
        alpha = 0.6  # transparency
        annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)

        cv2.putText(
            annotated_frame,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        # Write frame to output video
        out.write(annotated_frame)

        frame_index += 1

    cap.release()
    out.release()

#----------------------Calculation of statistical value about total video------------------------------

    total_frames = frame_index
    if total_frames == 0:
        # Frame cannot be processed 
        return output_video_path, "Error: no frames processed."

    # Average of number of people
    avg_person_count = total_person_count / total_frames
    # Video time 
    duration_sec = total_frames / fps if fps and fps > 0 else 0.0
    # High density frame ratio
    high_ratio = (high_frames / total_frames) * 100.0

    # Statistical text lists
    summary_lines = [
        "=== Crowd Density Summary ===",
        f"Total frames: {total_frames}",
        f"Video duration: {duration_sec:.2f} seconds",
        f"Max people in a frame: {max_person_count}",
        f"Average people per frame: {avg_person_count:.2f}",
        f"High density frames: {high_frames} ({high_ratio:.1f}% of frames)",
        "",
        "Density levels:",
        f"- LOW    (<= {low_threshold} people)   : Comfortable",
        f"- MEDIUM ({low_threshold+1}–{medium_threshold} people): Busy",
        f"- HIGH   (> {medium_threshold} people) : Crowded",
    ]
    summary_text = "\n".join(summary_lines)

    return output_video_path, summary_text


#----------------------Gradio UI------------------------------

def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Crowd Density Visualizer (by Yujin Kim)

            Upload a short video to analyze crowd density.  
            The app will:
            - detect **people** in each frame using YOLOv8  
            - draw bounding boxes to identify object
            - overlay text: `People: N | Density: LEVEL (Message)`  
              - LOW    → Comfortable  
              - MEDIUM → Busy  
              - HIGH   → Crowded  
            - compute summary statistics for the whole video
            """
        )
        # Input and outpuy video 
        video_input = gr.Video(label="Input video", sources=["upload"])
        video_output = gr.Video(label="Processed video")
        # Summary text box abour statistical value
        summary_box = gr.Textbox(
            label="Summary statistics",
            lines=10,
            interactive=False,
        )
        # Control crown density standard
        low_thr = gr.Slider(0, 50, value=5, step=1, label="Low density threshold (max people)")
        med_thr = gr.Slider(1, 100, value=15, step=1, label="Medium density threshold (max people)")

        # Start analysis
        analyze_button = gr.Button("Run Crowd Analysis")

        analyze_button.click(
            fn=analyze_crowd_final,
            inputs=[video_input, low_thr, med_thr],
            outputs=[video_output, summary_box],
        )

    return demo


if __name__ == "__main__":
    # Apply python code to "Gradio"
    demo = build_interface()
    demo.launch()
