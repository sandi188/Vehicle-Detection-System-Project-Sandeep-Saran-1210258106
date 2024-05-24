# pip install opencv-contrib-python
# pip install ultralytics

from ultralytics import YOLO, solutions
from ultralytics.utils.plotting import Annotator, colors
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

video_path = r"C:\Users\Lenovo\Documents\vduhm-main\Los Angeles.mp4"
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
# Init heatmap
heatmap_obj = solutions.Heatmap(
    colormap=cv2.COLORMAP_PARULA,
    view_img=True,
    shape="circle",
    classes_names=model.names,
)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
output_path = "output.mp4"
video_writer  = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

while cap.isOpened():
    success, frame = cap.read()
    current_frame += 1
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        tracks = model.track(frame, persist=True, show=False)
        annotated_frame = heatmap_obj.generate_heatmap(frame, tracks)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Display the progress bar
        cv2.rectangle(annotated_frame, (0, height-10), (int(width * current_frame / length), height), (0, 200, 255), -1)
        # % of video completed
        per_remaining = round((current_frame / length) * 100, 2)
        # Display the progress percentage
        cv2.putText(annotated_frame, f"{per_remaining}%", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


        # Display the annotated frame
        # cv2.imshow("YOLOv8 Inference", annotated_frame)
        video_writer.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        print("End of video")
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()