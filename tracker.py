import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import datetime
from flask import Flask, render_template, Response

app = Flask(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
heat_map_annotator = sv.HeatMapAnnotator()

stop_processing = False
unique_vehicle_ids = set()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    global unique_vehicle_ids

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # Update the set of unique vehicle IDs
    unique_vehicle_ids.update(detections.tracker_id)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels
    )
    annotated_frame = trace_annotator.annotate(
        annotated_frame, detections=detections)
    annotated_frame = heat_map_annotator.annotate(
        annotated_frame, detections=detections
    )
    # Add text on top
    cv2.putText(annotated_frame, "Vehicle Detection using heatmaps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Add date and time
    current_date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    cv2.putText(annotated_frame, current_date, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Display the number of unique vehicles detected
    cv2.putText(annotated_frame, f"Unique Vehicles: {len(unique_vehicle_ids)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return cv2.imencode('.jpg', annotated_frame)[1].tobytes()

def process_video(source_path: str):
    global stop_processing

    cap = cv2.VideoCapture(source_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_processing:
            break
        frame_bytes = callback(frame, cap.get(cv2.CAP_PROP_POS_FRAMES))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_processing = True
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_video(r"C:\Users\Lenovo\Documents\vduhm-main\Los Angeles.mp4"), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)