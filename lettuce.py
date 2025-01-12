import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("lettuce.pt")  # Update "lettuce.pt" with your path to the model

# Open the video capture
cap = cv2.VideoCapture("test3.mp4")  # Update "video.mp4" with your video path

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame is not read successfully break the loop
    if not ret:
        break

    # Perform detection with a high confidence threshold (optional)
    results = model(frame, conf=0.1)  # Adjust confidence threshold if needed

    # Process results and draw bounding boxes
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers

            # Get class and confidence (if available)
            try:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                # Display class name and confidence (optional)
                label = f"{class_name}"
                cv2.putText(frame, label, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except (IndexError, AttributeError):
                # Handle cases where confidence is not available
                class_name = "Unknown"
                label = f"{class_name}"
                cv2.putText(frame, label, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("YOLO Object Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()