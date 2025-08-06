import cv2
from ultralytics import YOLO

# Load model (use yolov8n.pt for speed on CPU)
model = YOLO('yolov8n.pt')

# Open your webcam (device 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 prediction on the frame
    results = model.predict(frame, device='cpu', verbose=False)  # use your choice of device

    # Check if a person (class 0 in COCO) is detected
    boxes = results[0].boxes
    detected = False
    for box in boxes:
        cls = int(box.cls[0])
        if cls == 0:  # 0 is the class index for person in COCO
            print("person detected")
            detected = True
            break

    # Get the annotated frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()