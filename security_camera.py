import cv2
import numpy as np
import os
import time
from tflite_runtime.interpreter import Interpreter

def load_labels(path="/home/ArikCool/my_proj/labelmap.txt"):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def initialize_model(model_path="/home/ArikCool/my_proj/1.tflite"):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def security_camera():
    interpreter = initialize_model()
    labels = load_labels()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Create directory for saving videos
    save_dir = "/home/ArikCool/SaveVid"
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(
        "libcamerasrc ! video/x-raw,format=RGB,width=320,height=240,framerate=5/1 ! videoconvert ! appsink",
        cv2.CAP_GSTREAMER
    )
    if not cap.isOpened():
        print("Error: Unable to access the camera!")
        return

    print("Starting security camera... Press 'q' to exit.")

    recording = False
    out = None
    last_detection_time = None
    recording_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame!")
            break

        resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
        input_data = np.expand_dims(resized_frame, axis=0)

        if input_dtype == np.uint8:
            input_data = input_data.astype(np.uint8)
        else:
            input_data = input_data.astype(np.float32) / 255.0

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        detected = False

        for i in range(len(scores)):
            confidence = scores[i]
            if confidence > 0.7:
                detected = True
                label_id = int(class_ids[i])
                label = labels[label_id] if 0 <= label_id < len(labels) else "Unknown"

                ymin, xmin, ymax, xmax = boxes[i]
                height, width, _ = frame.shape
                x, y, x2, y2 = (xmin * width, ymin * height, xmax * width, ymax * height)
                cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence * 100:.1f}%)", (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Handle video recording
        current_time = time.time()
        if detected:
            last_detection_time = current_time
            if not recording:
                # Start recording
                recording = True
                recording_start_time = current_time
                video_filename = os.path.join(
                    save_dir, f"recording_{int(recording_start_time)}.avi"
                )
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                out = cv2.VideoWriter(video_filename, fourcc, 10.0, (frame.shape[1], frame.shape[0]))
                print(f"Started recording: {video_filename}")

        if recording:
            out.write(frame)
            if last_detection_time and current_time - last_detection_time > 5:
                # Stop recording after 5 seconds of no detection
                print(f"Stopping recording: {video_filename}")
                recording = False
                out.release()

        cv2.imshow("Security Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out and recording:
        out.release()
    cv2.destroyAllWindows()
    print("Security camera stopped.")

if __name__ == "__main__":
    security_camera()
