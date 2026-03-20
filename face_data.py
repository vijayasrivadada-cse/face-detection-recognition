import cv2
import numpy as np
import os

# Create dataset folder if it does not exist
dataset_path = "face_dataset/"
if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

# Ask for name
file_name = input("Enter the name of the person: ")

# Start camera
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []

print("\n[INFO] Collecting data... Look at the camera.\nPress 'q' to stop.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Sort by largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # Extract face ROI
    offset = 10
    x1, y1 = max(0, x - offset), max(0, y - offset)
    x2, y2 = x + w + offset, y + h + offset
    face_section = gray_frame[y1:y2, x1:x2]

    # Resize to 100x100
    face_section = cv2.resize(face_section, (100, 100))

    # Save every 10th frame
    skip += 1
    if skip % 10 == 0:
        face_data.append(face_section)
        print(f"Saved sample: {len(face_data)}")

    # Show output windows
    cv2.imshow("Face", face_section)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Convert list to numpy array
face_data = np.array(face_data)
print("Dataset Shape:", face_data.shape)

# Reshape for saving
face_data = face_data.reshape((face_data.shape[0], -1))

# Save dataset file
np.save(dataset_path + file_name + ".npy", face_data)
print(f"\n[INFO] Dataset saved: {dataset_path}{file_name}.npy\n")

cap.release()
cv2.destroyAllWindows()