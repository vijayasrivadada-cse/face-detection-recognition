
import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Camera not detected!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        continue

    # Convert only AFTER frame is valid
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        cv2.imshow("faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for face in faces[:1]:
        x, y, w, h = face

        offset = 10
        face_offset = frame[y-offset:y+h+offset, x-offset:x+w+offset]

        # Protect against out-of-bound slicing
        if face_offset.size == 0:
            continue

        face_selection = cv2.resize(face_offset, (100, 100))

        cv2.imshow("Face", face_selection)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("faces", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()