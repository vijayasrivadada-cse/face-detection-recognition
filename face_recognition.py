import numpy as np
import cv2
import os

########## KNN CODE ############
def distance(v1, v2):
    # Euclidean 
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []
    
    for i in range(train.shape[0]):
        ix = train[i, :-1]     # Features
        iy = train[i, -1]      # Label

        d = distance(test, ix)
        dist.append([d, iy])

    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]

    labels = np.array(dk)[:, -1]

    # Get label with highest frequency
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]
################################

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

dataset_path = "./face_dataset/"

face_data = []
labels = []
class_id = 0
names = {}

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]  # remove .npy
        data_item = np.load(dataset_path + fx)

        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        labels.append(target)
        class_id += 1

# If no npy files exist → print error
if len(face_data) == 0:
    print("[ERROR] No .npy files found in face_dataset/")
    exit()

# Convert list to numpy
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

# Create trainset
trainset = np.concatenate((face_dataset, face_labels), axis=1)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        offset = 5
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = x + w + offset, y + h + offset

        face_section = gray[y1:y2, x1:x2]

        if face_section.size == 0:
            continue

        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())

        name = names[int(out)]
        cv2.putText(frame, name, (x, y - 10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv2.imshow("Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
