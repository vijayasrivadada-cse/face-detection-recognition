# 👁️ Face Detection and Recognition Project

## 📌 Description
This project is a real-time Face Detection and Recognition system using Python and OpenCV. It captures video from a webcam, detects faces, stores face data, and recognizes the person using a trained dataset.

---

## 🚀 Steps to Run

1. **Run `video_read.py`**  
   - This will check whether your webcam is working or not.  
   - A live video window will open.  

---

2. **Run `face_detection.py`**  
   - This checks whether the camera is able to detect your face.  
   - Uses Haar Cascade Classifier to detect faces.  
   - A rectangle will appear around your face.  

---

3. **Run `face_data.py`**  
   - This will open the webcam and collect your face data.  
   - You will be asked to enter your name.  
   - The system captures multiple images of your face.  
   - Face data is stored in the `face_dataset/` folder.
   - The face_dataset folder will be automatically created when running face_data.py.

---

4. **Run `face_recognition.py`**  
   - This will detect your face from the dataset.  
   - Recognizes the person using KNN algorithm.  
   - Displays your name with a bounding box around your face.  

---

## 🛠️ Requirements
Install the required libraries:

```bash
pip install opencv-python numpy
