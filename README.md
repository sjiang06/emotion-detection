This project classifies human emotions from a live webcam. Uses python cv2, imutil, and sklearn libraries.

ARCHIVE
Contains images for each emotion.

crop_faces.py
Detects the human faces from images in a file and outputs the cropped faces. Used in save_model.py

extract_images.py
Sorts the cropped images into their respective emotion label files. Used in save_model.py

facial_landmarks.py
python script used to test facial landmark predictor on a single still image.

finalized_test_model.sav
SVM model that webcam runs on to detect emotions.

trained_webcam_2.py
Runs script on model to detect emotions from live webcam.
