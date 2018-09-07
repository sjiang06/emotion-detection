# optimized trained_webcam
# does not do as thorough search of faces
# when cropping the frame

import cv2
import glob
import math
import numpy as np
import dlib
import datetime
from imutils.video import VideoStream
from imutils import face_utils
import imutils
from sklearn.externals import joblib

# get some required objects
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

print("[INFO] camera sensor warming up...")
vs = VideoStream().start() #Webcam object

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotions = ["anger", "laugh", "neutral", "profile_l", "profile_r", "smile", "smirk_l", "smirk_r", "surprise", "timid", "yawn"]
filename = 'finalized_test_model.sav' # filename of model stored in directory
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
data = {} #Make dictionary for all values
data['landmarks_vectorised'] = []

#get the 68 facial landmarks 
def get_landmarks(image):
	detections = detector(image, 1)
	for k,d in enumerate(detections): #For all detected face instances individually
		shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
		xlist = []
		ylist = []
		for i in range(1,68): #Store X and Y coordinates in two lists
			xlist.append(float(shape.part(i).x))
			ylist.append(float(shape.part(i).y))
		xmean = np.mean(xlist)
		ymean = np.mean(ylist)
		xcentral = [(x-xmean) for x in xlist]
		ycentral = [(y-ymean) for y in ylist]
		landmarks_vectorised = []
		for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
			landmarks_vectorised.append(w)
			landmarks_vectorised.append(z)
			meannp = np.asarray((ymean,xmean))
			coornp = np.asarray((z,w))
			dist = np.linalg.norm(coornp-meannp)
			landmarks_vectorised.append(dist)
			landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
		data['landmarks_vectorised'] = landmarks_vectorised
	if len(detections) < 1:
		data['landmarks_vestorised'] = "error"

#crop the input photo to same size as the training photos
def crop_input_photo(image) :
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
	#Detect face using 4 different classifiers
	face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
	if len(face) == 1:
		facefeatures = face
	else:
		facefeatures = ""
	#Cut and save face
	for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
		gray = gray[y:y+h, x:x+w] #Cut the frame to size
		try:
			out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
			return out          
		except:
		   return

#webcam script begins here
loaded_model = joblib.load(filename)

while(True) :
	tally = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

	# take the average of five predictions to optimize classification
	key = 0
	for x in range(0, 7) :

		frame = vs.read()
		frame = imutils.resize(frame, width=600)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		clean_frame = frame.copy()
	 
		# detect faces in the grayscale frame
		rects = detector(gray, 0)

		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		if(len(rects) == 0): continue
		shape = predictor(gray, rects[0])
		shape = face_utils.shape_to_np(shape)
 
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	 
		out = crop_input_photo(clean_frame)
		if(out is None) :
			print("No face yet detected, continuing...")
			continue
		else :
			print("Face detected")
		clahe_image = clahe.apply(out)
		get_landmarks(clahe_image)
		prediction_data = []

		if data['landmarks_vectorised'] == "error" or len(data['landmarks_vectorised']) == 0:
			print("No face detected on this one")
		else:
			print("Face detected, now predicting...")
			prediction_data.append(data['landmarks_vectorised'])
			array = np.array(prediction_data)
			pred_pro = loaded_model.predict(array)
			tally[pred_pro[0]] += 1

	if(key == ord("q")) : break
	max_tally = max(tally)
	if(max_tally != 0) : 
		print("Emotion:", emotions[tally.index(max_tally)])
	tally = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

cv2.destroyAllWindows()
vs.stop()