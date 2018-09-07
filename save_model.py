import cv2
import glob
import math
import numpy as np
import dlib
from sklearn.svm import SVC
from sklearn.externals import joblib

# get some required objects
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

emotions = ["anger", "laugh", "neutral", "profile_l", "profile_r", "smile", "smirk_l", "smirk_r", "surprise", "timid", "yawn"] #Define emotions
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
data = {} #Make dictionary for all values
data['landmarks_vectorised'] = []

# return files under the given emotion in the dataset
def get_files(emotion): 
	files = glob.glob("exp_dataset/%s/*" %emotion)
	return files

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

# make the data sets and labels to be fed into training
def make_sets():
    training_data = []
    training_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        training = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                training_labels.append(
                emotions.index(emotion))
    return training_data, training_labels

# script begins here
print("Collecting files") 
training_data, training_labels = make_sets()
npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
npar_trainlabs = np.array(training_labels)
print("Training SVM linear") #train SVM
clf.fit(npar_train, training_labels)
# save model to directory
print('Saving model')
filename = "finalized_test_model.sav"
joblib.dump(clf, filename)

