"""
Training Multiple Faces stored on a DataBase:
		==> Each face should have a unique numeric integer ID as 1,2,3 ... 
		==> LBPH computed model will be saved on trainer/ directory 
		==> for using PIL, install pillow library with "pip install pillow"

		Based on original code :
			https://github.com/thecodacus/Face-Recognition    
			https://github.com/Mjrovai/OpenCV-Face-Recognition/blob/master/FacialRecognition/02_face_training.py
"""


import cv2 
import numpy as np 
from PIL import Image 
import os 

def face_training():

	# Path for face image database 
	PATH = './dataset'

	# detector = cv2.CascadeClassifier(r'D:\BackupMega\CurrentBackUp\Studying-2018-2019\Digital_Image_Processing_Computer_Vision\human_\haarcascade\haarcascade_frontalface_alt.xml')
	# LBPH (Loal Binary Patterns Histograms) Face Recognizer 
	recognizer = cv2.face.LBPHFaceRecognizer_create()

	# function to get the images and label data 
	def getImagesAndLabels(path):

		imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
		faceSamples = []
		ids = []

		for imagePath in imagePaths:

			PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
			img_numpy = np.array(PIL_img, 'uint8')

			print('tqwewqeqwe' , os.path, os.path.split(imagePath))
			_id = int(os.path.split(imagePath)[-1].split('_')[1])
		
			faceSamples.append(img_numpy)
			ids.append(_id)

		return faceSamples, ids


	print("\n [INFO] Training faces. It will take a few seconds. Wait .... ")
	faces, ids = getImagesAndLabels(PATH)
	recognizer.train(faces, np.array(ids))

	# Save the model into trainer/trainer.yml
	recognizer.write('trainer/trainer.yml') 

	# Print the number of faces trained and end program 
	print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

if __name__ == '__main__':
	face_training()