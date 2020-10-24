import cv2
import numpy as np 
import os 

def face_recognition():
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read('./trainer/trainer.yml')

	faceCascade = cv2.CascadeClassifier(r'./haarcascade/haarcascade_frontalface_alt.xml')
	font = cv2.FONT_HERSHEY_SIMPLEX 

	# initialize id counter
	id = 0 

	# names related to ids: example ==> Marcelo : id = 1 , etc 
	path = 'dataset'
	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
	names = dict()
	for imagePath in imagePaths:
		name = os.path.split(imagePath)[-1].split('_')[2]
		_id = int(os.path.split(imagePath)[-1].split('_')[1])
		names[_id] = name 
	print('All names', names)

	# initialize and start realtime video capture 
	cam = cv2.VideoCapture(0)

	# set video width and height
	# cam.set(3,640)
	# cam.set(4,480)

	# Define min window size to be recognized as a face 
	# minW = 0.1*cam.get(3)
	# minH = 0.1*cam.get(4)
	minW = 20 
	minH = 20 

	while True:
		ret, img = cam.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (minW, minH))

		for (x,y,w,h) in faces:
			cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
			_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

			# Check if confidence is less than 100 ==> "0" is perfect match 
			if (confidence < 100):
				name = names[_id]
				confidence = "  {0}%".format(round(100-confidence))
			else:
				_id = "unknown"
				confidence = "  {0}%".format(round(100-confidence))

			cv2.putText(img, str(name), (x+5, y-5), font, 1, (0,0,255), 2)
			cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1 , (255,0,0), 1)

		cv2.imshow('camera', img)
		if cv2.waitKey(1) & 0xFF == ord('q'): # press 'p to quit'
			break

	print("\n [INFO] Exiting Program and cleanup stuff")
	cam.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	face_recognition()