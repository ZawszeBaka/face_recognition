import numpy as np
import cv2 

def face_detection():
	# be careful with the path , if the path is wrong, it causes the problem in detectMultiScale with !empty() error  
	faceCascade = cv2.CascadeClassifier(r'D:\BackupMega\CurrentBackUp\Studying-2018-2019\Digital_Image_Processing_Computer_Vision\human_\haarcascade\haarcascade_frontalface_alt.xml')

	cap = cv2.VideoCapture(0)

	# set width , height
	# cap.set(3,640)
	# cap.set(4,480)

	while True:
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize = (20,20)) # scaleFactor = 1.2, minNeighbors = 5, minSize=(20,20)

		for (x,y,w,h) in faces:
			cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

			# ROI : region of image
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]

		cv2.imshow('video', img)
		if cv2.waitKey(1) & 0xFF == ord('q'): # press 'p to quit'
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	face_detection()