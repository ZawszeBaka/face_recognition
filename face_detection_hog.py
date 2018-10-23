import sys
import dlib 

def face_detection_hog():

	face_detector = dlib.get_frontal_face_detector()

	cap = cv2.VideoCapture(0)

	# set width , height
	# cap.set(3,640)
	# cap.set(4,480)

	while True:
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		faces = face_detector(gray, 1)

		for i, face_rect in faces:

			
			
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

