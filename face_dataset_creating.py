import cv2
import os 

def face_dataset_creating():

	cam = cv2.VideoCapture(0)

	# set width, height
	# cam.set(3,640)
	# cam.set(4,480)

	face_detector = cv2.CascadeClassifier(r'./haarcascade/haarcascade_frontalface_alt.xml')

	# for each person, enter one numeric face id 
	face_id = input('\n enter user id and press <return> ==>   ')
	face_name = input('\n enter user name and press <return> ==>   ')
	print('\n [INFO] Initializing face capture. Look the camera and wait ... ')


	# Initialize individual sampling face count
	count = 0 
	while(True):
		ret, img = cam.read()

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_detector.detectMultiScale(gray, 1.3, 5)

		if len(faces) == 1:
			for (x,y,w,h) in faces:
				cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
				count += 1

				# Save the captured image into the datasets folder 
				print('id_' + str(count) + " saved ")
				cv2.imwrite("dataset/User_" + str(face_id) + '_' + face_name + '_' + str(count) + '.jpg', gray[y:y+h, x:x+w])
				
		cv2.imshow('image', img)

		if cv2.waitKey(1) & 0xFF == ord('q'): # press 'p to quit'
			break

	print("\n [INFO] Exiting Program and cleanup stuff")
	cam.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	face_dataset_creating()