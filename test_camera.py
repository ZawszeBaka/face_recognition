import cv2
import numpy as nntplib

def test_camera():
	cap = cv2.VideoCapture(0)

	# set Width 
	# cap.set(3,640)
	# set Height
	# cap.set(4,480)

	while True:
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# flip ?
		# frame = cv2.flip(frame, -1) # Flip camera vertically

		# show image
		cv2.imshow('frame', frame)
		cv2.imshow('gray', gray)

		if cv2.waitKey(1) & 0xFF == ord('q'): # press 'p to quit'
			break

		# or 
		# k = cv2.waitKey(30) & 0xff
		# if k == 27:  # press 'ESC to quit'
		# 	break;

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	test_camera()