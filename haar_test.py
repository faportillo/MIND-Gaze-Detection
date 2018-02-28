import cv2

CASCADE_HEAD = 'cascadeH5.xml'
CASCADE_ITEM = 'Person Head'

profile_cascade = cv2.CascadeClassifier('haar_profileface.xml') #head cascade
upper_cascade = cv2.CascadeClassifier('cascadG.xml')
eye_cascade = cv2.CascadeClassifier('haar_eyes.xml')

cap = cv2.VideoCapture('46010_9_Synchrony.mpg')


while(cap.isOpened()):
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	prof_rectangles = profile_cascade.detectMultiScale(frame,1.2,5)
	up_rectangles = upper_cascade.detectMultiScale(frame,1.5,10)
	for(i,(x,y,w,h)) in enumerate(prof_rectangles):
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
		'''eyes = eye_cascade.detectMultiScale(frame)
		for (ex,ey,ew,eh) in eyes:
        		cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)'''
	for(i,(x,y,w,h)) in enumerate(up_rectangles):
		'''eyes = eye_cascade.detectMultiScale(frame)
		for (ex,ey,ew,eh) in eyes:
        		cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)'''
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
