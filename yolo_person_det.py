import sys
sys.path.insert(0,'pytorch-yolo2')
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
from detect import *
import cv2
import numpy as np

def detect_cv2_video(cfgfile, weightfile, imgfile):
	m = Darknet(cfgfile)
	m.load_weights(weightfile)
	print('Loading weights from %s... Done!' % (weightfile))
	if m.num_classes == 20:
		namesfile = 'pytorch-yolo2/data/voc.names'
	elif m.num_classes == 80:
		namesfile = 'pytorch-yolo2/data/coco.names'
	else:
		namesfile = 'pytorch-yolo2/data/names'
	class_names = load_class_names(namesfile)
	use_cuda = 1
	if use_cuda:
		m.cuda()

	profile_cascade = cv2.CascadeClassifier('haar_profileface.xml') #head cascade
	eye_cascade = cv2.CascadeClassifier('haar_eyes.xml')

	cap = cv2.VideoCapture(imgfile)
	while(cap.isOpened()):
		res, img = cap.read()
		if res:
			baby_side = img[30:330,np.size(img,1)/2:np.size(img,1)]
			prof_rectangles = profile_cascade.detectMultiScale(baby_side,1.2,5)
			for(i,(x,y,w,h)) in enumerate(prof_rectangles):
				cv2.rectangle(baby_side,(x,y),(x+w,y+h),(0,0,255),2)
				baby_head = baby_side[y:y+h,x:x+w]
				baby_head = cv2.resize(baby_head,(m.width, m.height), interpolation=cv2.INTER_CUBIC)
				print(str(x)+','+str(y)+','+str(x+w)+','+str(y+h))
				#baby_head = cv2.cvtColor(baby_head, cv2.COLOR_BGR2GRAY)
				#baby_head = cv2.Canny(baby_head,0,25)
				eyes = eye_cascade.detectMultiScale(baby_head)
				sum_ex=0
				sum_ey=0
				e_len=0
				for (ex,ey,ew,eh) in eyes:
					sum_ex=sum_ex+((ex+ex+ew)/2)
					sum_ey=sum_ey+((ey+ey+eh)/2)
					e_len=e_len+1
				if e_len:
					centroid_x = np.nan_to_num(sum_ex/e_len)
					centroid_y = np.nan_to_num(sum_ey/e_len)
				cv2.circle(baby_head, (centroid_x,centroid_y),15,(255,255,0))
				for (ex,ey,ew,eh) in eyes:
					cv2.rectangle(baby_head,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
					cv2.line(baby_head,((ex+ex+ew)/2,(ey+ey+eh)/2),(centroid_x,centroid_y),(0,255,0),2)
				cv2.imshow('baby', baby_head)

			img = img[0:330, 0:np.size(img,1)]
			sized = cv2.resize(img, (m.width, m.height))
			bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
			#x1,y1,x2,y2 = bboxes[0],bboxes[1],bboxes[2],bboxes[3]
			print(str(len(bboxes)))
			print('------')
			draw_img = plot_boxes_cv2(img, bboxes, None, class_names=class_names)
			cv2.imshow(cfgfile, draw_img)
		    
		else:
		     print("Unable to read image")
		     exit(-1) 
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

detect_cv2_video('pytorch-yolo2/cfg/yolo.cfg','pytorch-yolo2/yolo.weights','46010_9_Synchrony.mpg')


cap.release()
cv2.destroyAllWindows()
