import sys
sys.path.insert(0,'pytorch-yolo2')
import time
from PIL import Image, ImageDraw
#from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
from detect import *
import cv2
import numpy as np

import torch
import pandas as pd
from skimage import io,transform
import torch.nn as nn
#import torchvision.transforms as transforms
from torch.autograd import Variable
import gazenet_v2 as gn
from gazenet_v2 import GazeNet

def detect_cv2_video(cfgfile,weightfile,imgfile):
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

	#GAZENET MODEL
	get_gaze = GazeNet()
	get_gaze.cuda()

	#Create haar cascades
	profile_cascade = cv2.CascadeClassifier('haar_profileface.xml') #head cascade
	face_cascade = cv2.CascadeClassifier('haar_frontalface_default.xml') #face cascade
	eye_cascade = cv2.CascadeClassifier('haar_eyes.xml')

	#Sharpen Kernel
	kernel = np.zeros((3,3),np.float32)
	kernel[1,1] = 5.0
	kernel[0,1] = -1.0
	kernel[1,0] = -1.0
	kernel[1,2] = -1.0
	kernel[2,1] = -1.0
	
	edge_k = -1*np.ones((3,3),np.float32)
	edge_k[1,1] = 8

	cap = cv2.VideoCapture(imgfile)
	while(cap.isOpened()):
		res, img = cap.read()
		#print(str(kernel))
		has_baby = False
		has_eyes = False
		if res:
			#img = cv2.filter2D(img,-1,kernel)
			heatmap = img
			baby_side = img[30:np.size(img,0)*2//3,np.size(img,1)/2:np.size(img,1)]
			
			#Do haar cascades 
			prof_rectangles = profile_cascade.detectMultiScale(baby_side,1.2,3)
			face_rectangles = face_cascade.detectMultiScale(baby_side,1.2,3)
			print(str(type(prof_rectangles)))
			print(str(type(face_rectangles)))
			if type(prof_rectangles) is np.ndarray:
				if prof_rectangles.size >0:
					haar_rectangles = prof_rectangles
			elif type(face_rectangles) is np.ndarray: 
				if face_rectangles.size >0:
					haar_rectangles = face_rectangles
			else:
					haar_rectangles = [0]

			if type(haar_rectangles) is np.ndarray:
				#Draw rectangles head and retrieve coordinates
				for(i,(x,y,w,h)) in enumerate(haar_rectangles):
					has_baby = True
					cv2.rectangle(baby_side,(x,y),(x+w,y+h),(0,0,255),2)
					baby_head = baby_side[y:y+h,x:x+w]#Check right size. **Assumes baby is always on right side of scene
					baby_head = cv2.resize(baby_head,(227, 227), interpolation=cv2.INTER_CUBIC)
					#baby_head=cv2.filter2D(baby_head,-1,kernel)
				
					#cv2.fastNlMeansDenoisingColored(baby_head,baby_head,2,10,7,21)
				
					print(str(x)+','+str(y)+','+str(x+w)+','+str(y+h))
					#baby_head = cv2.cvtColor(baby_head, cv2.COLOR_BGR2GRAY)
					#baby_head = cv2.Canny(baby_head,0,25)
				
					#Haar cascade for eyes 
					eyes = eye_cascade.detectMultiScale(baby_head)
					sum_ex=0
					sum_ey=0
					e_len=0
					for (ex,ey,ew,eh) in eyes:
						if e_len<=1:
							sum_ex=sum_ex+((ex+ex+ew)/2)
							sum_ey=sum_ey+((ey+ey+eh)/2)
						else:
							break
						e_len=e_len+1
					#Compute centroid of all eye points detected to determine eye position
					if e_len:
						centroid_x = np.nan_to_num(sum_ex/e_len)
						centroid_y = np.nan_to_num(sum_ey/e_len)
						cv2.circle(baby_head, (centroid_x,centroid_y),15,(255,255,0))
					ctr=0
					#Draw rectangles for first two eye cascades detected, assumes those are the strongest features detected
					for (ex,ey,ew,eh) in eyes:
						angle = np.absolute(np.arctan2(centroid_x-(ex+ex+ew)/2,(ey+ey+eh)/2-centroid_y))
						if ctr<=1:
							cv2.rectangle(baby_head,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
							cv2.line(baby_head,((ex+ex+ew)/2,(ey+ey+eh)/2),(centroid_x,centroid_y),(0,255,0),2)
							eye_position = (centroid_x,centroid_y)
							has_eyes = True
						else:
							break
						ctr=ctr+1
			if has_eyes and has_baby:
				re_img = cv2.resize(img,(227,227))
				#re_img -= cv2.mean(re_img)
				print(img)
				heatmap = gn.find_gaze(re_img,baby_head,eye_position, get_gaze)
				cv2.imshow('baby_face', baby_head)
				'''
					Do Gazenet stuff here
					check if has_baby and has_eyes are true then pass into network

				'''
			#img = img[0:330, 0:np.size(img,1)]
			sized = cv2.resize(img[0:330, 0:np.size(img,1)*2//3], (m.height, m.width)) #Resize to m.height, m.width
			bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda) #Yolo detection
			#x1,y1,x2,y2 = bboxes[0],bboxes[1],bboxes[2],bboxes[3]
			print(str(len(bboxes)))
			print('------')

			##Find way to draw boxes on global IMG instead of SIZED img
			draw_img = plot_boxes_cv2(sized, bboxes, None, class_names=class_names)
			cv2.imshow(cfgfile, draw_img) #display main scene
				
		else:
			 print("Unable to read image")
			 exit(-1) 
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

detect_cv2_video('pytorch-yolo2/cfg/yolo.cfg', 'pytorch-yolo2/yolo.weights','46010_9_Synchrony.mpg')


cap.release()
cv2.destroyAllWindows()
