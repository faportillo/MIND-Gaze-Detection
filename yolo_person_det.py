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
import math
import torch
import pandas as pd
from skimage import io,transform
import torch.nn as nn
#import torchvision.transforms as transforms
from torch.autograd import Variable
import gazenet_v2 as gn
from gazenet_v2 import GazeNet
import csv

def collision(rleft, rtop, width, height, center_x, center_y, radius): 
	""" Detect collision between a rectangle and circle. """

	rright, rbottom = rleft + width/2, rtop + height/2

	cleft, ctop = center_x-radius, center_y-radius
	cright, cbottom = center_x+radius, center_y+radius

	print(str(rleft)+","+str(rtop)+","+str(rright)+","+str(rbottom)+","+str(cleft)+","+str(ctop)+","+str(cright)+","+str(cbottom))

	if rright < cleft or rleft > cright or rbottom < ctop or rtop > cbottom:
		return False 

	for x in (rleft, rleft+width):
		for y in (rtop, rtop+height):
		    if math.hypot(x-center_x, y-center_y) <= radius:
		        return True  # collision detected

	if rleft <= center_x <= rright and rtop <= center_y <= rbottom:
		return True  # overlaid

	return False  # no collision detected

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

	csvfile = 'gaze_annotations_test.csv'
	
	
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

	frame_count = 0
	radius = 20
	gaze_count = 0
	look_start = 0.0
	look_end = 0.0
	
	cap = cv2.VideoCapture(imgfile)
	while(cap.isOpened()):
		res, img = cap.read()
		
		img = cv2.resize(img,(681,681),interpolation=cv2.INTER_LINEAR)
		bins_x = np.arange(img.shape[0]/13,img.shape[0],img.shape[0]/13) #Bins for quantizing head position
		bins_y = np.arange(img.shape[1]/13,img.shape[1],img.shape[1]/13) #Bins for quantizing head position
		re_img = cv2.resize(img,(227,227))
		heat_img = np.zeros((227,227,1)).astype(np.uint8)
		#print("BINSX: "+str(bins_x))
		#print("BINSY: "+str(bins_y))
		#print(str(kernel))
		'''Booleans to check'''
		has_baby = False
		has_eyes = False
		'''Count Frames to time gaze'''
		frame_count += 1
		is_looking = False

		centroid_x = 0
		centroid_y = 0
		if res:
			#img = cv2.filter2D(img,-1,kernel)
			heatmap = img
			#Do haar cascades 
			'''prof_rectangles = profile_cascade.detectMultiScale(baby_side,1.2,3)
			face_rectangles = face_cascade.detectMultiScale(baby_side,1.2,3)'''
			prof_rectangles = profile_cascade.detectMultiScale(img,1.2,3)
			face_rectangles = face_cascade.detectMultiScale(img,1.2,3)
			#print(str(type(prof_rectangles)))
			#print(str(type(face_rectangles)))
			if type(prof_rectangles) is np.ndarray:
				if prof_rectangles.size >0:
					haar_rectangles = prof_rectangles
			elif type(face_rectangles) is np.ndarray: 
				if face_rectangles.size >0:
					haar_rectangles = face_rectangles
			else:
					haar_rectangles = [0]
			
			cx_prime = 0
			cy_prime = 0
			if type(haar_rectangles) is np.ndarray:
				#Draw rectangles head and retrieve coordinates
				for(i,(x,y,w,h)) in enumerate(haar_rectangles):
					if x>=np.size(img,1)/2 and y>30 and y<=np.size(img,0)*2//3:
						has_baby = True
						
						head_posX = np.digitize(x,bins_x,right=True)
						head_posY = np.digitize(y,bins_y,right=True)
						#print("HEAD POS X,Y:"+str(x)+","+str(y))
						#print("HEAD POS X,Y (QUANTIZED):"+str(head_posX)+","+str(head_posY))
						#cv2.rectangle(img,(x-w/4,y-h/4),(x+w+w/4,y+h+h/4),(0,0,255),2)
						baby_head = img[y-h/3:y+h+h/2,x-w/3:x+w+w/2]#Check right size. **Assumes baby is always on right side of scene
						baby_head = cv2.resize(baby_head,(227, 227), interpolation=cv2.INTER_CUBIC)
						cv2.fastNlMeansDenoisingColored(baby_head,baby_head,2,10,7,21)
						#baby_head=cv2.filter2D(baby_head,-1,kernel)
						
			

			'''
				Do Gazenet stuff here
				check if has_baby and has_eyes are true then pass into network
			'''
			if has_baby:
				
				#print("\n\n\tREIMG Max: " + str(np.amax(re_img)))
				#print(re_img)
				cx_prime = head_posX.astype(float)/13#re_img.shape[0]
				cy_prime = head_posY.astype(float)/13#re_img.shape[1]
				#print("\n\n\tEYEPOSITION"+str((cx_prime,cy_prime)))
				'''HEATMAP STUFF START'''
				heatmap= gn.find_gaze(re_img,baby_head,(cx_prime,cy_prime), get_gaze)
				heat_m = np.asarray(heatmap[2])
				heat_m = np.flipud(heat_m)
				heat_m = np.fliplr(heat_m)
				heat_m_3d = np.reshape(heat_m,(heat_m.shape[0],heat_m.shape[1],1))
				heat_m_3d *= 255*4.5
				heat_img = heat_m_3d.astype(np.uint8)
				'''HEATMAP STUFF END'''
				cx_prime = int(cx_prime*re_img.shape[0])
				cy_prime = int(cy_prime*re_img.shape[1])
				
				gaze_pos = (int(round((1-heatmap[0])*re_img.shape[0])),int(round((1-heatmap[1])*re_img.shape[1])))
				
				cv2.imshow('baby_face', baby_head)
				
					
			
			sized = cv2.resize(img, (m.height, m.width)) #Resize to m.height, m.width
			bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda) #Yolo detection
			
			heat_img = cv2.applyColorMap(heat_img,cv2.COLORMAP_JET)
			combined_image = cv2.addWeighted(re_img,0.5,heat_img,0.5,0)	
			maxLoc = (0,0)
			mask=np.zeros((227,227,1))
			mask = mask.astype(np.uint8)
			if has_baby:		
				red = heat_img[:,:,2]#cv2.cvtColor(heat_img, cv2.COLOR_BGR2GRAY)
				gray = cv2.GaussianBlur(red, (61,61),0)
				(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
				cv2.circle(combined_image, maxLoc, radius, (255, 0, 0), 2)
				nx,ny = gray.shape
				cy,cx = np.ogrid[-maxLoc[0]:nx-maxLoc[0], -maxLoc[1]:ny-maxLoc[1]]
				mask = cx*cx + cy*cy <= radius*radius
				mask = np.transpose(mask.astype(np.uint8)) * 255
				mask = np.reshape(mask,(mask.shape[0],mask.shape[1],1))
				mask = mask.astype(np.uint8).copy()
				cv2.imshow('Circle Mask', mask)
			#print('------')
			for box, (x,y,w,h,conf,max_conf,idx) in enumerate(bboxes):
				if(x<=0.6) and (y>=0.1) and (y<=0.7) and idx==0:
					x1 = int(round((x-w)*mask.shape[0]))
					y1 = int(round((y-h/2.0)*mask.shape[1]))
					x2 = int(round((x+w/2.0+w/8.0)*mask.shape[0]))
					y2 = int(round((y+h/24.0)*mask.shape[1])) #Shorten bounding box
					sized = plot_boxes_cv2(combined_image, [bboxes[box]], None, class_names=class_names)
					if has_baby:
						is_looking = collision(x1,y1,x2,y2,(maxLoc[0]),(maxLoc[1]),radius)
						#print("LOOKING! "+ str(is_looking))
					#print("MASK SHAPE: "+str(x*mask.shape[0]))
					if is_looking:
						cv2.rectangle(mask,(x1,y1),(x2,y2),255,2)
						gaze_count += 1
					else:
						cv2.rectangle(mask,(x1,y1),(x2,y2),100,2)
						gaze_count -=1 if gaze_count>0 else 0
					#cv2.imshow('Circle Mask', mask)
			if gaze_count == 1:
				look_start = cap.get(cv2.CAP_PROP_POS_MSEC)
				#print("LOOKING TIME: " + str(look_start))
			elif gaze_count == 0:
				look_end = cap.get(cv2.CAP_PROP_POS_MSEC)
				if look_start > 0:
					look_duration = look_end - look_start
					if look_duration > 500.0: #Set threshold to eliminate jumpy gazes
						data_row = [imgfile,str(look_start*0.001),str(look_end*0.001),str(look_duration*0.001)]
						with open(csvfile, "a") as output:
							writer = csv.writer(output,lineterminator='\n')
							writer.writerow(data_row)

						look_end = 0.0
						look_start = 0.0
						print("\n\nLOOK DURATION: " + str(look_duration*0.001)+"sec\n\n")
			else:
				print("GAZE FRAME DURATION: "+str(gaze_count))
				
			cv2.imshow('Circle Mask', mask)
			combined_image = cv2.resize(combined_image,(454,454),interpolation=cv2.INTER_LINEAR)
			cv2.imshow('COMBINED', combined_image)
			'''re_img = cv2.resize(re_img,(454,454),interpolation=cv2.INTER_LINEAR)
			cv2.imshow("GAZE",re_img)'''
				
		else:
			 print("Unable to read image")
			 exit(-1) 
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

detect_cv2_video('pytorch-yolo2/cfg/yolo.cfg', 'pytorch-yolo2/yolo.weights','46010_9_Synchrony.mpg')


cap.release()
cv2.destroyAllWindows()
