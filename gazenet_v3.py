from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io,transform
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import numpy as np
import sys
from torch.nn._functions.padding import ConstantPadNd
import torch.nn.init as init
#import matplotlib.pyplot as plt
#Uncomment if using build < 0.4
from LocalResponseNorm import LRN

import DataProcess as dp
from torchvision import utils

'''
	TODO:
		Implement data loader 
		Display images and heatmaps for outputs
		Implement automatic head detection
			Crop out head and upsample to 224x224x1
			Haar-like features??
		Implement automatic eye detection
			Get bounding box centered around point between both eyes
			Haar-like features??
		Double check ShiftedGrids layer in model
		

'''


'''
	Input Image Size: 227x227x3x1
	Input Face Size: 227x227x3x1
	Input Face Pos Size: 1x169x1x1 -- Because head location is 13x13 area around eye center

'''

#Hyperparameters
num_epochs = 100
batch_size = 1024
learning_rate = 0.001

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
], 1)

class GazeNet(nn.Module):
	def __init__(self):
		super(GazeNet,self).__init__()
		'''
			Saliency Pathway
		'''
		self.sal_conv = nn.Sequential(
			nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
			Fire(512, 64, 256, 256),
			nn.Conv2d(512, 1, kernel_size=1),
			nn.ReLU()
		)
		'''
			Gaze Pathway
		'''
		#Gaze Conv layers
		self.gaze_conv = nn.Sequential( #input w&h 224x224
			nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
			Fire(512, 64, 256, 256),
			nn.Conv2d(512, 1000, kernel_size=1),
			nn.ReLU(),
			nn.AvgPool2d(13, stride=1),
			)
		#Face FC Layer
		self.fc_face = nn.Sequential(
			nn.Linear(1000,500),
			nn.ReLU())
		#Combined Face and EyePos FC Layers
		self.gaze_fc = nn.Sequential(
			nn.Linear(500+169,400),
			nn.ReLU(),
			nn.Linear(400,200),
			nn.ReLU(),
			nn.Linear(200,169),
			nn.Sigmoid())
		#Conv with reshaped gaze mask
		self.importance_map = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)

		#Shifted grids layer
		self.fc_0_0 = nn.Linear(13*13*1,25)
		self.fc_1_0 = nn.Linear(13*13*1,25)
		self.fc_m1_0 = nn.Linear(13*13*1,25)
		self.fc_0_m1 = nn.Linear(13*13*1,25)	
		self.fc_0_1 = nn.Linear(13*13*1,25)

		#Softmax function
		self.softmax = nn.Softmax(dim=1)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
				init.kaiming_uniform(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
		#load_weights(self)

	def forward(self, x_i,x_h,x_p):
		#Saliency pathway
		saliency = self.sal_conv(x_i)
		#print("\n\n\t\tSaliency Conv: "+str(saliency))
		#Gaze pathway
		#print(x_h)
		gaze = self.gaze_conv(x_h)
		#print("\n\n\t\tGAZE Conv: "+str(gaze))
		gaze = gaze.view(gaze.size(0),-1) #Reshape
		#print("\n\n\t\tGAZE Type: "+str(gaze.size()))
		gaze = self.fc_face(gaze)
		
		#Head position input
		x_p = x_p.view(x_p.size(0),-1)
		x_p = torch.mul(x_p,24) #scale by 24 **According to caffe prototxt file
		x_p = x_p.type(torch.cuda.FloatTensor)
		gaze = gaze.type(torch.cuda.FloatTensor)
		gaze = torch.cat((gaze,x_p),1) #Concat both arrays
		gaze = self.gaze_fc(gaze)
		gaze = gaze.view(1,1,13,13) #Gaze mask is 13x13 so reshape
		gaze = self.importance_map(gaze)

		#Do element-wise product b/w Saliency Map and Gaze Mask
		out = torch.mul(saliency,gaze)
		#Get shifted grids **Not sure if i did this right, someone doublecheck!!
		out = out.view(out.size(0),-1)
		fc1 = self.fc_0_0(out)
		fc2 = self.fc_1_0(out)
		fc3 = self.fc_m1_0(out)
		fc4 = self.fc_0_m1(out)
		fc5 = self.fc_0_1(out)
		'''print(fc1)
		print(fc2)
		print(fc3)
		print(fc4)
		print(fc5)'''
		
		fc1 = self.softmax(fc1)
		fc2 = self.softmax(fc2)
		fc3 = self.softmax(fc3)
		fc4 = self.softmax(fc4)
		fc5 = self.softmax(fc5)
		return fc1,fc2,fc3,fc4,fc5

	

def train_gaze():
    
    gazenet = GazeNet()
    #gazenet = load_weights(gazenet)
    gazenet.cuda() #Comment out if no CUDA
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gazenet.parameters(), lr=learning_rate)
    
    #cropface = dp.CropFaceAndResize(227)
    
    gaze_train_dataset = dp.GazeDataset(csv_file='/HD1/Data/train_annotations.txt',
                                               root_dir='/HD1/Data/',
                                               transform=transforms.Compose([
                                                   dp.Rescale(280),
                                                   dp.RandomCrop(227)
                                               ]))
    
    gaze_test_dataset = dp.GazeDataset(csv_file='/HD1/Data/test_annotations.txt',
                                               root_dir='/HD1/Data/',
                                               transform=transforms.Compose([
                                                   dp.Rescale(280),
                                                   dp.RandomCrop(227)
                                               ]))
    
#    train_loader = torch.utils.data.DataLoader(gaze_train_dataset, batch_size=batch_size,
#                        shuffle=True, num_workers=4)
#    test_loader = torch.utils.data.DataLoader(gaze_test_dataset, batch_size=batch_size,
#                        shuffle=True, num_workers=4)
    
    
    for epoch in range(num_epochs):
        # trainning
        ave_loss = 0
        for i, (sample_batched, inputs) in enumerate(gaze_train_dataset):
			print(i)
			#            print(inputs['image'].size())
			#            print(sample_batched['annotations'].size())
			#            print(inputs['image'].size())
			input1 = Variable(inputs['image']).cuda()
			input2 = Variable(inputs['head']).cuda()
			input3 = Variable(inputs['pos']).cuda()
			labels = Variable(sample_batched['label']).cuda()
			#            print(input2.size())
			#            print(input3.size())
			#            print(labels.size())
			optimizer.zero_grad()
			out = gazenet(input1, input2, input3)

			#            print(labels.size())
			
			#print(out)
			if isinstance(out, tuple):
				loss = sum((criterion(o,l) for o,l in zip(out,labels)))
			else:
				loss = criterion(out, labels)
			loss.backward()
			optimizer.step()            
			ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
			if (i+1) % 1000 == 0 or (i+1) == len(gaze_train_dataset):
				print ('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
					epoch, i+1, ave_loss))
			
			
        # testing
        correct_cnt, ave_loss = 0, 0
        total_cnt = 0
        for i, (sample_batched, inputs) in enumerate(gaze_test_dataset):
            
            input1 = Variable(inputs['image'], volatile=True).cuda()
            input2 = Variable(inputs['head'], volatile=True).cuda()
            input3 = Variable(inputs['pos'], volatile=True).cuda()
            labels = Variable(sample_batched['label'], volatile=True).cuda()
            
            out = gazenet(input1, input2, input3)
	    
            if isinstance(out, tuple):
                loss = sum((criterion(o,l) for o,l in zip(out,labels)))
            else:
                loss = criterion(out, labels)
	    out_mat = torch.cat((out[0],out[1],out[2],out[3],out[4]),0)
	    
	    #out_avg = torch.sum(out_mat,0)
	    #out_avg = torch.div(out_avg,5)
	    print(out_mat)
            _, pred_label = torch.max(out_mat, 1)
	    print(pred_label)
	    print(labels.data)
            total_cnt += input1.data.size()[0]
	    correct_cnt += (pred_label.data==labels.data).cpu().sum()
            # smooth average
            ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1 
            if(i+1) % 1000 == 0 or (i+1) == len(gaze_test_dataset):
                print( '==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                    epoch, i+1, ave_loss, correct_cnt * 1.0 / total_cnt))

    torch.save(gazenet.state_dict(), gazenet.name())
    return gazenet


def find_gaze(img,head,pos,model):

	#Load the weights
	#gazenet = load_weights(model)
	#gazenet.cuda()
	#Normalize images
	'''img /= 255
	head /= 255'''
	#print(img)
	

	#transfrom head position (aka eye location coordinate) into vector
	head_pos = np.zeros((1,1,169))
	z = np.zeros((13,13))
    #x = np.floor((pos[0]*13)/np.size(img,0))+1
	#y = np.floor((pos[1]*13)/np.size(img,0))+1
	x = int(np.floor((pos[0]*13)))
	y = int(np.floor((pos[1]*13)))
	z[x,y] = 1
	z = np.reshape(z, (1,1,169))
	head_pos=z
	head_pos = np.resize(head_pos,(1,169,1))#resize to column vector
	h_pos = torch.from_numpy(head_pos)

	#Resize images
	#print("\n\nimg size: " + str(np.shape(img)))
	#print("\n\n")
	img = np.reshape(img,(227,227,3))
	head = np.reshape(head,(227,227,3))
	#load model
	#gazenet.eval()
	
	#From CV2 image
	t_img = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)
	t_head = torch.from_numpy(head.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
	#print(head.transpose(2,0,1)/255.0)
	#cv2.imshow('Test head', head)
	#create image variable
	img_var = Variable(t_img).cuda()
	head_var = Variable(t_head).cuda()
	h_pos = Variable(h_pos).cuda()
	'''img_var = image2torch(img)
	head_var = image2torch(head)'''
	#evaluate image, output is Pytorch Variable
	gaze_outputs = model(img_var, head_var, h_pos)
	#print("\n\n\t\tGAZE OUTPUTS: "+str(gaze_outputs))
	#print("\n\n")

	outputs = gaze_outputs

	alpha = 0.3 #Might need later
	fc_0_0 = np.transpose(outputs[0].data.cpu().numpy())
	fc_1_0 = np.transpose(outputs[1].data.cpu().numpy())
	fc_m1_0 = np.transpose(outputs[2].data.cpu().numpy())
	fc_0_1 = np.transpose(outputs[3].data.cpu().numpy())
	fc_0_m1 = np.transpose(outputs[4].data.cpu().numpy())

	#construct heatmap for images
	heatmap = np.zeros((15,15))
	count_hm = np.zeros((15,15))
	
	#Reshape to squares
	f_0_0 = np.reshape(fc_0_0,(5,5)) 
	f_1_0 = np.reshape(fc_1_0,(5,5)) 
	f_m1_0 = np.reshape(fc_m1_0,(5,5)) 
	f_0_1 = np.reshape(fc_0_1,(5,5)) 
	f_0_m1 = np.reshape(fc_0_m1,(5,5)) 

	f_list = [f_0_0, f_1_0, f_m1_0, f_0_1, f_0_m1]
	v_x = [0, 1, -1, 0, 0]
	v_y = [0, 0, 0, -1, 1]

	#Create heatmap by shifting the grids
	for k in range(0,5):
		delta_x = v_x[k]
		delta_y = v_y[k]
		f = f_list[k]
		for x in range(0,5):
			for y in range(0,5):
				i_x = 3*(x) - delta_x
				i_x = max(i_x,0)
				if(x==0):
					i_x = 0

				i_y = 3*(y) - delta_y
				i_y = max(i_y,0)
				if(y==0):
					i_y = 0                
				f_x = 3*(x+1)-delta_x
				f_x = min(14,f_x)
				if(x==4):
					f_x = 14
				f_y = 3*(y+1)-delta_y
				f_y = min(14,f_y)
				if(y==4):
					f_y = 14
				heatmap[i_x:(f_x+1),i_y:(f_y+1)] += f[x,y]
				count_hm[i_x:(f_x+1),i_y:(f_y+1)] += 1 
	
	#Resize heatmap to match size of input image
	#heatmap = heatmap.astype(np.uint8)
	#heatmap = np.reshape(heatmap,(1,heatmap.shape[0],heatmap.shape[1]))

	hm_base = np.divide(heatmap,count_hm) #UNCOMMENT ONCE LRN IMPLEMENTED
	hm_results = cv2.resize(np.transpose(hm_base),(227,227),interpolation=cv2.INTER_LINEAR)
	hm_idx = np.argmax(hm_results)
	hm_r_c = np.unravel_index(hm_idx,(hm_results.shape[0],hm_results.shape[1]))
	y_predict = hm_r_c[0]/hm_results.shape[0]
	x_predict = hm_r_c[1]/hm_results.shape[1]
	return x_predict, y_predict, hm_results
	'''Fixed this for heatmap implementation	
	hm_results = np.reshape(hm_results,(hm_results.shape[0],hm_results.shape[1],1))
	cv2.imshow('heatmap',hm_results)
	heat_img = cv2.cvtColor(hm_results,cv2.COLOR_GRAY2RGB)
	heat_img = cv2.applyColorMap(heat_img,cv2.COLORMAP_JET)
	new_img = np.reshape(img,(img.shape[0],img.shape[1],3))
	#Blend images together
	combined_image = cv2.addWeighted(new_img,0.6,heat_img,0.4,0)
	cv2.imshow('Heatmap',combined_image)
	return combined_image'''

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x

def image2torch(img):
    img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    return img
	
        
#sys.stdin.read(1)
def load_weights(gazenet):
	
	#Shahbaz: Manually Loading Weights
	'''	
	print("Loading values")
	w = np.load("Pre-trained model/conv1_0.npy")
	b = np.load("Pre-trained model/conv1_1.npy")
	print(gazenet.sal_conv[0].weight.data.shape == w.shape)
	print(gazenet.sal_conv[0].bias.data.shape == b.shape)
	gazenet.sal_conv[0].weight.data = torch.from_numpy(w)
	gazenet.sal_conv[0].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/conv2_0.npy")
	b = np.load("Pre-trained model/conv2_1.npy")
	print(gazenet.sal_conv[4].weight.data.shape == w.shape)
	print(gazenet.sal_conv[4].bias.data.shape == b.shape)
	gazenet.sal_conv[4].weight.data = torch.from_numpy(w)
	gazenet.sal_conv[4].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/conv3_0.npy")
	b = np.load("Pre-trained model/conv3_1.npy")
	print(gazenet.sal_conv[8].weight.data.shape == w.shape)
	print(gazenet.sal_conv[8].bias.data.shape == b.shape)
	gazenet.sal_conv[8].weight.data = torch.from_numpy(w)
	gazenet.sal_conv[8].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/conv4_0.npy")
	b = np.load("Pre-trained model/conv4_1.npy")
	print(gazenet.sal_conv[10].weight.data.shape == w.shape)
	print(gazenet.sal_conv[10].bias.data.shape == b.shape)
	gazenet.sal_conv[10].weight.data = torch.from_numpy(w)
	gazenet.sal_conv[10].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/conv5_0.npy")
	b = np.load("Pre-trained model/conv5_1.npy")
	print(gazenet.sal_conv[12].weight.data.shape == w.shape)
	print(gazenet.sal_conv[12].bias.data.shape == b.shape)
	gazenet.sal_conv[12].weight.data = torch.from_numpy(w)
	gazenet.sal_conv[12].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/conv5_red_0.npy")
	b = np.load("Pre-trained model/conv5_red_1.npy")
	print(gazenet.sal_conv[14].weight.data.shape == w.shape)
	print(gazenet.sal_conv[14].bias.data.shape == b.shape)
	gazenet.sal_conv[14].weight.data = torch.from_numpy(w)
	gazenet.sal_conv[14].bias.data = torch.from_numpy(b)'''
	'''
	print("Loading Gaze Subnetwork")
	w = np.load("Pre-trained model/conv1_face_0.npy")
	b = np.load("Pre-trained model/conv1_face_1.npy")
	print(gazenet.gaze_conv[0].weight.data.shape == w.shape)
	print(gazenet.gaze_conv[0].bias.data.shape == b.shape)
	gazenet.gaze_conv[0].weight.data = torch.from_numpy(w)
	gazenet.gaze_conv[0].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/conv2_face_0.npy")
	b = np.load("Pre-trained model/conv2_face_1.npy")
	print(gazenet.gaze_conv[4].weight.data.shape == w.shape)
	print(gazenet.gaze_conv[4].bias.data.shape == b.shape)
	gazenet.gaze_conv[4].weight.data = torch.from_numpy(w)
	gazenet.gaze_conv[4].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/conv3_face_0.npy")
	b = np.load("Pre-trained model/conv3_face_1.npy")
	print(gazenet.gaze_conv[8].weight.data.shape == w.shape)
	print(gazenet.gaze_conv[8].bias.data.shape == b.shape)
	gazenet.gaze_conv[8].weight.data = torch.from_numpy(w)
	gazenet.gaze_conv[8].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/conv4_face_0.npy")
	b = np.load("Pre-trained model/conv4_face_1.npy")
	print(gazenet.gaze_conv[10].weight.data.shape == w.shape)
	print(gazenet.gaze_conv[10].bias.data.shape == b.shape)
	gazenet.gaze_conv[10].weight.data = torch.from_numpy(w)
	gazenet.gaze_conv[10].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/conv5_face_0.npy")
	b = np.load("Pre-trained model/conv5_face_1.npy")
	print("CONV5_FACE: "+str(gazenet.gaze_conv[12].weight.data))
	print(gazenet.gaze_conv[12].bias.data.shape == b.shape)
	gazenet.gaze_conv[12].weight.data = torch.from_numpy(w)
	gazenet.gaze_conv[12].bias.data = torch.from_numpy(b)
	print("\n\t\tConv5 SIZE: "+ str(np.shape(w)))'''




	print("Loading Gaze FC")
	w = np.load("Pre-trained model/fc6_face_0.npy")
	print("\n\t\tFC6 SIZE: "+ str(np.shape(w)))
	b = np.load("Pre-trained model/fc6_face_1.npy")
	print(gazenet.fc_face[0].weight.data.shape)
	print(gazenet.fc_face[0].bias.data.shape == b.shape)
	gazenet.fc_face[0].weight.data = torch.from_numpy(w)#.transpose(1,0)
	gazenet.fc_face[0].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/fc7_face_0.npy")
	print("\n\t\FC7 SIZE: "+ str(np.shape(w)))
	b = np.load("Pre-trained model/fc7_face_1.npy")
	print(gazenet.gaze_fc[0].weight.data.shape == w.shape)
	print(gazenet.gaze_fc[0].bias.data.shape == b.shape)
	gazenet.gaze_fc[0].weight.data = torch.from_numpy(w)
	gazenet.gaze_fc[0].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/fc8_face_0.npy")
	print("\n\t\tFC8 SIZE: "+ str(np.shape(w)))
	b = np.load("Pre-trained model/fc8_face_1.npy")
	print(gazenet.gaze_fc[2].weight.data.shape == w.shape)
	print(gazenet.gaze_fc[2].bias.data.shape == b.shape)
	gazenet.gaze_fc[2].weight.data = torch.from_numpy(w)
	gazenet.gaze_fc[2].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/importance_no_sigmoid_0.npy")
	b = np.load("Pre-trained model/importance_no_sigmoid_1.npy")
	print(gazenet.gaze_fc[4].weight.data.shape == w.shape)
	print(gazenet.gaze_fc[4].bias.data.shape == b.shape)
	gazenet.gaze_fc[4].weight.data = torch.from_numpy(w)
	gazenet.gaze_fc[4].bias.data = torch.from_numpy(b)

	print("Loading Shifted grids layer")
	w = np.load("Pre-trained model/fc_0_0_0.npy")
	b = np.load("Pre-trained model/fc_0_0_1.npy")
	print(gazenet.fc_0_0.weight.data.shape == w.shape)
	print(gazenet.fc_0_0.bias.data.shape == b.shape)
	gazenet.fc_0_0.weight.data = torch.from_numpy(w)
	gazenet.fc_0_0.bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/fc_0_1_0.npy")
	b = np.load("Pre-trained model/fc_0_1_1.npy")
	print(gazenet.fc_0_1.weight.data.shape == w.shape)
	print(gazenet.fc_0_1.bias.data.shape == b.shape)
	gazenet.fc_0_1.weight.data = torch.from_numpy(w)
	gazenet.fc_0_1.bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/fc_1_0_0.npy")
	b = np.load("Pre-trained model/fc_1_0_1.npy")
	print(gazenet.fc_1_0.weight.data.shape == w.shape)
	print(gazenet.fc_1_0.bias.data.shape == b.shape)
	gazenet.fc_1_0.weight.data = torch.from_numpy(w)
	gazenet.fc_1_0.bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/fc_m1_0_0.npy")
	b = np.load("Pre-trained model/fc_m1_0_1.npy")
	print(gazenet.fc_m1_0.weight.data.shape == w.shape)
	print(gazenet.fc_m1_0.bias.data.shape == b.shape)
	gazenet.fc_m1_0.weight.data = torch.from_numpy(w)
	gazenet.fc_m1_0.bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/fc_0_m1_0.npy")
	b = np.load("Pre-trained model/fc_0_m1_1.npy")
	print(gazenet.fc_0_m1.weight.data.shape == w.shape)
	print(gazenet.fc_0_m1.bias.data.shape == b.shape)
	gazenet.fc_0_m1.weight.data = torch.from_numpy(w)
	gazenet.fc_0_m1.bias.data = torch.from_numpy(b)

	print("Loading Importance MAP")
	w = np.load("Pre-trained model/importance_map_0.npy")
	b = np.load("Pre-trained model/importance_map_1.npy")
	print(gazenet.importance_map.weight.data.shape == w.shape)
	print(gazenet.importance_map.bias.data.shape == b.shape)
	gazenet.importance_map.weight.data = torch.from_numpy(w)
	gazenet.importance_map.bias.data = torch.from_numpy(b)
	
	return gazenet
	

if __name__== "__main__":
    trained_model = train_gaze()


