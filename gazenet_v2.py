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
import functional

#Uncomment if using build < 0.4
#from LocalResponseNorm import LocalResponseNorm


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
	Input Image Size: 224x224x3x1
	Input Face Size: 224x224x3x1
	^Crop to 224x224x3x1 when pre-processing
	Input Face Pos Size: 1x169x1x1 -- Because head location is 13x13 area around eye center

'''

#Hyperparameters
num_epochs = 100
batch_size = 256
learning_rate = 0.001



class GazeNet(nn.Module):
	def __init__(self):
		super(GazeNet,self).__init__()
		'''
			Saliency Pathway
		'''
		self.sal_conv = nn.Sequential(
			#layer1
			nn.Conv2d(3,96,kernel_size=11, stride=4,padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.LocalResponseNorm(5),
			#layer2
			nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2,groups=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3,stride=2),
			nn.LocalResponseNorm(5),
			#layer3
			nn.Conv2d(256,384,kernel_size=3, stride=1,padding=1),
			nn.ReLU(),
			#layer4
			nn.Conv2d(384,384,kernel_size=3, stride=1,padding=1,groups=2),
			nn.ReLU(),
			#layer5
			nn.Conv2d(384,256,kernel_size=3, stride=1,padding=1,groups=2),
			nn.ReLU(),
			#layer5_red
			nn.Conv2d(256,1,kernel_size=1, stride=1), #output is 13x13x1
			nn.ReLU())
		'''
			Gaze Pathway
		'''
		#Gaze Conv layers
		self.gaze_conv = nn.Sequential( #input w&h 224x224
			#layer1
			nn.Conv2d(3,96,kernel_size=11, stride=4,padding=1), #down to 55x55
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2), #pool to 27x27
			nn.LocalResponseNorm(5),
			#layer2
			nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2,groups=2), #stay @ 27x27
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3,stride=2), #pool to 13x13
			nn.LocalResponseNorm(5),
			#layer3
			nn.Conv2d(256,384,kernel_size=3, stride=1,padding=1),#Stay @ 13x13
			nn.ReLU(),
			#layer4
			nn.Conv2d(384,384,kernel_size=3, stride=1,padding=1,groups=2), #Stay @ 13x13
			nn.ReLU(),
			#layer5
			nn.Conv2d(384,256,kernel_size=3, stride=1,padding=1,groups=2),#Stay @ 13x13
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3,stride=2)) #pool to 6x6
		#Face FC Layer
		self.fc_face = nn.Sequential(
			nn.Linear(6*6*256,500),
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
		self.softmax = nn.Softmax()
		
		load_weights(self)

	def forward(self, x_i,x_h,x_p):
		#Saliency pathway
		saliency = self.sal_conv(x_i)

		#Gaze pathway
		gaze = self.gaze_conv(x_h)
		gaze = gaze.view(gaze.size(0),-1) #Reshape
		gaze = self.fc_face(gaze)
		#Head position input
		x_p = x_p.view(x_p.size(0),-1)
		x_p = torch.mul(x_p,24) #scale by 24 **According to caffe prototxt file
		gaze = torch.cat((gaze,x_p),0) #Concat both arrays
		gaze = self.gaze_fc(gaze)
		gaze = gaze.view(13,13) #Gaze mask is 13x13 so reshape
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
		
		fc1 = self.softmax(fc1)
		fc2 = self.softmax(fc2)
		fc3 = self.softmax(fc3)
		fc4 = self.softmax(fc4)
		fc5 = self.softmax(fc5)
		#Average outputs
		'''
		out = torch.add(fc1,fc2)
		out = torch.add(out,fc3)
		out = torch.add(out,fc4)
		out = torch.add(out,fc5)
		out = torch.div(out,5)
		'''

	

def train_gaze(train_loader, weightfile):
	#Do data loading here


		
	gazenet = GazeNet()
	gazenet.cuda() #Comment out if no CUDA

	#Load weights for model
	gazenet = load_weights(gazenet)

	#Loss and Optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(gazenet.parameters(), lr=learning_rate)

	for epoch in range(num_epochs):
		'''
			Next 3 lines not implemented yet!!!
		'''
		for i, (images,labels) in enumerate(train_loader):
			images = Variable(images).cuda()
			labels = Variable(labels).cuda()

			#Forward and Backward Pass
			optimizer.zero_grad() #Zero gradients
			outputs = gazenet(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			#**Need to check with Validation Set
	gazenet.save_state_dict(weightfile)
	#Test model
	gazenet.eval()
	correct = 0
	total = 0

	for images, labels in test_loader:
		images = Variable(images).cuda()
		outputs = gazenet(images)
		#Show heatmaps and print accuracy

	return gazenet


def find_gaze(img,head,pos,model):

	#Load the weights
	gazenet = load_weights(model)

	#Normalize images
	img = img/np.size(img,0)

	#transfrom head position (aka eye location coordinate) into vector
	head_pos = np.zeros((1,1,169))
	z = np.zeros((13,13))
	x = np.floor((pos[0]*13)/np.size(img,0))+1
	y = np.floor((pos[1]*13)/np.size(img,0))+1
	z[x,y] = 1
	head_pos[1,1,:]=z[:]
	head_pos = np.resize(head_pos,(1,169,1))#resize to column vector
	h_pos = torch.from_numpy(head_pos)

	#load model
	gazenet.eval()

	#From CV2 image
	img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
	head = torch.from_numpy(head.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
	#create image variable
	img_var = Variable(img, requires_grad=False).cuda()
	head_var = Variable(head, requires_grad=False).cuda()

	#evaluate image, output is Pytorch Variable
	gaze_outputs = gazenet([img_var, head, h_pos])

	outputs = gaze_outputs.data.cpu().numpy()

	alpha = 0.3 #Might need later
	fc_0_0 = np.transpose(outputs[0])
	fc_1_0 = np.transpose(outputs[1])
	fc_m1_0 = np.transpose(outputs[2])
	fc_0_1 = np.transpose(outputs[3])
	fc_0_m1 = np.transpose(outputs[4])

	#construct heatmap for images
	heatmap = np.zeros((15,15))
	count_hm = np.zeros((15,15))
	
	#Reshape to squares
	f_0_0 = np.reshape(fc_0_0[0,:],(5,5)) 
	f_1_0 = np.reshape(fc_1_0[0,:],(5,5)) 
	f_m1_0 = np.reshape(fc_m1_0[0,:],(5,5)) 
	f_0_1 = np.reshape(fc_0_1[0,:],(5,5)) 
	f_0_m1 = np.reshape(fc_0_m1[0,:],(5,5)) 

	f_list = [f_0_0, f_1_0, f_m1_0, f_0_1, f_0_m1]
	v_x = [0, 1, -1, 0, 0]
	v_y = [0, 0, 0, -1, 1]

	#Create heatmap by shifting the grids
	for k in range(0,4):
		delta_x = v_x[k]
		delta_y = v_y[k]
		for x in range(1,5):
			for y in range(1,5):
				i_x = 1 + 3*(x-1) - delta_x
				i_x = max(i_x,0)
				if(x==0):
					i_x = 0
				
				i_y = 1 + 3*(y-1) - delta_y
				i_y = max(i_y,1)
				if(y==1):
					i_y = 1

				f_x = 3*x-delta_x
				f_x = min(14,f_x)
				if(x==4):
					f_x = 14
				f_y = 3*y-delta_y
				f_y = min(14,f_y)
				if(y==4):
					f_y = 14
				hm[i_x:f_x,i_y:f_y] += f[x,y]
				count_hm[i_x:f_x,i_y:f_y] += 1
	
	#Resize heatmap to match size of input image
	hm_base = np.divide(hm,count_hm)
	hm_results = cv2.resize(np.transpose(hm_base),(img.width,img.height),interpolation=INTER_LINEAR)

	#Blend images together
	combined_image = cv2.addWeighted(img,0.6,hm_results,0.4)

	return combined_image
	
        
#sys.stdin.read(1)
def load_weights(gazenet):
	
	#Shahbaz: Manually Loading Weights
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
	gazenet.sal_conv[14].bias.data = torch.from_numpy(b)

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
	print(gazenet.gaze_conv[12].weight.data.shape == w.shape)
	print(gazenet.gaze_conv[12].bias.data.shape == b.shape)
	gazenet.gaze_conv[12].weight.data = torch.from_numpy(w)
	gazenet.gaze_conv[12].bias.data = torch.from_numpy(b)




	print("Loading Gaze FC")
	w = np.load("Pre-trained model/fc6_face_0.npy")
	b = np.load("Pre-trained model/fc6_face_1.npy")
	print(gazenet.fc_face[0].weight.data.shape == w.shape)
	print(gazenet.fc_face[0].bias.data.shape == b.shape)
	gazenet.fc_face[0].weight.data = torch.from_numpy(w)
	gazenet.fc_face[0].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/fc7_face_0.npy")
	b = np.load("Pre-trained model/fc7_face_1.npy")
	print(gazenet.gaze_fc[0].weight.data.shape == w.shape)
	print(gazenet.gaze_fc[0].bias.data.shape == b.shape)
	gazenet.gaze_fc[0].weight.data = torch.from_numpy(w)
	gazenet.gaze_fc[0].bias.data = torch.from_numpy(b)
	w = np.load("Pre-trained model/fc8_face_0.npy")
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
	


'''RUN NETWORK'''
sys.stdin.read(1)


