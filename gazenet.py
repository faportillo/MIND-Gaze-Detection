import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

'''
	Input Image Size: 227x227x3x1
	Input Face Size: 227x227x3x1
	^Crop to 224x224x3x1 when pre-processing
	Input Face Pos Size: 1x169x1x1 -- Because head location is 13x13 area around eye center

'''

#Hyperparameters
num_epochs = 100
batch_size = 256
learning_rate = 0.001

#Do data loading here


#Build models here

#Saliency Pathway
class Saliency(nn.Module):
	def __init__(self):
		super(saliency,self).__init__()
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

	def forward(self,x):
		out = self.sal_conv(x)
		return out

#Gaze pathway
class Gaze(nn.Module):
	def __init__(self):
		super(saliency,self).__init__()
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
			nn.Linear(500+169,400)
			nn.ReLU(),
			nn.Linear(400,200),
			nn.ReLU(),
			nn.Linear(200,169),
			nn.Sigmoid())
		#Conv with reshaped gaze mask
		self.importance_map = Conv2d(1,1,kernel_size=3,stride=1,pad=1)
			

	def forward(self,x,p):
		#Head crop input
		out = self.gaze_conv(x)
		out = out.view(out.size(0),-1) #Reshape
		out = self.fc_face(out)

		#Head position input
		p = p.view(p.size(0),-1)
		p = torch.mul(p,24) #scale by 24 **According to caffe prototxt file

		out = torch.cat((out,p),0) #Concat both arrays
		out = self.gaze_fc(out)
		out = out.view(13,13) #Gaze mask is 13x13 so reshape
		out = self.importance_map(out)
		
		return out


#
saliency = Saliency()
saliency.cuda()
gaze = Gaze()
gaze.cuda()

#Combine Saliency and Gaze Pathways with Element-Wise Product




