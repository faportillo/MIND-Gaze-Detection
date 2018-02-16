import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

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
	Input Face Size: 224x224x3x1
	^Crop to 224x224x3x1 when pre-processing
	Input Face Pos Size: 1x169x1x1 -- Because head location is 13x13 area around eye center

'''

#Hyperparameters
num_epochs = 100
batch_size = 256
learning_rate = 0.001

#Do data loading here


#Build model here

class GazeNet(nn.Module)
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
			nn.Linear(500+169,400)
			nn.ReLU(),
			nn.Linear(400,200),
			nn.ReLU(),
			nn.Linear(200,169),
			nn.Sigmoid())
		#Conv with reshaped gaze mask
		self.importance_map = Conv2d(1,1,kernel_size=3,stride=1,pad=1)

		#Shifted grids layer
		self.fc_0_0 = nn.Linear(13*13*1,25)
		self.fc_1_0 = nn.Linear(13*13*1,25)
		self.fc_m1_0 = nn.Linear(13*13*1,25)
		self.fc_0_m1 = nn.Linear(13*13*1,25)	
		self.fc_0_1 = nn.Linear(13*13*1,25)

		#Softmax function
		self.softmax = nn.Softmax()

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
		out = torch.add(fc1,fc2)
		out = torch.add(out,fc3)
		out = torch.add(out,fc4)
		out = torch.add(out,fc5)
		out = torch.div(out,5)
		
gazenet = GazeNet()
gazenet.cuda() #Comment out if no CUDA

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
	'''
		Next 3 lines not implemented yet!!!
	'''
	for i, (images,labels) in enumerate(train_loader):
		images = Variable(images).cuda()
		labels = Variable(labels.cuda()

		#Forward and Backward Pass
		optimizer.zero_grad() #Zero gradients
		outputs = gazenet(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		#**Need to check with Validation Set

#Test model
gazenet.eval()
correct = 0
total = 0

for images, labels in test_loader:
	images = Variable(images).cuda()
	outputs = gazenet(images)
	#Show heatmaps and print accuracy





