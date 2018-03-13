import csv
import os
import numpy as np
from skimage import io,transform
import pandas as pd
#import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

def show_gaze(image, annotations, label):
    """Show image with annotations"""
    #fig = plt.imshow(image)
    x=annotations[0, 0]*image.shape[1]
    y=annotations[0, 1]*image.shape[0]
    w=annotations[1, 0]*image.shape[1]
    h=annotations[1, 1]*image.shape[0]
    ''''eyex = annotations[2, 0]*image.shape[1]
    eyey = annotations[2, 1]*image.shape[0]
    targetx = annotations[3, 0]*image.shape[1]
    targety = annotations[3, 1]*image.shape[0]
    #fig,ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
    eye = patches.Circle((eyex,eyey),5)
    gaze = [(eyex,eyey), (targetx,targety)]
    (targetx, targety) = zip(*gaze)
    ax.add_patch(rect)
    ax.add_patch(eye)
    ax.add_line(lines.Line2D(targetx, targety, linewidth=2, color='yellow'))
    ax.axis('off')
    #plt.show()
    #plt.pause(0.001)  # pause a bit so that plots are updated'''


class GazeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        while True:
            img_name = os.path.join(self.root_dir,
                                    self.annotations.iloc[idx, 0])
            image = io.imread(img_name)
            
            #convert grayscale to RGB
            if image.ndim == 2:
                image = np.dstack([image] * 3)
                
            annotations = self.annotations.iloc[idx, 2:10].as_matrix()
            annotations = annotations.astype('float').reshape(-1, 2)
            label = AnnotationToLabel(annotations[3])
            #print(label)
            sample = {'image': image, 'annotations': annotations, 'label':label }
            if self.transform:
                sample = self.transform(sample)
                if(sample['label'][0] >= 0):
                    break
                idx = np.random.randint(1, len(self.annotations)-1)
            else:
                break
            #print("Resampling... " + str(idx))
            
        cropface = CropFaceAndResize(227)
        face = cropface(sample)
        img_var, head_var, h_pos = MakeInputReady(sample['image'],face['image'],sample['annotations'][2])
        #img_var = img_var.transpose((2, 0, 1))
        #head_var = head_var.transpose((2, 0, 1))
        inputs = {'image': img_var, 'head': head_var, 'pos':h_pos }
        return sample, inputs



class Flip(object):
    def __call__(self, sample):
        image, annotations, label = sample['image'], sample['annotations'], sample['label']
        temp = annotations *[1,1]
        img = np.flip(image,1)
        temp[0,0] =  (1 - temp[0,0]) - temp[1,0]
        temp[2,0] = 1 - temp[2,0]
        temp[3,0] = 1 - temp[3,0]
        label = AnnotationToLabel(temp[3])
        return {'image': img, 'annotations': temp, 'label':label }



class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, annotations, label = sample['image'], sample['annotations'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'annotations': annotations, 'label':label }
    
    
class RandomCrop(object):
    """Crop randomly the image in a sample."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, annotations, label = sample['image'], sample['annotations'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        i=0
        while i<5:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
    
            image2 = image[top: top + new_h,
                          left: left + new_w]
            
            annotations2 = (annotations*[w,h] - [left, top])/[new_w,new_h]
            annotations2[1] = annotations[1]*[w,h] /[new_w,new_h]
            if(np.min(annotations2)>=0 and np.max(annotations2[2])<=1):
                break
            i += 1
            #print("Resclaing Again...")
        if i==5:
            label[0]=-1
            return {'image': image2, 'annotations': annotations2, 'label':label }
        #print(annotations2,np.min(annotations2))
        label = AnnotationToLabel(annotations2[3])
        return {'image': image2, 'annotations': annotations2, 'label':label }

class CropFaceAndResize(object):
    """Crop the face in a sample."""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
    def __call__(self, sample):
        """Shahbaz: I didn't reset the annotations for this function,
        because there is no need for that"""
        
        image, annotations, label = sample['image'], sample['annotations'], sample['label']

        h, w = image.shape[:2]
        new_h = int(annotations[1][1]*h)
        new_w = int(annotations[1][0]*w)
        
        top = int(annotations[0][1]*h)
        left = int(annotations[0][0]*w)

        image = image[top: top + new_h,
                      left: left + new_w]
        new_h, new_w = self.output_size
        img = transform.resize(image, (new_h, new_w))
        
        #return image
        return {'image': img, 'annotations': annotations, 'label':label }
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, annotations, label = sample['image'], sample['annotations'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'annotations': torch.from_numpy(annotations),
                'label':label }, True
        
        

def AnnotationToLabel(sample):
    xx = sample[1]*15
    yy = sample[0]*15
    #print(xx,yy)
    v_x = [0, 1, -1, 0, 0]
    v_y = [0, 0, 0, -1, 1]
    output = np.zeros(5)
    for k in range(0,5):
        delta_x = v_x[k]
        delta_y = v_y[k]
        f = np.zeros((5,5))
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
                mid_x = (f_x + i_x)/2
                mid_y = (f_y + i_y)/2
                #print(k,x,y,f[x,y],i_x,f_x,i_y,f_y)
                f[x,y]=((xx-mid_x)*(xx-mid_x)+(yy-mid_y)*(yy-mid_y))
        #print(f)
        f = np.reshape(f,(1,25))
        output[k]=np.argmin(f)
    output = torch.from_numpy(output)
    output = output.type(torch.LongTensor)
    return output



def MakeInputReady(img,head,pos):
    #This is basically the first part of the find_gaze function
    head_pos = np.zeros((1,1,169))
    z = np.zeros((13,13))
    x = int(np.floor((pos[0]*13)))
    y = int(np.floor((pos[1]*13)))
    z[x,y] = 1
    z = np.reshape(z, (1,1,169))
    head_pos=z
    head_pos = np.resize(head_pos,(1,169,1))
    h_pos = torch.from_numpy(head_pos)
    img = np.reshape(img,(227,227,3))
    head = np.reshape(head,(227,227,3))
    
    #If there is no batch size:
    t_img = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)
    t_head = torch.from_numpy(head.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    
    #For batch training version
    #t_img = torch.from_numpy(img.transpose(2,0,1)).float()
    #t_head = torch.from_numpy(head.transpose(2,0,1)).float().div(255.0)
    
    img_var = t_img
    head_var = t_head
    h_pos = h_pos
    
    return img_var,head_var,h_pos
