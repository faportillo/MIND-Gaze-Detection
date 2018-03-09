import csv
import os
import numpy as np
from skimage import io,transform
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def show_gaze(image, annotations):
    """Show image with annotations"""
    #fig = plt.imshow(image)
    x=annotations[0, 0]*image.shape[1]
    y=annotations[0, 1]*image.shape[0]
    w=annotations[1, 0]*image.shape[1]
    h=annotations[1, 1]*image.shape[0]
    eyex = annotations[2, 0]*image.shape[1]
    eyey = annotations[2, 1]*image.shape[0]
    targetx = annotations[3, 0]*image.shape[1]
    targety = annotations[3, 1]*image.shape[0]
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
    eye = patches.Circle((eyex,eyey),5)
    gaze = [(eyex,eyey), (targetx,targety)]
    (targetx, targety) = zip(*gaze)
    ax.add_patch(rect)
    ax.add_patch(eye)
    ax.add_line(lines.Line2D(targetx, targety, linewidth=2, color='yellow'))
    ax.axis('off')
    plt.show()
    plt.pause(0.001)  # pause a bit so that plots are updated


class GazeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.annotations.iloc[idx, 0])
        image = io.imread(img_name)
        annotations = self.annotations.iloc[idx, 2:10].as_matrix()
        annotations = annotations.astype('float').reshape(-1, 2)
        sample = {'image': image, 'annotations': annotations}
        if self.transform:
            sample = self.transform(sample)
        return sample



class Flip(object):
    def __call__(self, sample):
        image, annotations = sample['image'], sample['annotations']
        temp = annotations *[1,1]
        img = np.flip(image,1)
        temp[0,0] =  (1 - temp[0,0]) - temp[1,0]
        temp[2,0] = 1 - temp[2,0]
        temp[3,0] = 1 - temp[3,0]
        return {'image': img, 'annotations': temp}



class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, annotations = sample['image'], sample['annotations']

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

        return {'image': img, 'annotations': annotations}
    
    
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
        image, annotations = sample['image'], sample['annotations']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        temp = annotations[1]
        annotations = (annotations*[w,h] - [left, top])/[new_w,new_h]
        annotations[1] = temp*[w,h] /[new_w,new_h]
        return {'image': image, 'annotations': annotations}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, annotations = sample['image'], sample['annotations']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'annotations': torch.from_numpy(annotations)}