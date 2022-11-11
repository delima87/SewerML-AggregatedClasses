import os
import pandas as pd
from torchvision.io import read_image
import torch
import numpy as np
from PIL import Image
import random


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, input_labels, image_transform=None, target_transform=None):
        #read dataset 
        self.img_labels = pd.read_csv(annotations_file, sep=",", encoding="utf-8", usecols = ["filename"] + input_labels )
        self.img_dir = img_dir
        self.image_transform = image_transform
        self.labels = np.array(self.img_labels.iloc[:,1:]).astype(int)
        self.num_labels = len(input_labels) 

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #image = Image.open(img_path).convert('RGB')
        image = Image.open(img_path).convert('RGB')
          
        if self.image_transform:
            image = self.image_transform(image)
        
        labels = torch.Tensor(self.labels[idx])
        
        unk_mask_indices = random.sample(range(self.num_labels), (self.num_labels))
        mask = labels.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        sample = {}
        sample['image'] = image
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = self.img_labels.iloc[idx,0]
        
        return sample
