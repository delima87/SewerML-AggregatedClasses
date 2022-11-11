from transformers import AutoModelForImageClassification,AutoFeatureExtractor
from PIL import Image
import requests
import pandas as pd
from tqdm import tqdm
#Load Testing Dataset from class and try to print some examples
from transformers import pipeline,AutoFeatureExtractor
import numpy as np
import os
from PIL import Image

class InfereHF:
    def __init__(self,modelName, dataset,input_labels):
        #Load model
        featureExtractor = AutoFeatureExtractor.from_pretrained(modelName)
        self.pipe = pipeline("image-classification", 
                        model= modelName,
                        feature_extractor=featureExtractor)
        self.dataset = dataset
        self.l = input_labels

    def predicSingleImage(self, id):
        file_name = self.dataset[id]['imageIDs']    
        raw_pred = self.predictImg(self.dataset[id]['image'],file_name)
        gt_l = self.dataset[id]['labels'].cpu().detach().numpy().astype(int)
        ground_truth = {'filename':file_name, self.l[0]:gt_l[0],self.l[1]:gt_l[1],self.l[2]:gt_l[2],self.l[3]:gt_l[3],self.l[4]:gt_l[4],
                    self.l[5]:gt_l[5], self.l[6]:gt_l[6],self.l[7]:gt_l[7],self.l[8]:gt_l[8],self.l[9]:gt_l[9]}
        return ground_truth, raw_pred
    
    def predictImg(self, image, file_name):
        pred = self.pipe(image)
        raw_pred = {'filename':file_name, self.l[0]:0,self.l[1]:0,self.l[2]:0,self.l[3]:0,self.l[4]:0,
                    self.l[5]:0, self.l[6]:0,self.l[7]:0,self.l[8]:0,self.l[9]:0}
        for p in pred:
            raw_pred[p['label']] = p['score']
        return raw_pred

    def predictTestDataset(self):
        #crete a csv with all predictions
        new_csv = pd.DataFrame(columns = ['filename'] + self.l)
        gt_csv = pd.DataFrame(columns = ['filename'] + self.l)
        for i in tqdm(range(len(self.dataset))):
            gt_d, raw_d = self.predicSingleImage(i)
            df2= pd.DataFrame(data=raw_d,index=[0])
            new_csv = pd.concat([new_csv,df2])
            df3= pd.DataFrame(data=gt_d,index=[0])
            gt_csv = pd.concat([gt_csv,df3])
        return gt_csv,new_csv

    def predictSetImgs(self):
        new_csv = pd.DataFrame(columns = ['filename'] + self.l)
        for root, dirs, files in os.walk(self.dataset):
            for file in tqdm(files):
                img_path = os.path.join(self.dataset,file)     
                image = Image.open(img_path).convert('RGB')
                raw_pred = self.predictImg(image,file)
                df2= pd.DataFrame(data=raw_pred,index=[0])
                new_csv = pd.concat([new_csv,df2])
        return new_csv

   
     