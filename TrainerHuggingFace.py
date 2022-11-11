from datasets import load_dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification,AutoFeatureExtractor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from PIL import Image
import requests
import numpy as np
import evaluate
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms, utils
import os
from skimage import io, transform
from torch import nn

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # print('shape labels',labels.view(-1,self.model.config.num_labels).shape)
        # print('shape labels',labels)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        _, pred = torch.max(logits, 1)
        # print('shape logits',logits.view(-1, self.model.config.num_labels).shape)
        # print('shape preds',pred.shape)
        # print('shape preds',pred)
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.BCEWithLogitsLoss()
        print(pred.shape)
        print(labels.shape)
        loss = loss_fct(pred.float(),labels.float())
        return (loss, outputs) if return_outputs else loss




class DeepSewerMLClassification(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.csvFile = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csvFile)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.csvFile.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.csvFile.iloc[idx,1]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample



class DataSetHF:
    def __init__(self, datasetName,is_data_stored_locally):
        self.datasetName = datasetName
        self.dataTrain, self.dataVal, self.labels, self.label2id, self.id2label = self.__loadDatasetImageFolder(is_data_stored_locally)
        self.n_SamplesTrain = 0
    
    
    def plotDataBalance(self):
        if self.n_SamplesTrain == 0:
            self.n_SamplesTrain = self.__countSamples()
        print('labels ', self.labels)
        print('count ', self.n_SamplesTrain)
        plt.bar(self.labels, self.n_SamplesTrain)
        plt.xlabel("Labels")
        plt.ylabel("No. of samples")
        plt.title("Data Balance Analysis")
        plt.show()

    def loadDatasetCSV(self, is_data_stored_locally,labelTrain='train',labelVal='validation'):
        #load dataset from a csv files
        pass
    def showRandomSamples(self):
        #display random pictures of the dataset
        pass

    def __loadDatasetImageFolder(self,is_data_stored_locally,labelTrain='train',labelVal='validation'):
        if is_data_stored_locally: 
            dataTrain = load_dataset("imagefolder", data_dir=self.datasetName, split='train')
            dataVal = load_dataset("imagefolder", data_dir=self.datasetName, split='validation')
        else:
            access_token = "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"
            dataTrain = load_dataset(self.datasetName,split=labelTrain,use_auth_token=access_token)
            dataVal = load_dataset(self.datasetName,split=labelVal,use_auth_token=access_token)
        

        labels = dataTrain.features["label"].names
        
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label
            
        return dataTrain, dataVal, labels, label2id, id2label
        
    def __countSamples(self):
        n_samples = np.zeros((len(self.labels)), dtype=int)
        for i in tqdm(range(len(self.dataTrain))):
            id_label = self.dataTrain[i]['label']
            n_samples[id_label] = n_samples[id_label] + 1
        return n_samples

class TrainerHF(DataSetHF):
    def __init__(self,modelName, datasetName, is_data_local):
        super().__init__(datasetName,is_data_local)
        self.model_checkpoint = modelName   
        self.dataset_name = datasetName
        self.metric = evaluate.load("accuracy")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(modelName)
        print('model parameters correctly loaded')
        self.__applyAugmentation()
        print('dataset correctly loaded')
        
    
    def trainModel(self,name,epochs,lr,batch_size):
        model,args = self.__loadModel(outName=name, n_epochs=epochs,lr=lr,batch_size=batch_size)
        trainer = Trainer(
        model,
        args,
        train_dataset=self.dataTrain,
        eval_dataset=self.dataVal,
        tokenizer=self.feature_extractor,
        compute_metrics=self.__compute_metrics,
        data_collator=self.__collate_fn,
        )
        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()
        #Evaluate
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    
    def __loadModel(self,outName='new_model', n_epochs= 4, lr = 5e-5, batch_size= 32):
        #model
        model = AutoModelForImageClassification.from_pretrained(
            self.model_checkpoint, 
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        args = TrainingArguments(
            outName,
            remove_unused_columns=False,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=n_epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
        )
        return model, args

    def __applyAugmentation(self):
        normalize = Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)
        train_transforms = Compose(
                [
                    RandomResizedCrop(self.feature_extractor.size),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    normalize,
                ]
            )

        val_transforms = Compose(
                [
                    Resize(self.feature_extractor.size),
                    CenterCrop(self.feature_extractor.size),
                    ToTensor(),
                    normalize,
                ]
            )
        def preprocess_train(example_batch):
            """Apply train_transforms across a batch."""
            example_batch["pixel_values"] = [
                train_transforms(image.convert("RGB")) for image in example_batch["image"]
            ]
            return example_batch

        def preprocess_val(example_batch):
            """Apply val_transforms across a batch."""
            example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
            return example_batch

        self.dataTrain.set_transform(preprocess_train)
        self.dataVal.set_transform(preprocess_val)
        

    def __compute_metrics(self, eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=eval_pred.label_ids)

    def __collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
        


