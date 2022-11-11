import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

class EvaluationSewerModels:
    def __init__(self, groundt_csv, predictions_csv, input_labels):
        self.predictions = pd.read_csv(predictions_csv, sep=",", encoding="utf-8", usecols = ["filename"] + input_labels )
        self.reference = pd.read_csv(groundt_csv, sep=",", encoding="utf-8", usecols = ["filename"] + input_labels )
        self.labels = input_labels
    
    def evaluate_thresholds(self,metric,th):
        thresholds = np.arange(0,1,0.1)
        ytrue = self.reference.iloc[:,1:].to_numpy() 
        ypred = self.predictions.iloc[:,1:].to_numpy()
        _ypred = np.where(ypred > th,1,0)
        if metric == 'f1':
            scores= fbeta_score(ytrue, _ypred, beta=1, average=None)
        if metric == 'f2':
            scores= fbeta_score(ytrue, _ypred, beta=2, average=None)
        if metric == 'acc':
            scores= accuracy_score(ytrue, _ypred, normalize=True)
        #df_scores = pd.DataFrame(scores,columns = self.labels)
        return scores
    
