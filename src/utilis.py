import os
import sys
import numpy as np
import pandas as pd
from src.exception import customeException
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dripath=os.path.dirname(file_path)
        os.makedirs(dripath,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        
    except Exception as e:
        raise customeException(e,sys)


def evoluate_models(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(x_train,y_train)
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]]=test_model_score
        return report    
    except Exception as e:
        raise customeException(e,sys)



def load_object(file_path):
    try:
        with open (file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise customeException(e,sys)        

        
