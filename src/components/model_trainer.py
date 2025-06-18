import os
import sys 
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import customeException
from src.logger import logging
from src.utilis import save_object,evoluate_models

@dataclass
class modelTrainerconfig:
    trained_moddel_fit_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = modelTrainerconfig()
    def Initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "XGBRegessor":XGBRegressor(),
                "Catboosting Regressior":CatBoostRegressor(verbose=False),
                "Adaboost Regressor":AdaBoostRegressor(),

            }
            model_report:dict=evoluate_models(x_train= x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            #To get best model score
            best_model_score=max(sorted(model_report.values()))
            #to
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise customeException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            save_object(
                file_path=self.model_trainer_config.trained_moddel_fit_path,
                obj=best_model

            )

            predicted=best_model.predict(x_test)
            r2_sco=r2_score(y_test,predicted)
            return r2_sco,best_model_name,best_model
        

        except Exception as e:
            raise customeException(e,sys)
            