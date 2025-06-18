import numpy as np
import pandas as pd
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import customeException
from src.logger import logging
from src.utilis import save_object
from dataclasses import dataclass


@dataclass

class Data_Transformation_Config:
    preprocessor_obj_file=os.path.join("artifacts","proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.Datta_gate=Data_Transformation_Config()

    def get_data_transformer_object(self):
        try:
            numerical_columns=["writing score","reading score"]
            categorical_columns=["gender","race/ethnicity","parental level of education",
                         "lunch","test preparation course"]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),# we change [handle_unknown='ignore']
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"categorcal columns :{ categorical_columns}")
            logging.info(f"Numerical columns:{numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline, numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)

                ]
            )

            return preprocessor





        except Exception as e:
            raise customeException(e,sys)

    def initiote_data_transformation(self,train_psth,test_path):
        try:
            train_df=pd.read_csv(train_psth)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data")
            logging.info("obtaing preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            target_colum="math score"
            

            input_feature_train_df=train_df.drop(columns=[target_colum],axis=1)
            #print("Input columns:", input_feature_train_df.columns.tolist())

            target_feature_train_df=train_df[target_colum]

            input_feature_test_df=test_df.drop(columns=[target_colum],axis=1)
            target_feature_test_df=test_df[target_colum]

            logging.info(f"Applying preprocessing")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)

            ]

            test_arr=np.c_[
                input_feature_test_df,np.array(target_feature_test_df)

            ]

            save_object(
                file_path=self.Datta_gate.preprocessor_obj_file,
                obj=preprocessing_obj
            )
            return(
                train_arr,test_arr,self.Datta_gate.preprocessor_obj_file,
            )
            

            
        except Exception as e:
            raise customeException(e,sys)        


