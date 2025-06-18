import os
import sys
#from src.logger import logging
from src.logger import logging
from src.exception import customeException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
from src.components.data_transformation import Data_Transformation_Config
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import modelTrainerconfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionconfig:
    train_data :str=os.path.join("artifacts","train.csv")
    test_data :str=os.path.join("artifacts","test.csv")
    raw_data :str=os.path.join("artifacts","data.csv")
class Dataingestion:
    def __init__(self):
        self.ingestion_config= DataIngestionconfig()
    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")

        try:

            df=pd.read_csv('notebook/data/studn.csv')
            #df=pd.read_csv("notebook\data\studn.csv")
            logging.info("read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data,index=False,header=True)
            logging.info("train test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data,index=False,header=True)

            logging.info("Ingestion af the data is completed")

            return(
                self.ingestion_config.train_data,
                self.ingestion_config.test_data,

            )







        except Exception as e:
            raise customeException(e,sys)


if __name__=="__main__":
    obj= Dataingestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_tranfrt=DataTransformation()
    train_arr,test_arr,_=data_tranfrt.initiote_data_transformation(train_data,test_data)



    trainer = ModelTrainer()
    print(trainer.Initiate_model_trainer(train_arr, test_arr))


