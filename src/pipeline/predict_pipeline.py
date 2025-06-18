import sys
import os
import pandas as pd
from src.exception import customeException
from src.utilis import load_object

class predict_pipelines:
    def __init__(self):
        pass
    def predics_model(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","proprocessor.pkl")
            print("Before loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("after loading")
            data_out=preprocessor.transform(features)
            out_put=model.predict(data_out)
            return out_put
        except Exception as e:
            raise customeException(e,sys)
        



class Custom_Data:
    def __init__(self, gender:str, race_ethicity:str,
                  parental_level_of_education:str, lunch:str, 
                  test_preparation_course:int,  reading_score:int,writing_score:int):
        
        self.genders=gender
        self.race_ethicitys=race_ethicity
        self.parental_level_of_educations=parental_level_of_education
        self.lunchs=lunch
        self.test_preparation_courses=test_preparation_course
        self.reading_scores=reading_score
        self.writing_scores=writing_score

    def  get_data_as_data_frame(self):
         
         try:
            custom_data_input={
                 "gender":[self.genders],
                 "race/ethnicity":[self.race_ethicitys],
                 "parental level of education":[self.parental_level_of_educations],
                 "lunch":[ self.lunchs],
                 "test preparation course":[self.test_preparation_courses],
                 "reading score":[self.reading_scores],
                 "writing score":[self.writing_scores],

             }
            return pd.DataFrame(custom_data_input)
            
         except Exception as e:
             raise customeException (e,sys)
           
        

