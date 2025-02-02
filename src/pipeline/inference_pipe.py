import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import pandas as pd

class Predictpipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            logging.info("Prediction Started")
            model = load_object(os.path.join('artifacts','model.pkl'))
            preprocessing = load_object(os.path.join('artifacts','preprocessing.pkl'))
            features = preprocessing.transform(features)
            prediction = model.predict(features)
            logging.info("Prediction Completed")
            return prediction
        except Exception as e:
            raise CustomException(e, sys)
class CustomData:
    def __init__(self, gender:str, race_ethnicity:int, parental_level_of_education:str, lunch:str, test_preparation_course:str, writing_score:int, reading_score:int):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.writing_score=writing_score
        self.reading_score=reading_score

    def get_data_frame(self):
        try:
            custom_data_dict = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "writing_score":[self.writing_score],
                "reading_score":[self.reading_score]
            }
            return pd.DataFrame(custom_data_dict)
        except Exception as e:
            raise CustomException(e, sys)