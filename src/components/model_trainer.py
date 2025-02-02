import sys
import os
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.utils import save_object,evaluate_model
import pickle

@dataclass

class ModelTrainerConfig:
    train_model_path: str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def init_model_training(self,train_array,test_array,preprocessor_path):
        logging.info("Model Training Initiated")
        try:
            logging.info("Model Training Started, splitting data")
            x_train,y_train,x_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "LinearRegression": LinearRegression()
            }
            model_report: dict = evaluate_model(X_train=x_train, y_train=y_train,x_test=x_test,y_test=y_test, models=models)
            best_score_model = max(model_report, key=lambda x: model_report[x]['r2_score'])
            best_model = models[best_score_model]
            best_score = model_report[best_score_model]['r2_score']
            print(f"Best Model: {best_score_model}, Best Score: {best_score}")
            if best_score<0.6:
                logging.error("Model Score is less than 0.6")
                raise CustomException("Model Score is less than 0.6")
            logging.info("Model Training Completed")
            preprocessing_ob = pickle.load(open(preprocessor_path, 'rb'))
            save_object(obj=best_model, file_path=self.model_trainer_config.train_model_path)
            pred=best_model.predict(x_test)
            r2_score_val=r2_score(y_test,pred)
            return r2_score_val
        except Exception as e:
            raise CustomException(e, sys)
        