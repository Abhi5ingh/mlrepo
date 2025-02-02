import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.metrics import mean_squared_error, r2_score

def save_object(obj, file_path):
    try:
        logging.info("Saving Object Started")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info("Object Saved")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, x_test, y_test, models):
    try:
        logging.info("Model Evaluation Started")
        model_report = {}
        for model_name, model in models.items():
            logging.info(f"Training {model_name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(x_test)
            model_report[model_name] = {
                "mse": mean_squared_error(y_test, y_pred),
                "r2_score": r2_score(y_test, y_pred)
            }
        return model_report
    except Exception as e:
        raise CustomException(e, sys)
 