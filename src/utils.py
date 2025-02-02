import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

def save_object(obj, file_path):
    try:
        logging.info("Saving Object Started")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info("Object Saved")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, x_test, y_test, models,param_grids):
    try:
        logging.info("Model Evaluation Started")
           # Perform Hyperparameter Tuning
        model_report = {}

        for model_name, model in models.items():
            logging.info(f"Tuning hyperparameters for {model_name}")

            # Perform Hyperparameter Tuning
            param_grid = param_grids.get(model_name, {})  # Get params if available
            if param_grid:
                search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
            else:
                best_model = model
                best_model.fit(X_train, y_train)
                best_params = {}

            # Evaluate on test data
            y_pred = best_model.predict(x_test)
            model_report[model_name] = {
                "mse": mean_squared_error(y_test, y_pred),
                "r2_score": r2_score(y_test, y_pred),
                "best_params": best_params
            }

        return model_report

    except Exception as e:
        raise CustomException(e, sys)
 