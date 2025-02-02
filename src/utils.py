import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import pickle


def save_object(obj, file_path):
    try:
        logging.info("Saving Object Started")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info("Object Saved")
    except Exception as e:
        raise CustomException(e, sys)