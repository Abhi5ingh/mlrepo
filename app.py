import pickle 
from flask import Flask, request, jsonify, render_template
import json
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.inference_pipe import Predictpipeline, CustomData
from src.exception import CustomException
import sys

app = Flask(__name__)

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        elif request.method == 'POST':
            data = CustomData(gender=request.form.get('gender'),
                              race_ethnicity=request.form.get('race_ethnicity'),
                              parental_level_of_education=request.form.get('parental_level_of_education'),
                                lunch=request.form.get('lunch'),
                                test_preparation_course=request.form.get('test_preparation_course'),
                                writing_score=request.form.get('writing_score'),
                                reading_score=request.form.get('reading_score'))
            pred_df = data.get_data_frame()
            pipeline = Predictpipeline()
            prediction = pipeline.predict(pred_df)
            return render_template('home.html', prediction_text='Predicted Math Score is {}'.format(prediction[0]))

        else:
            return "Invalid request"
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)