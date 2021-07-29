
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from nyoka import xgboost_to_pmml

import joblib
import os
import pandas as pd
import json
import logging
import boto3


BUCKET_NAME = "bucket-teradata"
access_key = 'AKIARKGO7XF453ZI4Y2B'
secret_access_key = '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'
CLIENT = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)

def post_file(name, file_key):
    upload_file_name = name
    upload_file_key = str(file_key) + '/' + str(upload_file_name)
    CLIENT.upload_file(upload_file_name, BUCKET_NAME, upload_file_key)
    print('Done')

def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

#     create_context(host=data_conf["host"], username=os.environ['TD_USERNAME'], password=os.environ['TD_PASSWORD'])

    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    target_name = 'Outcome'

    # read training dataset from Teradata and convert to pandas
#     train_df = DataFrame(data_conf["table"])
#     train_df = train_df.select([feature_names + [target_name]])
#     train_df = train_df.to_pandas()
    train_df = data_conf
    
    # split data into X and y
    X_train = train_df.drop(target_name, 1)
    y_train = train_df[target_name]

    print("Starting training...")

    # fit model to training data
    model = Pipeline([('scaler', MinMaxScaler()),
                     ('xgb', XGBClassifier(eta=hyperparams["eta"],
                                           max_depth=hyperparams["max_depth"]))])
    # xgboost saves feature names but lets store on pipeline for easy access later
    model.feature_names = feature_names
    model.target_name = target_name

    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    joblib.dump(model, "model.joblib")
    xgboost_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name, pmml_f_name="model.pmml")
    post_file("model.joblib", "diabetes")
    post_file("model.pmml", "diabetes")
    print("Saved trained model")

    
data_conf = pd.read_csv("diabetes.csv", header=0)
f = open('config.json',)
model_conf = json.load(f)

train(data_conf, model_conf)
