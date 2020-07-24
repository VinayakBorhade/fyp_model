from joblib import dump, load
import numpy as np
import pandas as pd
import sklearn
import io
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from variables_ecml_pkdd import *

linear_svc_model = load('linear_svc_model_v3.joblib')
vectorizer = load('vectorizer_v3.joblib')

def select_columns(columns, df):
    new_columns = []
    for c in columns:
        if c.lower() in df.columns:
            new_columns.append(c)
    
    # print("new_columns ", new_columns)

    df2 = df.loc[:, new_columns]
    return df2

def to_str(X):
    count=0
    X_str=[]
    for index, row in X.iterrows():
        temp=""
        for c in X.columns:
            temp = temp + ' ' + str(row[c])
        X_str.append(temp)
#         if count<10:
# #             print('index; ', index)
#             count+=1
#             for c in X.columns:
#                 print(row[c])
    return X_str


def _readLogs(file_path = "forensics_tomcat_logs_sample.txt"):
    print("reading logs")
    header_lower = [i.lower() for i in header]
    header_ecml_pkdd_lower = [i.lower() for i in header_ecml_pkdd]
    df = pd.read_csv(file_path, names = header_lower, sep = '   ')

    # print("df.columns ", df.columns)

    df = select_columns(header_ecml_pkdd_lower, df)

    # print(df.head())

    X = df.copy()

    # print(X.head())
    X_str = to_str(X)
    X_str_v = vectorizer.transform(X_str)
    y_pred = linear_svc_model.predict(X_str_v)

    print(y_pred)

_readLogs()