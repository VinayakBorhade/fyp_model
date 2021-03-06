from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from variables_csic import *
import re

lr = load('lr_model_v2.joblib')
vectorizer = load('vectorizer_v2.joblib')

def Find(string):
    # findall() has been used  
    # with valid conditions for urls in string 
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)
    return [x[0] for x in url]


def select_columns(columns, df):
    new_columns = []
    for c in columns:
        if c.lower() in df.columns:
            new_columns.append(c)
    
    # print("new_columns ", new_columns)

    df2 = df.loc[:, new_columns]
    return df2


def _readLogs2(file_path = "tomcat_attack.txt"):
    header_lower = [i.lower() for i in header]
    header_csic_lower = [i.lower() for i in header_csic]

    #list of dictionaries
    ld = []
    f = open(file_path, 'r')
    cnt = 0
    for l in f:
        row = l.split("   ")

        # print("row ", row)

        dict_row = {}
        dict_row['idx'] = cnt
        cnt += 1
        for i, r in enumerate(row):
            col = header_lower[i]

            # print("col ", col)

            if r == "" or r == '-':
                r = ""
            if col == 'url':
                r = r.split(" ")[1]
                # hard - coding the protocol and host name for csic model
                if len(Find(r)) == 0:
                    r = "http://" + "" + r
            dict_row[col] = r
        url_row = dict_row['url']
        # hard - coding the protocol and host name for csic model
        if len(Find(r)) == 0:
            url_row = "http://" + dict_row['host'] + url_row
        dict_row['url'] = url_row
        ld.append(dict_row)
    
    for i in ld:
        print(i)

    df = pd.DataFrame(ld)

    df = select_columns(header_csic_lower, df)
    X = df.copy()
    return X


def _readLogs(file_path="sql_injection_2.txt"):
    print("reading logs file")
    f = open(file_path, "r")
    cols=['IP address', 'logical username', 'Remote user', 'Date and time', 'url', 'postdata',
        'status code', 'bytes sent', 'accept', 'acceptEncoding', 'acceptCharset', 'acceptLanguage', 'pragma', 
        'cacheControl', 'connection', 'contentLength', 'contentType', 'host', 'userAgent', 'cookie', 'query', 'path']

    cols_res=["method","url","protocol","userAgent","pragma","cacheControl","accept","acceptEncoding","acceptCharset",
        "acceptLanguage","host","connection","contentLength","contentType","cookie","payload","label"]

    # list of dictionaries
    ld = []

    for l in f:
        # print('\nline ', l)
        d={}
        cur=""
        cnt = 0
        prev = 0
        c_i=0
        for c in l:
            if c == '[':
                # print("[ c ", c)
                prev = 1
                cnt += 1
            elif c == ']':
                cnt -= 1
            if cnt > 0 and c != '[' and c != ']':
                cur = cur + c
            if cnt == 0:
                if prev == 0:
                    continue
                prev = 0
                if cur == "-":
                    cur = "null"
                d[cols[c_i]]=cur
                cur = ""
                c_i += 1
        # print("\nd ", d)
        d2={}
        for k in cols_res:
            t = None
            if k == 'cookie':
                d2[k] = 'JSESSIONID=' + d[k]

            if k in cols and k != 'url' and k != 'cookie':
                d2[k] = d[k]
                continue
            
            if k == 'method' and len(d['url'].split(' '))==3:
                t = d['url'].split(" ")[0]
                d2[k] = t[1:]
            if k == 'url' and len(d['url'].split(' '))==3:
                t = d['url'].split(" ")[1]
                if t[0] != "/":
                    t = "/" + t
                t = 'http://' + d['host'] + t
                
                d2[k] = t
            if k == 'protocol' and len(d['url'].split(' '))==3:
                t = d['url'].split(" ")[2]
                t = t[:-1]
                d2[k] = t
            if k == 'payload':
                t = d['query']
                if t != '' and t[0] == '?':
                    t = t[1:]
                d2[k] = t
            if k == 'label':
                d2[k] = 0
        # print("\nd2 ", d2)
        ld.append(d2)

    # print("ld ", ld)
    df = pd.DataFrame(ld)
    print(df.info())
    print(df.head())
    X=df.drop(['label'], axis=1)
    y=df['label']
    return X

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


# returns np.array
def _predict(X):
    X_str=to_str(X)
    X_v = vectorizer.transform(X_str)

    y_predict = np.array(lr.predict_proba(X_v)[:, 1])
    # optimal_threshold:  0.5118798402838789
    # optimal_threshold = 0.5118798402838789
    # y_predict_val = [1 if i >= optimal_threshold else 0 for i in y_predict]
    print("prediction: ")
    for i, v in enumerate(y_predict):
        print(i, ": ", v)
    return y_predict

if __name__ == "__main__":
    print("testing on logs")
    # X = _readLogs(file_path="xss.txt")
    X = _readLogs2()
    print("now predicting")
    y = _predict(X)