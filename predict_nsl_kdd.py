from joblib import dump, load
import numpy as np
import pandas as pd
import sklearn
import io
import random
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from variables_nsl_kdd import *
from collections import Counter


# categorical_columns=['protocol_type', 'service', 'flag']

# col_names = ['num_conn', 'startTimet', 'orig_pt', 'resp_pt1', 'orig_ht',
# 'resp_ht', 'duration', 'protocol_type', 'resp_pt2', 'flag', 'src_bytes', 'dst_bytes',
# 'land', 'wrong_fragment', 'urg', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
# 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
# 'num_access_files', 'num_outbound_cmds', 'is_hot_login', 'is_guest_login', 'count', 
# 'srv_count', 'serror_rate', 'srv_serror_rate_sec', 'rerror_rate', 'srv_error_rate_sec', 
# 'same_srv_rate', 'diff_srv_rate', 'dst_host_diff_srv_rate', 'count_100', 'srv_count_100', 
# 'same_srv_rate_100', 'diff_srv_rate_100', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
# 'dst_host_serror_rat', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

# selected_col_names = ["protocol_type", "service", "flag", "src_bytes", 
# "dst_bytes", "wrong_fragment", "hot", "logged_in", 
# "root_shell", "num_root", "count", "srv_count", "serror_rate", 
# "rerror_rate", "same_srv_rate", "diff_srv_rate", "dst_host_count", 
# "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
# "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
# "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]


# Loading the models and other objects
dos_model = load('nsl_kdd_dos_model.joblib')
probe_model = load('nsl_kdd_probe_model.joblib')
dict_label_encoder = load('dict_label_encoder.joblib')
one_hot_encoder = load('one_hot_encoder.joblib')
scaler_dos = load('scaler_dos.joblib')
scaler_probe = load('scaler_probe.joblib')

def select_columns(data_frame, column_names):
    new_column_names = []
    for i in column_names:
        if i in data_frame.columns:
            new_column_names.append(i)
    new_frame = data_frame.loc[:, new_column_names]
    return new_frame

def _labelEnc(c_val, c):
    # c_val is the category value for the particular row
    encoded_val = dict_label_encoder[c].transform([c_val])[0]
    # if encoded_val != 5:
    # print("c_val ", c_val, " encoded_val, ", encoded_val)
    return encoded_val

def _readLogs(file_path = "slowloris.csv"):
    df_test = pd.read_csv(file_path)
    df_test['service'] = 'http'
    df_test['flag']    = 'S0'
    df_test['label']   = 'normal'
    df_test['urgent']  = 0
    df_test['dst_host_count']  = 0
    df_test['dst_host_srv_count']  = 0
    df_test['dst_host_same_srv_rate']  = 0
    df_test['dst_host_serror_rate'] = 0
    print('Dimensions of the Test set:',df_test.shape)
    df_test = select_columns(df_test, selected_col_names)

    # # debugging starts ---
    # # print('new df columns ', df_test.columns)
    # print('Test set:')
    # for col_name in df_test.columns:
    #     if df_test[col_name].dtypes == 'object' :
    #         unique_cat = len(df_test[col_name].unique())
    #         print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))
    # # debugging ends ---

    #Data Preprocessing
    # print("dum_cols ", dum_cols)

    df_test_categorical = df_test[categorical_columns]
    # df_test_categorical = df_test_categorical.apply(le2.transform)
    for c in categorical_columns:
        # using label encoder fitted on the trained data set to label encode df_test 
        # print("c ", c)
        df_test_categorical[c] = df_test_categorical[c].apply(_labelEnc, args = (c,))

        # print("converted ", df_test_categorical[c].unique(), " le.classes ", len(dict_label_encoder[c].classes_))
    
    # print("after label encoding ")
    print(df_test_categorical.head())

    # using one hot encoder on above 
    df_test_categorical_enc = one_hot_encoder.transform(df_test_categorical)

    # print("encoded dim ", df_test_categorical_enc.toarray().shape, " enc.cat ", one_hot_encoder.categories_)
    
    # converting back to df after above encoding
    df2 = pd.DataFrame(df_test_categorical_enc.toarray(), columns = dum_cols)
    
    #join the above df into original df
    df_test = df_test.join(df2)

    # drop the old columns
    df_test.drop('flag', axis=1, inplace=True)
    df_test.drop('protocol_type', axis=1, inplace=True)
    df_test.drop('service', axis=1, inplace=True)

    # print('df_test shape ', df_test.shape)

    # dropping label for predicting
    X = df_test.drop(['label'], axis = 1)

    # print('X.shape ', X.shape)

    return X

    
def _predict(X):
    # calculating probability for dos
    X_dos = scaler_dos.transform(X)
    X_dos1 = scaler_dos.transform(X)
    X_dos = dos_model.predict_proba(X_dos)
    X_dos1 = dos_model.predict(X_dos1)
    
    print('dos_model.classes_', dos_model.classes_ , ' dos 1st 10 ', X_dos[:10])
    print(Counter(X_dos1))

    # calculating probability for probe
    X_probe = scaler_probe.transform(X)
    X_probe1 = scaler_probe.transform(X)
    X_probe = probe_model.predict_proba(X_probe)
    X_probe1 = probe_model.predict(X_probe1)
    
    print('probe_model.classes_', probe_model.classes_ , ' probe 1st 10 ', X_probe[:10])
    print(Counter(X_probe1))

    return {'dos': X_dos[:, 1], 'probe': X_probe[:, 1]}

if __name__ == "__main__":
    print("testing on logs")
    X = _readLogs(file_path="slowloris.csv")
    print("now predicting")
    y = _predict(X)

    print("y", y)
# _readLogs()
