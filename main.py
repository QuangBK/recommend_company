import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
# from faker import Faker
import pickle
import sys


# fake = Faker()

le_g, enc_g = 0,0
le_i, enc_i = 0,0
le_5, enc_5 = 0,0
le_3, enc_3 = 0,0
le_21, enc_21 = 0,0
le_22, enc_22 = 0,0
le_23, enc_23 = 0,0

def gaussian_distribute(sample, mean, var, clip_min, clip_max, around):
    x = np.random.normal(mean, var, sample)
    x = np.clip(x, clip_min, clip_max)
    x = np.around(x, decimals=around)
    return x

def rand_distribute(sample, scale, around):
    x = np.random.rand(sample)
    x = x*scale
    x = np.around(x, decimals=around)
    return x
def encode_label(x, le=None, enc=None):
    if le is None:
        le = preprocessing.LabelEncoder()
        X_ = le.fit_transform(x)
    else:
        X_ = le.transform(x)
        
    X_ = X_.reshape(-1,1)
    
    if enc is None:
        enc = OneHotEncoder()
        X_ = enc.fit_transform(X_)
    else:
        X_ = enc.transform(X_)   
    
    return X_.toarray(), le, enc

def convert_raw_data(data):
    GPA = np.array([data['GAP']])
    semester = np.array([data['semester']])
    ctxh = np.array([data['ctxh']])
    
    GPA = GPA.reshape(-1,1)
    semester = semester.reshape(-1,1)
    ctxh = ctxh.reshape(-1,1)
    
    gender, _, _ = encode_label([data['gender']], le_g, enc_g)
    is_ok, _, _ = encode_label([data['is_ok']], le_i, enc_i)
    mutilchoie_5, _, _ = encode_label([data['mutilchoie_5']], le_5, enc_5)
    mutilchoie_3, _, _ = encode_label([data['mutilchoie_3']], le_3, enc_3)
    mutilchoie_2_1, _, _ = encode_label([data['mutilchoie_2_1']], le_21, enc_21)
    mutilchoie_2_2, _, _ = encode_label([data['mutilchoie_2_2']], le_22, enc_22)
    mutilchoie_2_3, _, _ = encode_label([data['mutilchoie_2_3']], le_23, enc_23)

    X_data = np.hstack((GPA,gender))
    X_data = np.hstack((X_data,semester))
    X_data = np.hstack((X_data,gender))
    X_data = np.hstack((X_data,is_ok))
    X_data = np.hstack((X_data,mutilchoie_5))
    X_data = np.hstack((X_data,mutilchoie_3))
    X_data = np.hstack((X_data,mutilchoie_2_1))
    X_data = np.hstack((X_data,mutilchoie_2_2))
    X_data = np.hstack((X_data,mutilchoie_2_3))
    return X_data

def main(list_arg):
    # save the classifier
    global le_g, enc_g, le_i, enc_i, le_5, enc_5
    global le_3, enc_3, le_21, enc_21
    global le_22, enc_22, le_23, enc_23

    with open('./model/model_svm.pkl', 'rb') as fid:
        clf = pickle.load(fid)

    with open('./pre_data/label_company.pkl', 'rb') as fid:
        name_company = pickle.load(fid)
        
    with open('./pre_data/le_g.pkl', 'rb') as fid:
        le_g = pickle.load(fid)
        
    with open('./pre_data/enc_g.pkl', 'rb') as fid:
        enc_g = pickle.load(fid)
        
    with open('./pre_data/le_i.pkl', 'rb') as fid:
        le_i = pickle.load(fid)
        
    with open('./pre_data/enc_i.pkl', 'rb') as fid:
        enc_i = pickle.load(fid)
        
    with open('./pre_data/le_5.pkl', 'rb') as fid:
        le_5 = pickle.load(fid)
        
    with open('./pre_data/enc_5.pkl', 'rb') as fid:
        enc_5 = pickle.load(fid)
        
    with open('./pre_data/le_3.pkl', 'rb') as fid:
        le_3 = pickle.load(fid)
        
    with open('./pre_data/enc_3.pkl', 'rb') as fid:
        enc_3 = pickle.load(fid)
        
    with open('./pre_data/le_21.pkl', 'rb') as fid:
        le_21 = pickle.load(fid)
        
    with open('./pre_data/enc_21.pkl', 'rb') as fid:
        enc_21 = pickle.load(fid)
        
    with open('./pre_data/le_22.pkl', 'rb') as fid:
        le_22 = pickle.load(fid)
        
    with open('./pre_data/enc_22.pkl', 'rb') as fid:
        enc_22 = pickle.load(fid)
        
    with open('./pre_data/le_23.pkl', 'rb') as fid:
        le_23 = pickle.load(fid)
        
    with open('./pre_data/enc_23.pkl', 'rb') as fid:
        enc_23 = pickle.load(fid)

    # data_raw = {'GAP': 8.5, 'is_ok': 'yes','gender': 'male', 'semester': 5, 'ctxh': 7, 'mutilchoie_5': 'fine',
    #             'mutilchoie_3': 2, 'mutilchoie_2_1' : 'yes', 'mutilchoie_2_2': 'yes',
    #             'mutilchoie_2_3': 'no'}
    data_raw = {'GAP': float(list_arg[0]), 'is_ok': list_arg[1],'gender': list_arg[2],
                'semester': int(list_arg[3]), 'ctxh': int(list_arg[4]),
                'mutilchoie_5': list_arg[5],
                'mutilchoie_3': int(list_arg[6]), 'mutilchoie_2_1' : list_arg[7],
                'mutilchoie_2_2': list_arg[8],
                'mutilchoie_2_3': list_arg[9]}
    data_test = convert_raw_data(data_raw)
    pred = clf.predict_proba(data_test)
    index_com = pred[0].argsort()[-5:][::-1]
    for i in index_com:
        print (name_company[i], pred[0][i])
    return name_company[index_com]

if __name__ == '__main__':
    main(sys.argv[1:])