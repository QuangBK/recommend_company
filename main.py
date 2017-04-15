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
le_noi_muon_lam, enc_noi_muon_lam = 0,0

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

# def convert_raw_data(data):
#     GPA = np.array([data['GPA']])
#     semester = np.array([data['semester']])
#     ctxh = np.array([data['ctxh']])
    
#     GPA = GPA.reshape(-1,1)
#     semester = semester.reshape(-1,1)
#     ctxh = ctxh.reshape(-1,1)
    
#     gender, _, _ = encode_label([data['gender']], le_g, enc_g)
#     is_ok, _, _ = encode_label([data['is_ok']], le_i, enc_i)
#     mutilchoie_5, _, _ = encode_label([data['mutilchoie_5']], le_5, enc_5)
#     mutilchoie_3, _, _ = encode_label([data['mutilchoie_3']], le_3, enc_3)
#     mutilchoie_2_1, _, _ = encode_label([data['mutilchoie_2_1']], le_21, enc_21)
#     mutilchoie_2_2, _, _ = encode_label([data['mutilchoie_2_2']], le_22, enc_22)
#     mutilchoie_2_3, _, _ = encode_label([data['mutilchoie_2_3']], le_23, enc_23)

#     X_data = np.hstack((GPA,gender))
#     X_data = np.hstack((X_data,semester))
#     X_data = np.hstack((X_data,gender))
#     X_data = np.hstack((X_data,is_ok))
#     X_data = np.hstack((X_data,mutilchoie_5))
#     X_data = np.hstack((X_data,mutilchoie_3))
#     X_data = np.hstack((X_data,mutilchoie_2_1))
#     X_data = np.hstack((X_data,mutilchoie_2_2))
#     X_data = np.hstack((X_data,mutilchoie_2_3))
#     return X_data

def convert_raw_data(data):
    GPA = np.array([data['GPA']])
    score_web = np.array([data['score_web']])
    score_math_1 = np.array([data['score_math_1']])
    score_Mac_LeNin = np.array([data['score_Mac_LeNin']])
    score_mobile = np.array([data['score_mobile']])
    score_db = np.array([data['score_db']])
    age = np.array([data['age']])
    semester = np.array([data['semester']])
    ctxh = np.array([data['ctxh']])
    
    
    GPA = GPA.reshape(-1,1)
    score_web = score_web.reshape(-1,1)
    score_math_1 = score_math_1.reshape(-1,1)
    score_Mac_LeNin = score_Mac_LeNin.reshape(-1,1)
    score_mobile = score_mobile.reshape(-1,1)
    score_db = score_db.reshape(-1,1)
    age = age.reshape(-1,1)
    semester = semester.reshape(-1,1)
    ctxh = ctxh.reshape(-1,1)

    gender, _, _ = encode_label([data['gender']], le_g, enc_g)
    is_ok, _, _ = encode_label([data['is_ok']], le_i, enc_i)
    mutilchoie_5, _, _ = encode_label([data['mutilchoie_5']], le_5, enc_5)
    mutilchoie_3, _, _ = encode_label([data['mutilchoie_3']], le_3, enc_3)
    mutilchoie_2_1, _, _ = encode_label([data['mutilchoie_2_1']], le_21, enc_21)
    mutilchoie_2_2, _, _ = encode_label([data['mutilchoie_2_2']], le_22, enc_22)
    mutilchoie_2_3, _, _ = encode_label([data['mutilchoie_2_3']], le_23, enc_23)
    noi_muon_lam, _, _ = encode_label([data['noi_muon_lam']], le_noi_muon_lam, enc_noi_muon_lam)

    X_data = np.hstack((GPA,score_web))
    X_data = np.hstack((X_data,score_math_1))
    X_data = np.hstack((X_data,score_Mac_LeNin))
    X_data = np.hstack((X_data,score_mobile))
    X_data = np.hstack((X_data,score_db))
    X_data = np.hstack((X_data,gender))
    X_data = np.hstack((X_data,age))
    X_data = np.hstack((X_data,semester))
    X_data = np.hstack((X_data,is_ok))
    X_data = np.hstack((X_data,mutilchoie_5))
    X_data = np.hstack((X_data,mutilchoie_3))
    X_data = np.hstack((X_data,mutilchoie_2_1))
    X_data = np.hstack((X_data,mutilchoie_2_2))
    X_data = np.hstack((X_data,mutilchoie_2_3))
    X_data = np.hstack((X_data,noi_muon_lam))
    return X_data

def main(list_arg):
    # save the classifier
    global le_g, enc_g, le_i, enc_i, le_5, enc_5
    global le_3, enc_3, le_21, enc_21
    global le_22, enc_22, le_23, enc_23, le_noi_muon_lam, enc_noi_muon_lam

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

    with open('./pre_data/le_noi_muon_lam.pkl', 'rb') as fid:
        le_noi_muon_lam = pickle.load(fid)
        
    with open('./pre_data/enc_noi_muon_lam.pkl', 'rb') as fid:
        enc_noi_muon_lam = pickle.load(fid)

    fields = ['GPA', 'score_web', 'score_math_1', 'score_Mac_LeNin', 'score_mobile',
            'score_db', 'gender', 'age', 'semester', 'ctxh', 'is_ok', 'mutilchoie_5',
            'mutilchoie_3', 'mutilchoie_2_1', 'mutilchoie_2_2', 'mutilchoie_2_3',
            'noi_muon_lam']
    type_fields = [2,2,2,2,2,2,1,0,0,0,1,1,0,1,1,1,1]
    data_raw = {}
    for i in xrange(len(fields)):
        if type_fields[i] == 0:
            data_raw[fields[i]] = int(list_arg[i])
        if type_fields[i] == 1:
            data_raw[fields[i]] = list_arg[i]
        if type_fields[i] == 2:
            data_raw[fields[i]] = float(list_arg[i])
    # print data_raw
    # data_raw = {'GPA': float(list_arg[0]),
    #             '' 
    #             'is_ok': list_arg[1],'gender': list_arg[2],
    #             'age': int(list_arg[3])
    #             'semester': int(list_arg[3]), 'ctxh': int(list_arg[4]),
    #             'mutilchoie_5': list_arg[5],
    #             'mutilchoie_3': int(list_arg[6]), 'mutilchoie_2_1' : list_arg[7],
    #             'mutilchoie_2_2': list_arg[8],
    #             'mutilchoie_2_3': list_arg[9]}
    data_test = convert_raw_data(data_raw)
    pred = clf.predict_proba(data_test)
    index_com = pred[0].argsort()[-5:][::-1]
    for i in index_com:
        print (name_company[i], pred[0][i])
    return name_company[index_com]

if __name__ == '__main__':
    main(sys.argv[1:])

# Run code
# params as list fields 
#     fields = ['GPA', 'score_web', 'score_math_1', 'score_Mac_LeNin', 'score_mobile',
#             'score_db', 'gender', 'age', 'semester', 'ctxh', 'is_ok', 'mutilchoie_5',
#             'mutilchoie_3', 'mutilchoie_2_1', 'mutilchoie_2_2', 'mutilchoie_2_3',
#             'noi_muon_lam']
# python main.py 8.5 5.6 7.4 8.0 9.0 7.0 male 21 5 15 yes fine 2 yes yes no HCM