import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from faker import Faker
fake = Faker()

# #scale: GPA, gender, so ngay ctxh, hoc ki hien tai, gia dinh kho khan?, ...
# scale_f = [10, 2, 10, 2, 50, 5, 2, 3, 2] 
# #shift:
# shift_f = [0 for i in xrange(10)]
# shift_f[0] = 100

num_of_sample = 2500

X, y = make_classification(n_samples=num_of_sample, n_features=10, 
                           n_redundant=0, n_informative = 7,n_clusters_per_class=1,
                           n_classes= 100)

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

print ('Generate random data...')
name_company = np.array([fake.name() for i in xrange(100)])

GPA = gaussian_distribute(num_of_sample, 7.0, 1.0, 0, 10, 1)
gender = np.random.choice(['male', 'female'], size=(num_of_sample), p= [0.6, 0.4])
semester = rand_distribute(num_of_sample, 8, 0)
# is_ok = gaussian_distribute(num_of_sample, 0.2, 0.21, 0, 1, 0)
is_ok = np.random.choice(['yes', 'no'], size=(num_of_sample), p= [0.9, 0.1])
ctxh = gaussian_distribute(num_of_sample, 15, 8, 5, 70, 0)
mutilchoie_5 = np.random.choice(['very_good', 'good', 'fine', 'not good', 'bad'], size=(num_of_sample), p= [0.2, 0.35, 0.15, 0.2, 0.1])
mutilchoie_3 = gaussian_distribute(num_of_sample, 1, 1, 0, 2, 0)
mutilchoie_2_1 = np.random.choice(['yes', 'no'], size=(num_of_sample), p= [0.3, 0.7])
mutilchoie_2_2 = np.random.choice(['yes', 'no'], size=(num_of_sample), p= [0.5, 0.5])
mutilchoie_2_3 = np.random.choice(['yes', 'no'], size=(num_of_sample), p= [0.65, 0.35])

# extra_info = X[:,0:2]

#=================================
gender, le_g, enc_g = encode_label(gender)
is_ok, le_i, enc_i = encode_label(is_ok)
mutilchoie_5, le_5, enc_5 = encode_label(mutilchoie_5)
mutilchoie_3, le_3, enc_3 = encode_label(mutilchoie_3)
mutilchoie_2_1, le_21, enc_21 = encode_label(mutilchoie_2_1)
mutilchoie_2_2, le_22, enc_22 = encode_label(mutilchoie_2_2)
mutilchoie_2_3, le_23, enc_23 = encode_label(mutilchoie_2_3)

# y_data = encode_label(y)

GPA = GPA.reshape(-1,1)
semester = semester.reshape(-1,1)
ctxh = ctxh.reshape(-1,1)
#==================================
X_data = np.hstack((GPA,gender))
X_data = np.hstack((X_data,semester))
X_data = np.hstack((X_data,gender))
X_data = np.hstack((X_data,is_ok))
X_data = np.hstack((X_data,mutilchoie_5))
X_data = np.hstack((X_data,mutilchoie_3))
X_data = np.hstack((X_data,mutilchoie_2_1))
X_data = np.hstack((X_data,mutilchoie_2_2))
X_data = np.hstack((X_data,mutilchoie_2_3))

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

print ('Start: training')

clf = svm.SVC(probability=True)

X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.3, random_state=42)

clf.fit(X_train, y_train)

print ('Done: training.')

pred_test = clf.predict_proba(X_test)

loss_log = log_loss(y_test, pred_test)

print ('Log_loss: ' + str(loss_log))

import pickle
# save the classifier
with open('./model/model_svm.pkl', 'wb') as fid:
    pickle.dump(clf, fid)

with open('./pre_data/label_company.pkl', 'wb') as fid:
    pickle.dump(name_company, fid)
    
with open('./pre_data/le_g.pkl', 'wb') as fid:
    pickle.dump(le_g, fid)
    
with open('./pre_data/enc_g.pkl', 'wb') as fid:
    pickle.dump(enc_g, fid)
    
with open('./pre_data/le_i.pkl', 'wb') as fid:
    pickle.dump(le_i, fid)
    
with open('./pre_data/enc_i.pkl', 'wb') as fid:
    pickle.dump(enc_i, fid)
    
with open('./pre_data/le_5.pkl', 'wb') as fid:
    pickle.dump(le_5, fid)
    
with open('./pre_data/enc_5.pkl', 'wb') as fid:
    pickle.dump(enc_5, fid)
    
with open('./pre_data/le_3.pkl', 'wb') as fid:
    pickle.dump(le_3, fid)
    
with open('./pre_data/enc_3.pkl', 'wb') as fid:
    pickle.dump(enc_3, fid)
    
with open('./pre_data/le_21.pkl', 'wb') as fid:
    pickle.dump(le_21, fid)
    
with open('./pre_data/enc_21.pkl', 'wb') as fid:
    pickle.dump(enc_21, fid)
    
with open('./pre_data/le_22.pkl', 'wb') as fid:
    pickle.dump(le_22, fid)
    
with open('./pre_data/enc_22.pkl', 'wb') as fid:
    pickle.dump(enc_22, fid)
    
with open('./pre_data/le_23.pkl', 'wb') as fid:
    pickle.dump(le_23, fid)
    
with open('./pre_data/enc_23.pkl', 'wb') as fid:
    pickle.dump(enc_23, fid)

print ('Finish.')
