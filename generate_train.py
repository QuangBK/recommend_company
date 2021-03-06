import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from faker import Faker
import pandas as pd
import pickle
import argparse
fake = Faker()

# #scale: GPA, gender, so ngay ctxh, hoc ki hien tai, gia dinh kho khan?, ...
# scale_f = [10, 2, 10, 2, 50, 5, 2, 3, 2] 
# #shift:
# shift_f = [0 for i in xrange(10)]
# shift_f[0] = 100

fields = ['GPA', 'score_web', 'score_math_1', 'score_Mac_LeNin', 'score_mobile',
        'score_db', 'gender', 'age', 'semester', 'ctxh', 'is_ok', 'mutilchoie_5',
        'mutilchoie_3', 'mutilchoie_2_1', 'mutilchoie_2_2', 'mutilchoie_2_3',
        'noi_muon_lam']

list_city = ['HCM', 'Ha_Noi', 'Da_Nang', 'HUE', 'Nha_Trang', 'Khac']
age_range = range(18,27)
num_of_sample = 2500

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

def encode_label_y(x, le=None, enc=None):
    if le is None:
        le = preprocessing.LabelEncoder()
        X_1 = le.fit_transform(x)
    else:
        X_1 = le.transform(x)
        
    X_ = X_1.reshape(-1,1)
    
    if enc is None:
        enc = OneHotEncoder()
        X_ = enc.fit_transform(X_)
    else:
        X_ = enc.transform(X_)   
    
    return X_.toarray(), X_1, le, enc

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

#=====================================================================================
def main(list_arg):
    print list_arg.input_file, list_arg.save
    if list_arg.input_file == None:
        print ('Generate random data...')
        name_company_fake = np.array([fake.name() for i in xrange(100)])
        _, y = make_classification(n_samples=num_of_sample, n_features=10, 
                                   n_redundant=0, n_informative = 7,n_clusters_per_class=1,
                                   n_classes= 100)

        company_data = np.array([name_company_fake[i] for i in y])

        GPA = gaussian_distribute(num_of_sample, 7.0, 1.0, 0, 10, 1)
        score_web = gaussian_distribute(num_of_sample, 8.5, 1.2, 0, 10, 1)
        score_math_1 = gaussian_distribute(num_of_sample, 7.5, 0.8, 0, 10, 1)
        score_Mac_LeNin = gaussian_distribute(num_of_sample, 5.5, 1.2, 0, 10, 1)
        score_mobile = gaussian_distribute(num_of_sample, 7.0, 1.0, 0, 10, 1)
        score_db = gaussian_distribute(num_of_sample, 6.0, 1.2, 0, 10, 1)

        gender = np.random.choice(['male', 'female'], size=(num_of_sample), p= [0.6, 0.4])
        age = np.random.choice(age_range, size=(num_of_sample))
        semester = rand_distribute(num_of_sample, 8, 0)
        # is_ok = gaussian_distribute(num_of_sample, 0.2, 0.21, 0, 1, 0)
        is_ok = np.random.choice(['yes', 'no'], size=(num_of_sample), p= [0.9, 0.1])
        ctxh = gaussian_distribute(num_of_sample, 15, 8, 5, 70, 0)
        mutilchoie_5 = np.random.choice(['very_good', 'good', 'fine', 'not good', 'bad'], size=(num_of_sample), p= [0.2, 0.35, 0.15, 0.2, 0.1])
        mutilchoie_3 = gaussian_distribute(num_of_sample, 1, 1, 0, 2, 0)
        mutilchoie_2_1 = np.random.choice(['yes', 'no'], size=(num_of_sample), p= [0.3, 0.7])
        mutilchoie_2_2 = np.random.choice(['yes', 'no'], size=(num_of_sample), p= [0.5, 0.5])
        mutilchoie_2_3 = np.random.choice(['yes', 'no'], size=(num_of_sample), p= [0.65, 0.35])
        noi_muon_lam = np.random.choice(list_city, size=(num_of_sample), p= [0.2, 0.1, 0.2, 0.1, 0.3, 0.1])
        # extra_info = X[:,0:2]
        #=================================
        if list_arg.save:
            print ('Save data...')
            dict_data = {
                'GPA': GPA, 
                'score_web': score_web, 
                'score_math_1': score_math_1,
                'score_Mac_LeNin': score_Mac_LeNin, 
                'score_mobile': score_mobile,
                'score_db': score_db, 
                'gender': gender, 
                'age': age, 
                'semester': semester, 
                'ctxh': ctxh, 
                'is_ok': is_ok, 
                'mutilchoie_5': mutilchoie_5,
                'mutilchoie_3': mutilchoie_3, 
                'mutilchoie_2_1': mutilchoie_2_1, 
                'mutilchoie_2_2': mutilchoie_2_2, 
                'mutilchoie_2_3': mutilchoie_2_3,
                'noi_muon_lam': noi_muon_lam,
                'company': company_data,
            }
            df = pd.DataFrame(data=dict_data)
            df.to_csv('./dataset/data.csv', index=False)
    else:
        df_data = pd.read_csv(list_arg.input_file)
        GPA = df_data.as_matrix(['GPA'])
        score_web = df_data.as_matrix(['score_web'])
        score_math_1 = df_data.as_matrix(['score_math_1'])
        score_Mac_LeNin = df_data.as_matrix(['score_Mac_LeNin'])
        score_mobile = df_data.as_matrix(['score_mobile'])
        score_db = df_data.as_matrix(['score_db'])

        gender = df_data.as_matrix(['gender'])
        age = df_data.as_matrix(['age'])
        semester = df_data.as_matrix(['semester'])
        is_ok = df_data.as_matrix(['is_ok'])
        ctxh = df_data.as_matrix(['ctxh'])
        mutilchoie_5 = df_data.as_matrix(['mutilchoie_5'])
        mutilchoie_3 = df_data.as_matrix(['mutilchoie_3'])
        mutilchoie_2_1 = df_data.as_matrix(['mutilchoie_2_1'])
        mutilchoie_2_2 = df_data.as_matrix(['mutilchoie_2_2'])
        mutilchoie_2_3 = df_data.as_matrix(['mutilchoie_2_3'])
        noi_muon_lam = df_data.as_matrix(['noi_muon_lam'])

        company_data = df_data.as_matrix(['company'])

        number_of_rows = GPA.shape[0]
        GPA = GPA.reshape(number_of_rows, )
        score_web = score_web.reshape(number_of_rows, )
        score_math_1 = score_math_1.reshape(number_of_rows, )
        score_Mac_LeNin = score_Mac_LeNin.reshape(number_of_rows, )
        score_mobile = score_mobile.reshape(number_of_rows, )
        score_db = score_db.reshape(number_of_rows, )

        gender = gender.reshape(number_of_rows, )
        age = age.reshape(number_of_rows, )
        semester = semester.reshape(number_of_rows, )
        is_ok = is_ok.reshape(number_of_rows, )
        ctxh = ctxh.reshape(number_of_rows, )
        mutilchoie_5 = mutilchoie_5.reshape(number_of_rows, )
        mutilchoie_3 = mutilchoie_3.reshape(number_of_rows, )
        mutilchoie_2_1 = mutilchoie_2_1.reshape(number_of_rows, )
        mutilchoie_2_2 = mutilchoie_2_2.reshape(number_of_rows, )
        mutilchoie_2_3 = mutilchoie_2_3.reshape(number_of_rows, )
        noi_muon_lam = noi_muon_lam.reshape(number_of_rows, )      

        company_data = company_data.reshape(number_of_rows, )        
    #=================================
    gender, le_g, enc_g = encode_label(gender)
    is_ok, le_i, enc_i = encode_label(is_ok)
    mutilchoie_5, le_5, enc_5 = encode_label(mutilchoie_5)
    mutilchoie_3, le_3, enc_3 = encode_label(mutilchoie_3)
    mutilchoie_2_1, le_21, enc_21 = encode_label(mutilchoie_2_1)
    mutilchoie_2_2, le_22, enc_22 = encode_label(mutilchoie_2_2)
    mutilchoie_2_3, le_23, enc_23 = encode_label(mutilchoie_2_3)
    noi_muon_lam, le_noi_muon_lam, enc_noi_muon_lam = encode_label(noi_muon_lam)
    _, company_data, le_company_data, enc_company_data = encode_label_y(company_data)

    y = company_data
    name_company = le_company_data.classes_
    # y_data = encode_label(y)

    GPA = GPA.reshape(-1,1)
    score_web = score_web.reshape(-1,1)
    score_math_1 = score_math_1.reshape(-1,1)
    score_Mac_LeNin = score_Mac_LeNin.reshape(-1,1)
    score_mobile = score_mobile.reshape(-1,1)
    score_db = score_db.reshape(-1,1)
    age = age.reshape(-1,1)
    semester = semester.reshape(-1,1)
    ctxh = ctxh.reshape(-1,1)
    #==================================
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


    print ('Start: training')

    clf = svm.SVC(probability=True)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.3, random_state=42)

    clf.fit(X_train, y_train)

    print ('Done: training.')

    pred_test = clf.predict_proba(X_test)

    loss_log = log_loss(y_test, pred_test)

    print ('Log_loss: ' + str(loss_log))

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

    with open('./pre_data/le_noi_muon_lam.pkl', 'wb') as fid:
        pickle.dump(le_noi_muon_lam, fid)
        
    with open('./pre_data/enc_noi_muon_lam.pkl', 'wb') as fid:
        pickle.dump(enc_noi_muon_lam, fid)

    with open('./pre_data/le_company_data.pkl', 'wb') as fid:
        pickle.dump(le_company_data, fid)
        
    with open('./pre_data/enc_company_data.pkl', 'wb') as fid:
        pickle.dump(enc_company_data, fid)

    print ('Finish.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example with long option names')
    parser.add_argument('--save', action="store_true", default=False,
        help='Save file or not')
    parser.add_argument('--input_file', dest='input_file', action='store',
                        default=None, help='Input file path')

    args = parser.parse_args(sys.argv[1:])
    main(args)