# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:01:54 2018

@author: 87405
"""

import pandas as pd
import numpy as np
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from time import time

rawdata=pd.read_csv("D:/TEAM/460tech/finalProj/diabetic_data.csv",
                 delimiter=",", usecols=(0,1,2,3,4,7,8,9,11,18,19,20,23,47,48,49),
                 names=["encounter_id","patient_nbr","race","gender","age",
                        "discharge_disposition_id","admission_source_id","time_in_hospital",
                        "medical_specialty","diag1","diag2","diag3","A1Cresult",
                        "change","diabetesMed","readmitted"],low_memory=False)
rawdata.drop([0],inplace=True)
rawdata.drop_duplicates("patient_nbr",inplace=True)

for i in rawdata.index:
    if rawdata.discharge_disposition_id[i] in ['13','14','18','19','20','21','26']:
        rawdata.drop([i],inplace=True)
    elif rawdata.admission_source_id[i] == '26':
        rawdata.drop([i],inplace=True)
    elif rawdata.gender[i] not in ['Female','Male']:
        rawdata.drop([i],inplace=True)
    else:
        if rawdata.medical_specialty[i] == '?':
            rawdata.medical_specialty[i] = 'msNA'
        elif rawdata.medical_specialty[i] not in ['Cardiology','Family/GeneralPractice','InternalMedicine']:
            if "Surgery" in rawdata.medical_specialty[i]:
                rawdata.medical_specialty[i] = 'Surgery'
            else:
                rawdata.medical_specialty[i] = 'Other'                
        if rawdata.admission_source_id[i] == '7':
            rawdata.admission_source_id[i] = 'EmergencyRoom'
        elif rawdata.admission_source_id[i] in ['1','2']:
            rawdata.admission_source_id[i] = 'PhysicianClinicReferral'
        else:
            rawdata.admission_source_id[i] = 'Admission_Other'            
        if ('390'<=rawdata.diag1[i]<='459' or rawdata.diag1[i]=='785'):
            rawdata.diag1[i]= 'Circulatory'
        elif ('460'<=rawdata.diag1[i]<='519' or rawdata.diag1[i]=='786'):
            rawdata.diag1[i]= 'Respiratory'
        elif ('520'<=rawdata.diag1[i]<='579' or rawdata.diag1[i]=='787'):
            rawdata.diag1[i]= 'Digestive'
        elif ('250'<rawdata.diag1[i]<'251'):
            rawdata.diag1[i]= 'Diabetes'
        elif ('800'<=rawdata.diag1[i]<='999'):
            rawdata.diag1[i]= 'Injury'
        elif ('710'<=rawdata.diag1[i]<='739'):
            rawdata.diag1[i]= 'Musculoskeletal'
        elif ('580'<=rawdata.diag1[i]<='629' or rawdata.diag1[i]=='788'):
            rawdata.diag1[i]= 'Genitourinary'
        elif ('140'<=rawdata.diag1[i]<='239'):
            rawdata.diag1[i]= 'Neoplasms'
        else:
            rawdata.diag1[i]= 'OtherDiag'
            
        if ('390'<=rawdata.diag2[i]<='459' or rawdata.diag2[i]=='785'):
            rawdata.diag2[i]= 'Circulatory'
        elif ('460'<=rawdata.diag2[i]<='519' or rawdata.diag2[i]=='786'):
            rawdata.diag2[i]= 'Respiratory'
        elif ('520'<=rawdata.diag2[i]<='579' or rawdata.diag2[i]=='787'):
            rawdata.diag2[i]= 'Digestive'
        elif ('250'<rawdata.diag2[i]<'251'):
            rawdata.diag2[i]= 'Diabetes'
        elif ('800'<=rawdata.diag2[i]<='999'):
            rawdata.diag2[i]= 'Injury'
        elif ('710'<=rawdata.diag2[i]<='739'):
            rawdata.diag2[i]= 'Musculoskeletal'
        elif ('580'<=rawdata.diag2[i]<='629' or rawdata.diag2[i]=='788'):
            rawdata.diag2[i]= 'Genitourinary'
        elif ('140'<=rawdata.diag2[i]<='239'):
            rawdata.diag2[i]= 'Neoplasms'
        else:
            rawdata.diag2[i]= 'OtherDiag'
        
        if ('390'<=rawdata.diag3[i]<='459' or rawdata.diag3[i]=='785'):
            rawdata.diag3[i]= 'Circulatory'
        elif ('460'<=rawdata.diag3[i]<='519' or rawdata.diag3[i]=='786'):
            rawdata.diag3[i]= 'Respiratory'
        elif ('520'<=rawdata.diag3[i]<='579' or rawdata.diag3[i]=='787'):
            rawdata.diag3[i]= 'Digestive'
        elif ('250'<rawdata.diag3[i]<'251'):
            rawdata.diag3[i]= 'Diabetes'
        elif ('800'<=rawdata.diag3[i]<='999'):
            rawdata.diag3[i]= 'Injury'
        elif ('710'<=rawdata.diag3[i]<='739'):
            rawdata.diag3[i]= 'Musculoskeletal'
        elif ('580'<=rawdata.diag3[i]<='629' or rawdata.diag3[i]=='788'):
            rawdata.diag3[i]= 'Genitourinary'
        elif ('140'<=rawdata.diag3[i]<='239'):
            rawdata.diag3[i]= 'Neoplasms'
        else:
            rawdata.diag3[i]= 'OtherDiag'
            
        if rawdata.discharge_disposition_id[i] == '1':
            rawdata.discharge_disposition_id[i] = 'DischargedToHome'
        else:
            rawdata.discharge_disposition_id[i] = 'DischargedToOther'
del rawdata['encounter_id']
del rawdata['patient_nbr']
'''
##########data distribution######
datareadmit=rawdata.copy()
for i in datareadmit.index:
    if datareadmit.readmitted[i] != '<30':
        datareadmit.drop([i],inplace=True)
RDD={}
for col_name in rawdata.columns: 
    tmp1 =rawdata[col_name].value_counts()
    tmp2 =datareadmit[col_name].value_counts()
    tmp=tmp2/tmp1
    RDD[col_name] = tmp
##################################
'''    
############flat data##############        
'''
dumdata=pd.get_dummies(rawdata[["race","gender","age","discharge_disposition_id",
                                "admission_source_id","time_in_hospital",
                                "medical_specialty","diag1","diag2","diag3",
                                "A1Cresult","change","diabetesMed"]])
'''    
dumdata=pd.get_dummies(rawdata[["race","gender","age","discharge_disposition_id",
                                "admission_source_id","time_in_hospital",
                                "medical_specialty","diag1","diag2","diag3",
                                "A1Cresult","change","diabetesMed"]])
dic={'<30':0,'NO':1,'>30':1}
dat=[]
resV=[]
for i in rawdata.index:
    X=[]
    for n in list(dumdata):
        X.append(dumdata[n][i])
    y=dic[rawdata.readmitted[i]]
    dat.append(X)
    resV.append(y)
###################################

###########Outlier Detection#####################
outl=OneClassSVM()
outl.fit(dat)
out=outl.decision_function(dat)
newdatX=[]
res=[]
for i in range(len(dat)):
    if out[i][0]<300:
        newdatX.append(dat[i])
        res.append(resV[i])
################################################

#############feature selection########
newdatY = SelectKBest(chi2, k=15).fit_transform(newdatX, res)
lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(newdatY, res)
model = SelectFromModel(lsvc, prefit=True)
newdat = model.transform(newdatY)
#####################################

#######get balanced trainset and testset#######
trainamount=5000
readmit_index=[]
unreadmit_index=[]
for i in range(len(res)):
    if res[i] == 0:
        readmit_index.append(i)
    else:
        unreadmit_index.append(i)
trainindex1=random.sample(readmit_index,int(trainamount/2))
trainindex2=random.sample(unreadmit_index,int(trainamount/2))
trainindex=[]
for i in range(len(newdat)):
    if i in trainindex1 or i in trainindex2:
        trainindex.append(i)
trainset=[newdat[i] for i in trainindex]
trainres=[res[i] for i in trainindex]
testindexY=[]
for i in range(len(newdat)):
    if i in readmit_index:
        if i not in trainindex1:
            testindexY.append(i)
testindexX=[]
for i in range(len(newdat)):
    if i in unreadmit_index:
        if i not in trainindex2:
            testindexX.append(i)
testindex1=random.sample(testindexY,int(trainamount/8))
testindex2=random.sample(testindexX,int(trainamount/8))
testindex=[]
for i in range(len(newdat)):
    if i in testindex1 or i in testindex2:
        testindex.append(i)
testset=[newdat[i] for i in testindex]
testres=[res[i] for i in testindex]    
############################################

#####Guassian Naive Bayes Classifier######
startTime = time()
clf = GaussianNB()
clf.fit(trainset, trainres)
GaussianNB(priors=None) 
SVMres = clf.predict(testset)
stopTime = time()    
print('use ', stopTime-startTime, ' second')
TP=0
FN=0
FP=0
TN=0
for i in range(len(testres)):
    if testres[i] == SVMres[i]:
        if testres[i] ==1:
            TP += 1
        else:
            TN += 1
    else:
        if testres[i] == 1:
            FN += 1
        else:
            FP += 1
print("For the Gaussian Naive_Bayes Classifier")
print("Accuracy on training set: {:.3f}".format(clf.score(trainset,trainres)))
print("Accuracy on testing set: {:.3f}".format(clf.score(testset,testres)))
print("The confusion matrix is: ")
print("Actual\Predict","Readmitted=No","Readmitted=Yes")
print("Readmitted=No      %s           %s"%(TP,FN))
print("Readmitted=Yes     %s           %s"%(FP,TN))
print("Sensitivity = %.3f"%(TP*1.0/(TP+FP)))
print('Specificity = %.3f'%(TN/(TN+FN)))
print('Precision   = %.3f'%(TP/(TP+FP)))
print('Recall      = %.3f'%(TP/(TP+FN)))
print('F1          = %.3f'%(2*(TP/(TP+FP))*(TP/(TP+FN))/(TP/(TP+FP)+TP/(TP+FN))))
#########################################

#####Bernoulii Naive Bayes Classifier#####
startTime = time()
clf = BernoulliNB()
clf.fit(trainset, trainres)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
SVMres = clf.predict(testset)
stopTime = time()    
print('use ', stopTime-startTime, ' second')
TP=0
FN=0
FP=0
TN=0
for i in range(len(testres)):
    if testres[i] == SVMres[i]:
        if testres[i] ==1:
            TP += 1
        else:
            TN += 1
    else:
        if testres[i] == 1:
            FN += 1
        else:
            FP += 1
print("For the Bernoulli Naive_Bayes Classifier")
print("Accuracy on training set: {:.3f}".format(clf.score(trainset,trainres)))
print("Accuracy on testing set: {:.3f}".format(clf.score(testset,testres)))
print("The confusion matrix is: ")
print("Actual\Predict","Readmitted=No","Readmitted=Yes")
print("Readmitted=No      %s           %s"%(TP,FN))
print("Readmitted=Yes     %s           %s"%(FP,TN))
print("Sensitivity = %.3f"%(TP*1.0/(TP+FP)))
print('Specificity = %.3f'%(TN/(TN+FN)))
print('Precision   = %.3f'%(TP/(TP+FP)))
print('Recall      = %.3f'%(TP/(TP+FN)))
print('F1          = %.3f'%(2*(TP/(TP+FP))*(TP/(TP+FN))/(TP/(TP+FP)+TP/(TP+FN))))
#######################################

#######Nearest Centroid Classifier#### 
startTime = time()
clf = NearestCentroid()
clf.fit(trainset, trainres)
NearestCentroid(metric='euclidean', shrink_threshold=None)
SVMres = clf.predict(testset)
stopTime = time()    
print('use ', stopTime-startTime, ' second')
TP=0
FN=0
FP=0
TN=0
for i in range(len(testres)):
    if testres[i] == SVMres[i]:
        if testres[i] ==1:
            TP += 1
        else:
            TN += 1
    else:
        if testres[i] == 1:
            FN += 1
        else:
            FP += 1
print("For the Nearest Neighbors Classifier")
print("Accuracy on training set: {:.3f}".format(clf.score(trainset,trainres)))
print("Accuracy on testing set: {:.3f}".format(clf.score(testset,testres)))
print("The confusion matrix is: ")
print("Actual\Predict","Readmitted=No","Readmitted=Yes")
print("Readmitted=No      %s           %s"%(TP,FN))
print("Readmitted=Yes     %s           %s"%(FP,TN))
print("Sensitivity = %.3f"%(TP*1.0/(TP+FP)))
print('Specificity = %.3f'%(TN/(TN+FN)))
print('Precision   = %.3f'%(TP/(TP+FP)))
print('Recall      = %.3f'%(TP/(TP+FN)))
print('F1          = %.3f'%(2*(TP/(TP+FP))*(TP/(TP+FN))/(TP/(TP+FP)+TP/(TP+FN))))
###########################################

######Neutral network classifier#####
startTime = time()
clf = MLPClassifier()  # class   
clf.fit(trainset, trainres)# training the svc model
MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-8, hidden_layer_sizes=(9,3), learning_rate='constant',
       learning_rate_init=0.008, max_iter=500,  momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)#i lovenot you  
SVMres = clf.predict(testset)
stopTime = time()    
print('use ', stopTime-startTime, ' second')
TP=0
FN=0
FP=0
TN=0
for i in range(len(testres)):
    if testres[i] == SVMres[i]:
        if testres[i] ==1:
            TP+=1
        else:
            TN+=1
    else:
        if testres[i]==1:
            FN+=1
        else:
            FP+=1
print("For the Neural network Classifier")
print("Accuracy on training set: {:.3f}".format(clf.score(trainset,trainres)))
print("Accuracy on testing set: {:.3f}".format(clf.score(testset,testres)))
print("The confusion matrix is: ")
print("Actual\Predict","Readmitted=No","Readmitted=Yes")
print("Readmitted=No      %s           %s"%(TP,FN))
print("Readmitted=Yes     %s           %s"%(FP,TN))
print("Sensitivity = %.3f"%(TP*1.0/(TP+FP)))
print('Specificity = %.3f'%(TN/(TN+FN)))
print('Precision   = %.3f'%(TP/(TP+FP)))
print('Recall      = %.3f'%(TP/(TP+FN)))
print('F1          = %.3f'%(2*(TP/(TP+FP))*(TP/(TP+FN))/(TP/(TP+FP)+TP/(TP+FN))))
#######################################

######Support vector#################
startTime = time()
clf = SVC()  # class   
clf.fit(trainset, trainres)# training the svc model
SVC(C=1, cache_size=200, class_weight={}, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)#i lovenot you  

SVMres = clf.predict(testset)
stopTime = time()    
print('use ', stopTime-startTime, ' second')
TP=0
FN=0
FP=0
TN=0
for i in range(len(testres)):
    if testres[i] == SVMres[i]:
        if testres[i] ==1:
            TP += 1
        else:
            TN += 1
    else:
        if testres[i] == 1:
            FN += 1
        else:
            FP += 1
print("For the SVM Classifier")
print("Accuracy on training set: {:.3f}".format(clf.score(trainset,trainres)))
print("Accuracy on testing set: {:.3f}".format(clf.score(testset,testres)))
print("The confusion matrix is: ")
print("Actual\Predict","Readmitted=No","Readmitted=Yes")
print("Readmitted=No      %s           %s"%(TP,FN))
print("Readmitted=Yes     %s           %s"%(FP,TN))
print("Sensitivity = %.3f"%(TP*1.0/(TP+FP)))
print('Specificity = %.3f'%(TN/(TN+FN)))
print('Precision   = %.3f'%(TP/(TP+FP)))
print('Recall      = %.3f'%(TP/(TP+FN)))
print('F1          = %.3f'%(2*(TP/(TP+FP))*(TP/(TP+FN))/(TP/(TP+FP)+TP/(TP+FN))))
########################################

################Decision Tree########
startTime = time()
clf = tree.DecisionTreeClassifier()
clf.fit(trainset, trainres)
SVMres = clf.predict(testset)
stopTime = time()    
print('use ', stopTime-startTime, ' second')
TP=0
FN=0
FP=0
TN=0
for i in range(len(testres)):
    if testres[i] == SVMres[i]:
        if testres[i] ==1:
            TP += 1
        else:
            TN += 1
    else:
        if testres[i] == 1:
            FN += 1
        else:
            FP += 1
print("For the SVM Classifier")
print("Accuracy on training set: {:.3f}".format(clf.score(trainset,trainres)))
print("Accuracy on testing set: {:.3f}".format(clf.score(testset,testres)))
print("The confusion matrix is: ")
print("Actual\Predict","Readmitted=No","Readmitted=Yes")
print("Readmitted=No      %s           %s"%(TP,FN))
print("Readmitted=Yes     %s           %s"%(FP,TN))
print("Sensitivity = %.3f"%(TP*1.0/(TP+FP)))
print('Specificity = %.3f'%(TN/(TN+FN)))
print('Precision   = %.3f'%(TP/(TP+FP)))
print('Recall      = %.3f'%(TP/(TP+FN)))
print('F1          = %.3f'%(2*(TP/(TP+FP))*(TP/(TP+FN))/(TP/(TP+FP)+TP/(TP+FN))))
########################################

############Random forest#################
startTime = time()
clf = RandomForestClassifier()
clf.fit(trainset, trainres)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10000, n_jobs=-1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
SVMres = clf.predict(testset)
stopTime = time()    
print('use ', stopTime-startTime, ' second')
TP=0
FN=0
FP=0
TN=0
for i in range(len(testres)):
    if testres[i] == SVMres[i]:
        if testres[i] ==1:
            TP += 1
        else:
            TN += 1
    else:
        if testres[i] == 1:
            FN += 1
        else:
            FP += 1
print("For the RandomForestClassifier")
print("Accuracy on training set: {:.3f}".format(clf.score(trainset,trainres)))
print("Accuracy on testing set: {:.3f}".format(clf.score(testset,testres)))
print("The confusion matrix is: ")
print("Actual\Predict","Readmitted=No","Readmitted=Yes")
print("Readmitted=No      %s           %s"%(TP,FN))
print("Readmitted=Yes     %s           %s"%(FP,TN))
print("Sensitivity = %.3f"%(TP*1.0/(TP+FP)))
print('Specificity = %.3f'%(TN/(TN+FN)))
print('Precision   = %.3f'%(TP/(TP+FP)))
print('Recall      = %.3f'%(TP/(TP+FN)))
print('F1          = %.3f'%(2*(TP/(TP+FP))*(TP/(TP+FN))/(TP/(TP+FP)+TP/(TP+FN))))
################################################

