# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 12:34:28 2018

@author: 87405
"""
import pandas as pd
from time import time

l1=pd.read_csv("D:/TEAM/460tech/miniProj/adult.csv",
               delimiter=",", usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14),
               names=["age","work","fnlwgt","edu","edu_n","mar_s","job","rel",
                      "race","sex","cap_g","cap_l","hpw","nat","sal"])

#pre-processing: abandon attribute including fnlwgt, cap_g, and cap_l
age_group=list(l1.age)
for i in range(len(age_group)):
    if age_group[i] <= 30:
        age_group[i] = 'age<=30'
    elif  age_group[i] > 30 and age_group[i] <=40:
        age_group[i] = '30<age<=40'
    elif  age_group[i] > 40 and age_group[i] <=50:
        age_group[i] = '40<age<=50'
    elif  age_group[i] > 50 and age_group[i] <=60:
        age_group[i] = '50<age<=60'
    elif  age_group[i] > 60:
        age_group[i] = 'age>60'
l1.age=list(age_group)
hpw_group=list(l1.hpw)
for i in range(len(hpw_group)):
    if hpw_group[i] <= 25:
        hpw_group[i] = 'hpk<=25'
    elif hpw_group[i] > 25 and hpw_group[i] <=35:
        hpw_group[i] = '25<hpk<=35'
    elif hpw_group[i] > 35 and hpw_group[i] <=45:
        hpw_group[i] = '35<hpk<=45'
    elif hpw_group[i] > 45 and hpw_group[i] <=55:
        hpw_group[i] = '45<hpk<=55'
    elif hpw_group[i] > 55:
        hpw_group[i] = 'hpk>55'
l1.hpw=list(hpw_group)
edu_n_group=list(l1.edu_n)
for i in range(len(edu_n_group)):
    edu_n_group[i]=str(edu_n_group[i])
l1.edu_n=list(edu_n_group)
del l1['fnlwgt']
del l1['cap_g']
del l1['cap_l']

l2=[]
for i in range(len(l1.age)):
    l2.append([l1.age[i],l1.work[i],l1.edu[i],l1.edu_n[i], \
               l1.mar_s[i],l1.job[i],l1.rel[i],l1.race[i],l1.sex[i], \
               l1.hpw[i],l1.nat[i],l1.sal[i]])

index = {}
for col_name in l1.columns: 
    tmp =l1[col_name].value_counts()
    index[col_name] = tmp
print (index)

def apriori(D, minSup, index):
#keys means one itemset
#key means one item in itemset
#cutKeys means keys after cutting step
#C means Support	
#count all keys and their support
#type of Support is dict
#keys1 is the list of all single keys
    n = len(D)
    cutKeys1 = []
    for col_name in index:
        for I in (index[col_name].keys()):
            if index[col_name][I]*1.0/n >= minSup:
                cutKeys1.append([I])
    i=0
    while i < len(D):
        j=0
        while j < len(D[i]):
            if [D[i][j]] not in cutKeys1:
                D[i].remove(D[i][j])    
            else:
                j+=1
        if D[i]==[]:
            D.remove(D[i])
        else:
            i+=1
#scan keys first time to throw out infrequent item    
    keys = cutKeys1
    all_keys = []
    trash = []
    while keys != []:
        Sup = getC(D, keys)
        cutKeys = getCutKeys(keys, Sup, minSup, n, trash)[0]
        for key in cutKeys:
            all_keys.append(key)
        keys = apriori_gen(cutKeys, trash)
    Sup = getC(D,all_keys)
    all_keys = getCutKeys(all_keys, Sup, minSup, n,trash)[0]
    return all_keys

def getC(D, keys):
# count support for each key in keys
	Sup = []
	for key in keys:
		s = 0
		for T in D:
			have = True
			for k in key:
				if k not in T:
					have = False
			if have:
				s += 1
		Sup.append(s)
	return Sup

def getCutKeys(keys, Sup, minSup, length, trash):
#cutting step to remove those < minsup
    i=0
    while i < len(keys):
        if float(Sup[i]) / length < minSup:
            trash.append(keys[i])
            del keys[i]
            del Sup[i]
        else:
            i += 1
    return keys,trash


def apriori_gen(keys1, trash):
#develop keys with other key
    keys2 = []
    for k1 in keys1:
        for k2 in keys1:
            if k1 != k2:
                key = []
                for k in k1:
                    if k not in key:
                        key.append(k)
                for k in k2:
                    if k not in key:
                        key.append(k)                            
                key.sort()
                if key not in keys2:
                    keys2.append(key)
    for i in range(len(trash)):
        j=0
        while j < len(keys2):
            have = 0
            for k in range(len(trash[i])):
                if trash[i][k] in keys2[j]:
                    have+=1
            if have == len(trash[i]):
                keys2.remove(keys2[j])
            else:
                j+=1
                
    return keys2


startTime = time()
F = apriori(l2, 0.6, index)
print ('frequent itemset, support, confidence')
for i in range(len(F)):
    Support=getC(l2,[F[i]])
    print (F[i],Support[0],Support[0]/len(l2))
stopTime = time()    
print('use ', stopTime-startTime, ' second')
