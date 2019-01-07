# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:19:23 2018

@author: Radian
"""
import pandas as pd
import numpy as np
from time import time

class Node(object):
    def __init__(self, data):
        self.data = data;
        self.count = 1
        self.sonNodes = []
        self.parents = -1
        
    def addSon(self, newSon):
        self.sonNodes.append(newSon)
    
    def deleteSon(self, index):
        del self.sonNodes[index]
    
    def removeSon(self, son):
        self.sonNodes.remove(son)
        
    def findSon(self, sonData):
        sor_id = 0
        for son in self.sonNodes:
            if son.data == sonData:
                return sor_id
            sor_id+=1
        return -1
    def disp(self, ind=1):
        print ('  '*ind, self.data, ' ', self.count)
        for son in self.sonNodes:
            son.disp(ind+1)

class FpTree:
    def __init__(self):
        self.root = Node(data='**root**')
        self.deep_count = 0
        self.bottomSonsDeepCount = []
        self.bottomSons = []
    def createTree(self, data):        
        # Scan each person 
        j=0
        for j in range(len(data)):
            currentNode = self.root   
            # every type of each person
            for i in range(len(data[j])):
                sonData = data[j][i]
                son_id = currentNode.findSon(sonData)
                # searching for existing same node 
                if son_id != -1:
                    currentNode.sonNodes[son_id].count += 1
                    currentNode = currentNode.sonNodes[son_id]
                # add child node if not exist same node
                else:
                    newSon = Node(sonData)
                    newSon.parents = currentNode
                    currentNode.addSon(newSon)
                    # take the new node as the start of new iteration
                    currentNode = newSon                   
            j+=1

    def getMaxFrequencyItem(self):
        self.bottomSonsDeepCount = []
        self.bottomSons = []
        # Deep search for all bottom sons
        for son in self.root.sonNodes:
            self.__accessSon(son)                
        # get the max depth 
        maxDepth = max(self.bottomSonsDeepCount)
        # using the maxdepth for searching all nodes back to root
        for index in range(len(self.bottomSonsDeepCount)):
            if maxDepth == self.bottomSonsDeepCount[index]:
                print('\nMaxFrequencyCount：', maxDepth)
                self.__getFrequencyItem(self.bottomSons[index])
                
    def __accessSon(self, son):
        self.deep_count +=1
        # access to the deep firstly
        if len(son.sonNodes) == 0:
            self.bottomSonsDeepCount.append(self.deep_count)
            self.bottomSons.append(son)
        else:
            # continue on searching inside nodes
            for s in son.sonNodes:
                self.__accessSon(s)        
        self.deep_count -= 1
        
    def __getFrequencyItem(self, bottomSon):
        currentNode = bottomSon
        if currentNode.parents != -1:
            print('\nFrequency： ', currentNode.count)
        else:
            return    
        while currentNode.parents != -1:
            print(currentNode.data)
            currentNode = currentNode.parents
#loading data            
l1 = pd.read_csv('D:/TEAM/460tech/miniProj/adult.csv',
                   names=["age","work","fnlwgt","edu","edu_n","mar_s","job","rel",
                      "race","sex","cap_g","cap_l","hpw","nat","sal"])
#pre-processing
#drop reductant col
l1.drop(['cap_g'],axis=1,inplace=True)
l1.drop(['cap_l'],axis=1,inplace=True)
l1.drop(['fnlwgt'],axis=1,inplace=True)
l1 = l1.replace('?', np.nan)
#replace scatter data with interval
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
#replace int with object type
edu_n_group=list(l1.edu_n)
for i in range(len(edu_n_group)):
    edu_n_group[i]=str(edu_n_group[i])
l1.edu_n=list(edu_n_group)
#create list for data
l2=[]
for i in range(len(l1.age)):
    l2.append([l1.age[i],l1.work[i],l1.edu[i],l1.edu_n[i], \
               l1.mar_s[i],l1.job[i],l1.rel[i],l1.race[i],l1.sex[i], \
               l1.hpw[i],l1.nat[i],l1.sal[i]])
#Set minSup
minSup = 0.6
threshold = l1.index.size * minSup
#Step 1: create index of data
time1=time()
index = {}
for col_name in l1.columns: 
    tmp =l1[col_name].value_counts()
    index[col_name] = tmp
print (index)

time2=time()

#Step 2: scale down data using minSup
n = len(l2)
cutKeys1 = []
A={}
for col_name in index:
    for I in (index[col_name].keys()):
        if index[col_name][I]*1.0/n >= minSup:
            A[I]=index[col_name][I]
B=sorted(A.items(),key=lambda item:item[1],reverse=True)
for i in range(len(B)):
    cutKeys1.append(B[i][0])            
i=0
data=[]
while i < len(l2):
    j=0
    while j < len(l2[i]):
        if l2[i][j] not in cutKeys1:
            l2[i].remove(l2[i][j])    
        else:
            j+=1
    if l2[i]==[]:
            l2.remove(l2[i])
    else:
        data.append(sorted(l2[i],key=lambda x:A[x],reverse=True))
        #sorted personal row by frequency going down
        i+=1

time3=time()
#Step 3: Call the fuction and create FP tree
tree = FpTree()
tree.createTree(data)
tree.getMaxFrequencyItem()
tree.root.disp()
time4 = time()
print('1st scan use ',time2-time1, ' second')
print('type scan use ',time3-time2, ' second')
print('FP growth use ',time4-time3, ' second')
