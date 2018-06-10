#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 22:52:11 2017

@author: mdsamad
"""


#from pylab import *

import seaborn.apionly as sns

import time

from sklearn import svm 
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc


from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from itertools import combinations
from sklearn.datasets import load_wine
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import StratifiedKFold

 
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


# Wine dataset

def import_data (db_name= 'wine_2class', 
                 dtArray= None, tget=None, fNames = None):
    
    if db_name == 'wine_2class':
        
        dataSt = load_wine()

        Xr = pd.DataFrame(data=dataSt.data ,    
              columns=dataSt.feature_names)

        y = dataSt.target      

        Xr = Xr.iloc[y<2]
        y = y[y<2]

    if db_name == 'wine_3class':
        
        dataSt = load_wine()

        Xr = pd.DataFrame(data=dataSt.data ,    
              columns=dataSt.feature_names)

        y = dataSt.target      

       


    if db_name == 'titanic':
        
# Titanic dataset

        tips_data = sns.load_dataset('titanic')

        tips_data = tips_data.dropna()
        
        y = tips_data.survived 
        
        Xr = tips_data.drop(['survived','alive','deck','sex'],axis=1)
        
        
    if db_name == None:
        
        Xr = pd.DataFrame(data=dtArray ,    
              columns=fNames)

        y = tget  
        
        

    return Xr, y




def exhAna (Xdata, y, cv=10, max_dim=None):
    
    
    X = Xdata
    
    print ('Input data dimension after processing', X.shape)
    
    cat_var = X.select_dtypes(include=['category','object'])
    

    for p in list(cat_var):
        
        n_var = pd.get_dummies(X[p])
        X = pd.concat([X, n_var], axis=1)
                       
    X=X.drop(cat_var,axis=1)
    
    print ('Input data dimension after processing:', X.shape)
    
    fList = list(X) # list of features
    nFea = X.shape[1] # number of features
    
    if max_dim == None:
        max_dim = nFea
     
    # SVM Classifier model :Selecting Classifier SVM
    sclf = svm.SVC(C=1, cache_size=600, kernel='linear', probability = True)
    
    ## Create the classification pipeline
    pipe_lr = Pipeline([('scl',StandardScaler()),('clf',sclf)])
     
    
    # total number of features
    bestMean = []
    bestStd = []
    mx_chk = 0
  
    for k in range(max_dim):
        
        # Value for r in nCr
        numFr = k+1 
        print ('For dimension', numFr)
        st_time = time.time()
        # Creating all combinations of r features out of n such that nCr
        Ind   = np.asarray (list(combinations(range(nFea),numFr))) # In list format
       
        # Total number of combinations (nCr)
        nComb = Ind.shape[0] 
    
        maccur = []
        saccur = []

        # Evaluate each and every combination 
        for r in range (0, nComb):
            
            trainData = X.iloc[:,Ind[r,:]]
    
            scores = cross_val_score(estimator=pipe_lr, X=trainData, y=y, cv=cv,
                                     scoring = 'roc_auc', n_jobs = 1)
                    
            maccur.append (np.mean(scores))
            saccur.append (np.std(scores))
            
        bestInd = maccur.index (max(maccur)) 
       
        Indx = Ind [bestInd,:]

        print ('Best mean AUC', maccur[bestInd])
        print ('Best std AUC', saccur[bestInd])
        
        el_time = time.time() - st_time
    
        print ('Time elapsed', el_time)
        print ('For total combinations of', nComb)
    
        bst_fea = []
        for r in Indx:    
           bst_fea.append(fList[r]) 
           print(fList[r])
           
       
        bestMean.append (maccur[bestInd])
        bestStd.append (saccur[bestInd]/np.sqrt(len(scores)))
        
        if mx_chk<maccur[bestInd]:
                mx_chk = maccur[bestInd]
                mx_fea = bst_fea
        
        else:
            
            #print('GOOD ENOUGH')
           break
            
    
    print ('Best dimension', len(mx_fea))
    print ('Best feature AUC', mx_chk)
    print ('Best Feature Combinaiton', mx_fea)
    
    
    ## Fitting the classifier model using best features
    bstModel = pipe_lr.fit(X[mx_fea],y)
    bst_clf = bstModel.named_steps['clf']
    print("Coefficients",bst_clf.coef_)
    print ("Intercept", bst_clf.intercept_)
    
    print ('Decision equation,')
    print ('Y =', end="")
    
    
    bst_coef = bst_clf.coef_[0]
    
    for t in range (len(mx_fea)):
        
        print ('(',bst_coef[t],')','*',mx_fea[t], '+', end="")
    
    print (bst_clf.intercept_[0])
    
    return X, bst_coef, mx_fea
    
    # Take two best feautures from best combination of features
    # Plot decision boundary on 2D feature map
    
    
def exhAnaMulti (Xdata, y, cv=10, max_dim=None):
    
    
    X = Xdata
    
    print ('Input data dimension after processing', X.shape)
    
    cat_var = X.select_dtypes(include=['category','object'])
    
    ncls = len(np.unique(y))
    

    for p in list(cat_var):
        
        n_var = pd.get_dummies(X[p])
        X = pd.concat([X, n_var], axis=1)
                       
    X=X.drop(cat_var,axis=1)
    
    print ('Input data dimension after processing:', X.shape)
    
    fList = list(X) # list of features
    nFea = X.shape[1] # number of features
    
    if max_dim == None:
        max_dim = nFea
     
    # SVM Classifier model :Selecting Classifier SVM
    sclf = OneVsRestClassifier(SVC(kernel='linear', probability=True, cache_size=600))

    ## Create the classification pipeline
    pipe_lr = Pipeline([('scl',StandardScaler()),('clf',sclf)])
    kfold = StratifiedKFold(y=y, n_folds=cv, random_state=1)

     
    
    # total number of features
    bestMean = []
    bestStd = []
    mx_chk = 0
  
    for k in range(max_dim):
        
        # Value for r in nCr
        numFr = k+1 
        print ('For dimension', numFr)
        st_time = time.time()
        # Creating all combinations of r features out of n such that nCr
        Ind   = np.asarray (list(combinations(range(nFea),numFr))) # In list format
       
        # Total number of combinations (nCr)
        nComb = Ind.shape[0] 
    
        maccur = []
        saccur = []

        # Evaluate each and every combination 
        for r in range (0, nComb):
            
            trainData = X.iloc[:,Ind[r,:]]

            mauc = []
            for t, (train, test) in enumerate (kfold):
           
                probas = pipe_lr.fit(trainData.values[train],
                                 y[train]).predict_proba(trainData.values[test])
     
                roc_auc = []
                for p in range (ncls):
                    
                     fpr0, tpr0, _ = roc_curve(y[test],probas[:,p], pos_label=p)
                     roc_auc.append(auc(fpr0,tpr0))
           

                mauc.append(np.mean(roc_auc)) # Mean AUCs over three groups
          
  
        # Mean AUCs over five fold Cross validation
        
            maccur.append (np.mean(mauc))
            saccur.append (np.std(mauc))
            
        bestInd = maccur.index (max(maccur)) 
       
        Indx = Ind [bestInd,:]

        print ('Best mean AUC', maccur[bestInd])
        print ('Best std AUC', saccur[bestInd])
        
        el_time = time.time() - st_time
    
        print ('Time elapsed', el_time)
        print ('For total combinations of', nComb)
    
        bst_fea = []
        for r in Indx:    
           bst_fea.append(fList[r]) 
           print(fList[r])
           
       
        bestMean.append (maccur[bestInd])
        bestStd.append (saccur[bestInd]/np.sqrt(cv))
        
        if mx_chk<maccur[bestInd]:
                mx_chk = maccur[bestInd]
                mx_fea = bst_fea
        
        else:
            
            #print('GOOD ENOUGH')
           break
            
    
    print ('Best dimension', len(mx_fea))
    print ('Best feature AUC', mx_chk)
    print ('Best Feature Combinaiton', mx_fea)
  
    
    

def deciBndry (X, y, bst_coef, mx_fea):
    
    # SVM Classifier model :Selecting Classifier SVM
    sclf = svm.SVC(C=1, cache_size=600, kernel='linear', probability = True)
    
    ## Create the classification pipeline
    pipe_lr = Pipeline([('scl',StandardScaler()),('clf',sclf)])

    print ('Plotting 2D map of two best features')
    
   
    ind2 = sorted(range(len(bst_coef)), reverse=True, key = lambda k: np.abs(bst_coef[k]))[0:2]
    
    bst2fea = []
    for k in ind2:
        bst2fea.append(mx_fea[k])
    
    print ('Top 2 features in the best combinations', bst2fea)
    
    bst2 = pipe_lr.fit(X[bst2fea],y)
    
    plt.scatter(X[bst2fea[0]].loc[y==True], X[bst2fea[1]].loc[y == True], s=30, edgecolors='black',
              facecolors='red', linewidths=1, label='Positive case')


    plt.scatter(X[bst2fea[0]].loc[y == False], X[bst2fea[1]].loc[y == False], s=30,edgecolors='black',
                   facecolors='blue', marker ='s', linewidths=1, label='Negative case')
    
    plt.xlabel(bst2fea[0])
    plt.ylabel(bst2fea[1])
    
    # Create a mesh to plot in
    x_min, x_max = X[bst2fea[0]].min(), X[bst2fea[0]].max()
    y_min, y_max = X[bst2fea[1]].min(), X[bst2fea[1]].max() 
    
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = bst2.predict(np.c_[xx.ravel(), yy.ravel()])
   
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.gray, alpha=0.2)
    
    plt.legend(loc='best', ncol=1, mode="", shadow=False, fontsize = 12, fancybox=True)
    

#Xr, y = import_data(db_name='wine')    

#X, bst_coef, mx_fea = exhAna(Xr,y,cv=10, max_dim=10)

#deciBndry(X, y, bst_coef=bst_coef, mx_fea= mx_fea)














