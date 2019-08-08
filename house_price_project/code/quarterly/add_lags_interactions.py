# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:20:37 2019

@author: tapesh.santra
"""
import pandas as pd
def add_lag(df,features,lags):    
    #Creates new data coluns by adding lag to the specified features
    #df= dtaframe
    #features = features that needs to have a lag
    # lags=number of rows that meeds to be shifted
    
    df1=df[features] # 
    
    L=[];#list of lagged datasets
    L.append(df1)
    df_rest=df[list(set(list(df))-set(features))]
    L.append(df_rest)
    for l in lags:
        F1=[]
        for f in features:
            F1.append(f+"_lag_"+str(l))#create feature headers with different names
        df2=df1.shift(l)
        df2.columns=F1
        L.append(df2)
    df3=pd.concat(L,axis=1)
    return (df3,F1)
        
        
    #df2=pd.concat([df, df1.shift(), df1.shift(2)],axis=1)
    #return df1

def add_interactions(df,features):
    #create separate columns containing interactions between pairs of features in the dataset
    new_vars=[]
    for f1 in features:
        for f2 in features:
            if f1 != f2:
                n_f=f1+"_x_"+f2
                new_vars.append(n_f)
                df[n_f]=df[f1]*df[f2]
    return(df,new_vars)

def add_polynomial_terms(df,features,orders):
    #create separate columns containing interactions between pairs of features in the dataset
    new_vars=[]
    for f1 in features:
        for o in orders:
            n_f=f1+"_"+str(o)
            new_vars.append(n_f)
            df[n_f]=df[f1]**o
    return(df,new_vars)