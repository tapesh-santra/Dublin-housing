# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:30:34 2019

@author: sbi_user
"""

import pandas as pd
from pandas import datetime
from matplotlib import pyplot as plt
from pandas.tseries.offsets import Day, MonthEnd, YearEnd
from add_lags_interactions import add_lag,add_polynomial_terms
import lin_reg_var_select as lrvs
from sklearn.linear_model import MultiTaskElasticNet
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from pandas.tseries.offsets import MonthEnd

#params:
# MODEL_TYPE = String
# PARAMS= {'name':value, 'name':values}
# NONLIN_TYPE =['POLY','LOG','EXP','NONE'], for the momnet just poly
# NONLIN_FEAT =[feature1,feature2...] ** not implemented.. all features will be transformed now
# ORDER= [2,3,4...] if TYPE=Poly
# BASE = [2,10,exp] if TYPE = LOG




class model:
    def __init__(self,params,X,Y):
        self.params=params
        self.original_predictors=list(X)
#        if 'time' in self.original_predictors:
#            self.original_predictors.remove('time')
        if params['NONLIN_TYPE']=='POLY':
            #add non linear terms
            self.X=self.add_nonlinear_terms(X)
            #print(self.X)
            self.Y=Y
        if params['STANDARDIZE']:
            #standardize
            self.standardize()
        self.predictor_names=list(self.X)
        self.target_names=list(Y)
        self.Y_final=self.Y.iloc[-1,:]
        self.time=self.X.iloc[-1, X.columns.get_loc('time')]
        self.date=self.X.index[-1]
#        print(self.X)
#        print(self.Y)
#        print(self.Y_final)
#        print(self.time)
        self.make_model()
    
        

    def add_nonlinear_terms(self,X):
        df,var_names=add_polynomial_terms(X,list(X),self.params['ORDER'])
        return(df)
    
    def standardize(self):
        self.X_mean=self.X.mean()
        self.Y_mean=self.Y.mean()
        self.X_std=self.X.std()
        self.Y_std=self.Y.std()
        self.X=(self.X-self.X_mean)/self.X_std
        self.Y=(self.Y-self.Y_mean)/self.Y_std
        

    def make_model(self):
        max_iter=1000
        tol=0.015
        l1_ratio=0.8 # we want a relatively sparse model
        elastic=MultiTaskElasticNet(fit_intercept=True, max_iter=max_iter,tol=tol,l1_ratio=l1_ratio)
        
        #Note that we are assuming that error are independent of each other GIVEN THE PREDICTORS
        #Otherwise cross validation won't be applicable
        #We will perform a grid search to find best parameters
        
        print('################ Find hyper-parameter values#######################')
        search=GridSearchCV(estimator=elastic,param_grid={'alpha':np.logspace(-5,2,8)},scoring='neg_mean_squared_error',n_jobs=1,refit=True,cv=10)
        search.fit(self.X,self.Y)
        
        #Now create a final elastic net model using the optimal hyper parameters
        print('################ Build final model ##############################')
        optimal_alpha=search.best_params_['alpha']
        #optimal_l1_ratio=search.best_params_['l1_ratio']
        self.model=MultiTaskElasticNet(fit_intercept=True,alpha=optimal_alpha,l1_ratio=l1_ratio,max_iter=max_iter,tol=tol)
        self.model.fit(self.X.values,self.Y.values)
        self.predicted=pd.DataFrame(index=self.Y.index, columns= self.Y.columns, data=self.model.predict(self.X.values))
        self.predicted=self.predicted*self.Y_std+self.Y_mean
        #second_model=(mean_squared_error(y_true=Y_train,y_pred=elastic.predict(X_train)))

    
    def predict(self,X,plot=False,Y_True=None,plot_list=None):
        # If plot = True , Y_true should contain the True values and this function will plot a comparion between true vs predicted
        if self.params['STANDARDIZE'] and self.params['NONLIN_TYPE']=='POLY':
            #X1=X.copy() # don't modify the original
            X1=self.add_nonlinear_terms(X)
            #print('Unnormalized predictors: ',X1)
            X1=(X1-self.X_mean)/self.X_std # standardized
            #print('Normalized predictors: ',X1)
            Y1=self.model.predict(X1.values)
            
            dfY1=pd.DataFrame(index=Y_True.index,columns=list(Y_True),data=Y1)
            dfY1=dfY1*self.Y_std+self.Y_mean
            #Y1=Y1*
            if plot:
                X_ax=Y_True.index
                label_true=[l+'_True' for l in plot_list]
                label_pred=[l+'_Pred' for l in plot_list]
                plt.figure(figsize=(6,4))
                plt.plot(X_ax,Y_True[plot_list],label=label_true)
                plt.plot(X_ax,dfY1[plot_list],label=label_pred)
                plt.legend(loc='best')
                plt.show()
            return(dfY1)
            
            
            
            
    def forecast(self):
        pred=None
        if self.params['STANDARDIZE'] and self.params['NONLIN_TYPE']=='POLY': #if standardized and polynomial
            
            Xp=self.Y_final*self.Y_std + self.Y_mean # destandardize Y, this is needed to calculate the non linear term
            #print(Xp)
            Xp['time']=self.time+1 #- self.X_mean['time'])/self.X_std
            dfp=pd.DataFrame(index=[self.date],columns=self.original_predictors,data=Xp.values.reshape(1,-1))
            dfp=self.add_nonlinear_terms(dfp) # add the non linear terms
            #print(dfp)
            dfp=(dfp-self.X_mean)/self.X_std # standardize, then predict
            #print(dfp)
            pred=self.model.predict(dfp.values)
            self.time=self.time+1
            #print(self.date)
            self.date=self.date+MonthEnd(1)
            
            df=pd.DataFrame(index=[self.date],columns=self.target_names,data=pred)
            self.Y_final=df
            #print(self.date)
        return(pred,self.date)
    
    def multistep_forecast(self,steps):
        df=pd.DataFrame(columns=self.target_names)
        for i in range(steps):
            pred,date=self.forecast()
            print(pred.shape)
            df.loc[date,:]=np.multiply(pred,self.Y_std.values.reshape(1,-1)) + self.Y_mean.values.reshape(1,-1)
        return df
            
    def plot_coeffs(self):
        C=self.model.coef_[-1,:]
        indexes=np.where(np.abs(C)>0.0001)
        
        #significant predictors
        C_sig=C[indexes[0]]
        preds_sig=[self.predictor_names[int(i)] for i in indexes[0]]
        
        f,ax=plt.subplots()
        f.set_size_inches((10,2))
        ax.bar(range(len(C_sig)),C_sig)
        ax.set_xticks(range(len(C_sig)))
        ax.set_xticklabels(labels=preds_sig)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    
    def variable_importance(self,orig_var_names,labels):
        all_preds=list(self.X)# all predictors
        imp=[]
        for v in orig_var_names:
            v1=[ap for ap in all_preds if v in ap]
            print(v1)
            X1=self.X.copy()
            X1[v1]=0
            Y1=self.model.predict(X1)
            imp.append(np.sum((self.Y.values-Y1)**2))
        #print(imp)
        indexes=np.argsort(np.array(imp))
        #print(indexes)
        preds1=[labels[i] for i in indexes]
        imps1=[imp[i] for i in indexes]
        imps1=imps1/np.max(imps1)
        #plot importance
        f,ax=plt.subplots()
        ax.barh(range(len(imp)),imps1)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(labels=preds1)
        ax.set_xlabel(xlabel='Importance',fontsize=12)
        plt.tight_layout()
        plt.show()
        
        

       
                
        
        
    
    