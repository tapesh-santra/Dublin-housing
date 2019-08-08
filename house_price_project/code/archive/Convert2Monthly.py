# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from pandas import datetime
from matplotlib import pyplot as plt
from pandas.tseries.offsets import Day, MonthEnd, YearEnd
from add_lags_interactions import add_lag
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


#def ConvertFromMonthly(filename):



#date parsers
#parses date from 'dd/mm/yyyy' format into date time format.
def parser(x):
    return datetime.strptime(x,'%d/%m/%Y')


# read monthly quarterly and yearly data
monthly=pd.read_csv("..//data//monthly.csv",header=0,parse_dates=[0,1], index_col=1, squeeze=True, date_parser=parser)
quarterly=pd.read_csv("..//data//quarterly.csv",header=0,parse_dates=[0,1], index_col=1, squeeze=True, date_parser=parser)
yearly=pd.read_csv("..//data//yearly.csv",header=0,parse_dates=[0,1], index_col=1, squeeze=True, date_parser=parser)

#drop missing values

# resample quarterly and yearly data in monthly format
qrtr_2_mthly=quarterly.resample('1M').interpolate(method='quadratic')
yrly_2_mthly=yearly.resample('1M').interpolate(method='quadratic')



#join the 3 datasets to have just one main data
#we don't need the start_dte column any more
monthly=monthly.drop(['START_DTE'],axis=1)
qrtr_2_mthly=qrtr_2_mthly.drop(['START_DTE'],axis=1)
yrly_2_mthly=yrly_2_mthly.drop(['START_DTE'],axis=1)

# left join by end dte
data=monthly.join([qrtr_2_mthly,yrly_2_mthly],how='left')

# set target and predictor variables

target='RES_PROP_PRICE_INDX_APART'

predictors=['UNEMPL_RTE',
 'CPI',
 #'AT_WORK',
 'AVG_INC_ALL_SMALL',
 'AVG_INC_CNSTR_SMALL',
 'AVG_INC_IT_SMALL',
 'AVG_INC_PUB_ADMIN_SMALL',
 'AVG_INC_FIN_SMALL',
 'AVG_INC_ALL_MED',
 'AVG_INC_CNSTR_MED',
 'AVG_INC_IT_MED',
 'AVG_INC_PUB_ADMIN_MED',
 'AVG_INC_FIN_MED',
 'AVG_INC_ALL_LRG',
 'AVG_INC_CNSTR_LRG',
 'AVG_INC_IT_LRG',
 'AVG_INC_PUB_ADMIN_LRG',
 'AVG_INC_FIN_LRG',
 'OCCUP_AGRI',
 'OCCUP_IND',
 'OCCUP_CNSTR',
 'OCCUP_RETAIL',
 'OCCUP_TRPT',
 'OCCUP_ACCOM',
 'OCCUP_IT',
 'OCCUP_FIN',
 'OCCUP_SCI',
 'OCCUP_ADMIN',
 'OCCUP_PUB_ADMIN',
 'OCCUP_EDU',
 'OCCUP_HLTH',
 'GNP_CURR_MKT_PRC',
 'PLAN_PER_DWLN',
# 'PRPTY_BUILT',
 'Net migration',
 'Population']

# all variables
all_vars=predictors.copy()
all_vars.append(target)

#create training dataset
data1=data[all_vars]
data1=data1.dropna(axis=0)

#Create features



#standardize data , this will give us much better fir
data1=(data1-data1.mean())/data1.std()

#add lag variables
data2=add_lag(data1,all_vars,[1])
data2=data2.dropna(axis=0)

#We shall only use the lagged variable as predictors
preds1=list(data2)

for p in predictors:
    preds1.remove(p)
preds1.remove(target)

print('############### Data preparation complete, entering model building phase ############')

#we shall fit a multivariate regression
# our output variables are the original variables
# our input variables are the lagged variables
Y=data2[all_vars]
X=data2[preds1] 

# We shall first devide the data into training and testing data
#this is to see if our strategy is working, we shall leave six months of data for testing
test_mths=12

Y_train=Y.iloc[:Y.shape[0]-test_mths,:]
X_train=X.iloc[:X.shape[0]-test_mths,:] 

Y_test=Y.iloc[Y.shape[0]-test_mths:,:]     
X_test=X.iloc[X.shape[0]-test_mths:,:]  
      
 
max_iter=1000
tol=0.015
l1_ratio=0.95 # we want a relatively sparse model
elastic=MultiTaskElasticNet(fit_intercept=True, max_iter=max_iter,tol=tol,l1_ratio=l1_ratio)

#Note that we are assuming that error are independent of each other GIVEN THE PREDICTORS
#Otherwise cross validation won't be applicable
#We will perform a grid search to find best parameters

print('################ Find hyper-parameter values#######################')
search=GridSearchCV(estimator=elastic,param_grid={'alpha':np.logspace(-5,2,8)},scoring='neg_mean_squared_error',n_jobs=1,refit=True,cv=10)
search.fit(X_train,Y_train)

#Now create a final elastic net model using the optimal hyper parameters
print('################ Build final model ##############################')
optimal_alpha=search.best_params_['alpha']
#optimal_l1_ratio=search.best_params_['l1_ratio']
elastic=MultiTaskElasticNet(fit_intercept=True,alpha=optimal_alpha,l1_ratio=l1_ratio,max_iter=max_iter,tol=tol)
elastic.fit(X_train,Y_train)
second_model=(mean_squared_error(y_true=Y_train,y_pred=elastic.predict(X_train)))

# moment of truth
coeff=elastic.coef_
intcpt= elastic.intercept_.reshape(len(elastic.intercept_),1)

#plot the coefficients
#not that the last variable is the target variable
#lets take the last coefficient rows
C=coeff[-1,:]
indexes=np.where(np.abs(C)>0.0001)

#significant predictors
C_sig=C[indexes[0]]
preds_sig=[preds1[int(i)] for i in indexes[0]]

f,ax=plt.subplots()
f.set_size_inches((10,2))
ax.bar(range(len(C_sig)),C_sig)
ax.set_xticks(range(len(C_sig)))
ax.set_xticklabels(labels=preds_sig)
plt.xticks(rotation=90)
plt.show()

#X_last=X_test.iloc[0,:].values.T
#X_last=X_last.reshape(X_last.shape[0],1)
#
#Y_pred=np.zeros((Y_test.shape[1],test_mths)) # the dimention is: each column is a prediction, this is opposite of the test data set where each row is a datapoint
#for i in range(test_mths):
#    y=np.dot(coeff,X_last)+intcpt 
#    Y_pred[:,i]=y[:,0]
#    X_last=y

X_last=X_test.iloc[0,:].values
Y_pred=np.zeros((test_mths,Y_test.shape[1]))
for i in range(test_mths):
    y=elastic.predict(X_last.reshape(1,-1))
    #print(y)
    Y_pred[i,:]=y
    X_last=y 
Y_pred=Y_pred.T
#plot the prediction and the real values side by side    
plt.figure(2)
plt.plot(Y_test.index,Y_test.values[:,-1])
plt.plot(Y_test.index,Y_pred[-1,:])
plt.ylim([0,np.max(Y_pred)])
plt.show()



