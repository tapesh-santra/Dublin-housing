# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
import make_model as mm

#def ConvertFromMonthly(filename):



#date parsers
#parses date from 'dd/mm/yyyy' format into date time format.
def parser(x):
    return datetime.strptime(x,'%d/%m/%Y')


# read monthly quarterly and yearly data
monthly=pd.read_csv("..//..//data//monthly.csv",header=0,parse_dates=[0,1], index_col=1, squeeze=True, date_parser=parser)
quarterly=pd.read_csv("..//..//data//quarterly.csv",header=0,parse_dates=[0,1], index_col=1, squeeze=True, date_parser=parser)
yearly=pd.read_csv("..//..//data//yearly.csv",header=0,parse_dates=[0,1], index_col=1, squeeze=True, date_parser=parser)

#drop missing values

# resample quarterly and yearly data in monthly format
#qrtr_2_mthly=quarterly.resample('1M').interpolate(method='quadratic')
yrly_2_qrtrly=yearly.resample('3M').interpolate(method='quadratic')



#join the 3 datasets to have just one main data
#we don't need the start_dte column any more
monthly=monthly.drop(['START_DTE'],axis=1)
#qrtr_2_mthly=qrtr_2_mthly.drop(['START_DTE'],axis=1)
yrly_2_qrtrly=yrly_2_qrtrly.drop(['START_DTE'],axis=1)

# left join by end dte
data=quarterly.join([monthly,yrly_2_qrtrly],how='left')

# set target and predictor variables

target='RES_PROP_PRICE_INDX_APART'

predictors=['UNEMPL_RTE',
 'CPI',
 'AT_WORK',
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
 #'OCCUP_AGRI',
# 'OCCUP_IND',
 'OCCUP_CNSTR',
# 'OCCUP_RETAIL',
# 'OCCUP_TRPT',
# 'OCCUP_ACCOM',
 'OCCUP_IT',
 'OCCUP_FIN',
# 'OCCUP_SCI',
 #'OCCUP_ADMIN',
 'OCCUP_PUB_ADMIN',
 #'OCCUP_EDU',
 #'OCCUP_HLTH',
 'GNP_CURR_MKT_PRC',
 'PLAN_PER_DWLN',
# 'PRPTY_BUILT',
 'Net migration',
 'Population']


#Create features
data['CNSTR']=((data['AVG_INC_CNSTR_SMALL']+data['AVG_INC_CNSTR_MED']+data['AVG_INC_CNSTR_LRG'])/3)*data['OCCUP_CNSTR']
data['IT']=((data['AVG_INC_IT_SMALL']+data['AVG_INC_IT_MED']+data['AVG_INC_IT_LRG'])/3)*data['OCCUP_IT']
data['PUB_ADMIN']=((data['AVG_INC_PUB_ADMIN_SMALL']+data['AVG_INC_PUB_ADMIN_MED']+data['AVG_INC_PUB_ADMIN_LRG'])/3)*data['OCCUP_PUB_ADMIN']
data['FIN']=((data['AVG_INC_FIN_SMALL']+data['AVG_INC_FIN_MED']+data['AVG_INC_FIN_LRG'])/3)*data['OCCUP_FIN']
data['ALL']=((data['AVG_INC_ALL_SMALL']+data['AVG_INC_ALL_MED']+data['AVG_INC_ALL_LRG'])/3)*data['AT_WORK']


# remove the old predictors keep the new ones
predictors=['UNEMPL_RTE',
 'CPI',
 #'CNSTR',
 'IT',
 'PUB_ADMIN',
 'FIN',
 'ALL',
 'GNP_CURR_MKT_PRC',
 'PLAN_PER_DWLN',
 'Net migration',
 'Population']

# all variables
all_vars=predictors.copy()
all_vars.append(target)

#create training dataset
data1=data[all_vars]
data1=data1.dropna(axis=0)



#add lag variables
data2,lagged_vars=add_lag(data1,all_vars,[1])
data2=data2.dropna(axis=0)

#We shall only use the lagged variable as predictors
preds1=lagged_vars

#add time variable
data2['time']=np.arange(data2.shape[0])+1
preds1.append('time')

Y=data2[all_vars]
X=data2[preds1] 

# Divide the data into training and testing data
#this is to see if our strategy is working, we shall leave six months of data for testing
test_mths=6

Y_train=Y.iloc[:Y.shape[0]-test_mths,:]
X_train=X.iloc[:X.shape[0]-test_mths,:] 

Y_test=Y.iloc[Y.shape[0]-test_mths:,:]     
X_test=X.iloc[X.shape[0]-test_mths:,:]  


print('############### Data preparation complete, entering model building phase ############')
 
params={'STANDARDIZE':True, 'NONLIN_TYPE':'POLY','ORDER':[2]}
model=mm.model(params,X_train.copy(),Y_train.copy()) 
pred=model.multistep_forecast(test_mths)
model.plot_coeffs()
dfY1=model.predict(X_train.copy(),plot=True,Y_True=Y_train.copy(),plot_list=['RES_PROP_PRICE_INDX_APART'])
#model.make_model()     
      
      
#################################################################################
#***************** BUILD AND TEST THE MODEL NOW********************************##
################################################################################    

