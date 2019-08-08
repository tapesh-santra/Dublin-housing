# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from pandas import datetime
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

#READ DATA



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
 'CNSTR',
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


 #VISUALIZE   
    
x = X.index
Y1 = Y['RES_PROP_PRICE_INDX_APART'].values
Y2=Y['PLAN_PER_DWLN'].values

y1_label='PROPERTY PRICE INDEX'
y2_label='PLANNING PERMISSIONS GRANTED'

outfile='..//results//PLAN.gif'

T1= 'PRPTY PRICE'
T2= 'PLAN PER GRTD'

fig, ax1 = plt.subplots(figsize=(7,4))
color = 'tab:red'
ax1.set_xlabel('Years')
ax1.set_ylabel(y1_label, color=color,fontsize=12)
line1,=ax1.plot(x, Y1, color=color)
line1_1,=ax1.plot(x,Y1,'o',color=color,ms=10)
t1=ax1.text(x[-1], Y1[-1], T1, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color)   

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(y2_label, color=color,fontsize=12)  # we already handled the x-label with ax1
line2,=ax2.plot(x, Y2, color=color)
line2_1,=ax2.plot(x,Y2,'s',color=color,ms=10)
t2=ax2.text(x[-1], Y2[-1], T2, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color)



fig.tight_layout()  # otherwise the right y-label is slightly clipped

def animate(i):
    line1.set_data(x[:i],Y1[:i])  # update the data
    line1_1.set_data(x[i],Y1[i])
    t1.set_position((x[i],Y1[i]))
    line2.set_data(x[:i],Y2[:i])  # update the data
    line2_1.set_data(x[i],Y2[i])
    t2.set_position((x[i],Y2[i]))
    return (line1, line1_1, t1, line2, line2_1,t2)

ani = FuncAnimation(fig, animate, np.arange(1, len(x)), interval=100)

plt.show()
ani.save(outfile, writer='pillow', fps=60)



