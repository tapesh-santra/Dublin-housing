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
from matplotlib.animation import FuncAnimation


#READ DATA
def parser(x):
    return datetime.strptime(x,'%d/%m/%Y')
df_pred=pd.read_csv('..//results//predicted.csv',header=0, parse_dates=[0],index_col='END_DTE',squeeze=True,date_parser=parser)
#monthly=pd.read_csv("..//data//monthly.csv",header=0,parse_dates=[0,1], index_col=1, squeeze=True, date_parser=parser)
    
# VISUALIZE
    
x=df_pred.index
Y1 = df_pred['PRPTY_PRICE_ACTUAL'].values
Y2=df_pred['RES_PROP_PRICE_INDX_APART'].values

y1_label='ACTUAL'
y2_label='PREDICTED'

outfile='..//results//FORECAST.gif'

T1= 'ACTUAL'
T2= 'PREDICTED'


fig, ax1 = plt.subplots(figsize=(10,4))
color = 'tab:red'
ax1.set_xlabel('Years')
ax1.set_ylabel(y1_label, color=color,fontsize=12)


I=np.where(Y1!=None)
x1=x[I]
Y1_1=Y1[I]
line1,=ax1.plot(x1, Y1_1[I], color=color)
line1_1,=ax1.plot(x1,Y1_1,'o',color=color,ms=10)
t1=ax1.text(x1[-1], Y1_1[-1], T1, fontsize=12)
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
    if Y1[i]!=None:
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



