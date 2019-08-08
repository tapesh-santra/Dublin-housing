# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 21:19:09 2017

@author: ANAHILIX
"""
import numpy as np
#import math
import time
from joblib import Parallel, delayed
from itertools import compress
import pandas as pd
import sys
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from scipy.linalg import LinAlgError
#from CategoricalDataProcessor import CategoricalDataProcessor as cdp
#import pickle
from collections import defaultdict
#import joblib.parallel
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

#from sklearn.preprocessing import 
#%%
class regression_with_variable_selection:
    def __init__(self,params):
        # K is the number of neighbour 
        self.params=params
        #self.P=P
        #self.N=N
        
        
            
    def grow_by_one(self,M,V,TH,Y,X):
        #M=Current model
        #V=variables to choose from
        #TH=error threshold
        #Y=output
        #X=input data
        #F= feature type flag, true=categorical, false=continuous
        ##K=minimum number of nearest neighbour
        #T=Type of error
        SGM=[]
        ERR=[]
        MINERR=10**50
      
        SGM=[np.append(M,v) for v in V] # list of all possible grown models
        #Models=Parallel(n_jobs=2,backend="threading")(delayed(self.grow)(gm,X,Y) for gm in SGM) # Errors of each models
        Models=[self.grow(gm,X,Y) for gm in SGM]
        ERR=[m['ERR'] for m in Models]
        E1=np.array(ERR)# convert to array
        #print('Error Vector: ',ERR)
        #print('Sum of all errors..: ',np.sum(E1),', Dimension of the error vector: ', np.shape(E1))
        #print(np.shape(E1),np.shape(TH),E1,TH)
        I=(E1<=TH) # Index of models with errors less than the threshold
        SGM=list(compress(SGM,I)) # select models with errors less than the threshold
        ERR=list(compress(ERR,I))#errors of these models
        MINERR=np.min(E1) # select minimum error
        #print(SGM)
        ALL_MODELS={'SGM':SGM ,'ERR':ERR , 'MINERR':MINERR} # put everything in a dictionary
        #if self.params['ALGO']=='SSE_BIC':
        betas=[m['BETA'] for m in Models]# convert to array
        ALL_MODELS['BETA']=list(compress(betas,I))
        return ALL_MODELS       
            
    
    def grow(self,GM,X,Y):  
       #print('In self.grow')
       #GM=np.append(M,v) # Grow the model by one
        #d0l=di[v]# initial parameter for the new kernel
        #d0=np.append(d,d0l); # append it to the optimal parameters of the old kernels
        #print("We are here",V)
       #print(GM)
       X1=X[GM]# get features for the grown model 
       
        #print(X1)
       I=X1.notnull() # index of non null elements
       I1=np.ones(np.shape(I)[0],dtype=bool) # initialize boolean index
       #print(GM)
       for gmv in GM:
           I1=I1&I[gmv] # take only non null data
            
        #print(I1)
       e=10**50;#assign an arbitrarily large error to a model that does not have any data to train
       K=sum(I1)
       model={'ERR':e} # model error
       if K>=2: #self.params['K']:
           #print(np.shape(X1.values))
           X2=[]
           #print(X1.shape)
           if X1.shape[1]==1:
               X2=X1.loc[I1].values #features
           else:
               X2=X1.loc[I1,:].values 
                  
           #print(X2)
           Y2=Y[I1].values.reshape(len(Y),1) #outputs
           
           #print('******************************',GM)
           e=0;
           beta=[]
           if self.params['ALGO']=="SSE_BIC":
               e,beta=self.mse_bic(Y2,X2)
           elif self.params['ALGO']=="SSE_AIC":
               e,beta=self.mse_aic(Y2,X2)
           elif self.params['ALGO']=="BAYES":
               e,beta=self.bayesian_reg(Y2,X2)
           elif self.params['ALGO']=="LASSO":
               e,beta=self.lasso_reg(Y2,X2)
           elif self.params['ALGO']=="ELASTIC_NET":
               e,beta=self.elastic_net_reg(Y2,X2)
           elif self.params['ALGO']=="RIDGE":
               e,beta=self.ridge_reg(Y2,X2)

           
               
           model['ERR']=e # store error
           model['BETA']=beta # store beta               
           #print('regression error for model : ', GM, 'is' ,e)
           
       elif K==1:
           e=10**50;
           model['ERR']=beta;
           #np.abs(Y[I1].values[0]) prediction is the same as the output
       #   print('1 sample error is £££££££££: ', e)
       #   K1=self.params['K']
           
           
       return(model)
    
        
    def mse_bic(self,Y,X): # normal least square with BIC 
        if not X.size:
            X1=np.ones((Y.shape[0],1))
        else:
            X1=np.insert(X,0,1,axis=1)
        n=np.shape(X1)[0]
        p=np.shape(X1)[1]
        X1t=np.transpose(X1)
        #print(np.matmul(np.matmul(X1t,X1),X1t))
        beta=np.matmul(np.matmul(np.linalg.inv(np.matmul(X1t,X1)),X1t),Y)
        
        Y1=np.matmul(X1,beta);
        se=np.mean(np.power((Y-Y1),2))
        e=n*np.log(se)+ p*np.log(n)
        return e,beta
    
    def mse_aic(self,Y,X): # normal least square with AIC
        e=10**50
        try:           
            if not X.size:
                X1=np.ones((Y.shape[0],1))
            else:
                X1=np.insert(X,0,1,axis=1)
            n=np.shape(X1)[0]
            p=np.shape(X1)[1]+1
    
            X1t=np.transpose(X1)
            #print(np.matmul(np.matmul(X1t,X1),X1t))
            #M=np.matmul(np.linalg.inv(np.matmul(X1t,X1)),X1t)
            #print(M.shape,Y.shape)
            #N=np.matmul(M,Y.reshape(len(Y),1))
            #print(N)
            #print(X1t.shape,X1.shape,Y.shape,len(X1t),len(X1),len(Y))
            beta=np.matmul(np.matmul(np.linalg.inv(np.matmul(X1t,X1)),X1t),Y)
            #print(beta)
            Y1=np.matmul(X1,beta);
            #print(np.shape(Y1))
            se=np.mean(np.power((Y-Y1),2))
            e=n*np.log(se)+ 2*p
        except LinAlgError:
             #print(X)
             return (10**50,[])
        #print(len(e),len(beta))    
        return e,beta  
    
    def lasso_reg(self,Y,X): # lasso regression
        if not X.size:
            X1=np.ones((Y.shape[0],1))
        else:
            X1=np.insert(X,0,1,axis=1)
        
        clf = Lasso(alpha=self.params['alpha'],fit_intercept=False)
        clf.fit(X1,Y)
        beta=clf.coef_;
        #beta=np.insert(beta, 0,clf.intercept_)
        
        Y1=clf.predict(X1)
        #print(np.shape(Y1))
        sse=np.sum(np.power((Y-np.reshape(Y1,-1)),2))
        return sse,beta
    
    def elastic_net_reg(self,Y,X): # elastic net regression
        if not X.size:
            X1=np.ones((Y.shape[0],1))
        else:
            X1=np.insert(X,0,1,axis=1)
        clf = ElasticNet(alpha=self.params['alpha'],fit_intercept=False,l1_ratio=self.params['L1_RATIO'])
        clf.fit(X1,Y)
        beta=clf.coef_;
        #beta=np.insert(beta, 0,clf.intercept_)
        Y1=clf.predict(X1)
        sse=np.sum(np.power((Y-np.reshape(Y1,-1)),2))
        return sse,beta
    
    def ridge_reg(self,Y,X): #ridge regression
        if not X.size:
            X1=np.ones((Y.shape[0],1))
        else:
            X1=np.insert(X,0,1,axis=1)
        clf=Ridge(self.params['alpha'], fit_intercept=False)
        clf.fit(X1,Y)
        beta=clf.coef_;
        #beta=np.insert(beta, 0,clf.intercept_)
        Y1=clf.predict(X1)
        sse=np.sum(np.power((Y-np.reshape(Y1,-1)),2))
        return sse,beta
        
    
    def bayesian_reg(self,Y,X):
        #Jeffries and Zellner's g prior        
        
        if not X.size:
            X1=np.ones((Y.shape[0],1))
        else:
            X1=np.insert(X,0,1,axis=1)
        n=len(Y)
        p=np.shape(X1)[1]
        g=np.min([1/n, np.sqrt(p/n)])
        X1t=np.transpose(X1)
        #print(np.matmul(np.matmul(X1t,X1),X1t))
        beta=np.matmul(np.matmul(np.linalg.inv(np.matmul(X1t,X1)),X1t),Y)
        
        Y1=np.matmul(X1,beta);
        sse=np.sum(np.power((Y-Y1),2))
        sse_nm=np.sum(np.power((Y-np.mean(Y)),2))
        #V=(  (1/(1+g))*sse + (g/(1+g))*sse_nm  )**(-(n-1)/2)
        #e=((g/(1+g))**(p/2))*V        
        logV=(-(n-1)/2)*np.log( (1/(1+g))*sse + (g/(1+g))*sse_nm)
        loge= (p/2)*np.log((g/(1+g)))+logV
        e=-loge
        #print(((1/(1+g))*sse+(g/(1+g))*sse_nm  )**(-(n-1)/2))
        beta=beta*(1+g) # g=argmin(1/n , )
        return e,beta
        
        

    
       
    def model_search(self,Y,X):
        #Y= output variables
        #X=Feature matrix
        #K=select TOP K models at each stage
        #F= feature type flag, true=categorical, false=continuous
        ##K2 = minimum number of nearest neighbour
        #T=type of error, T=SSQ,HR
    
         
        ALL_M=[];#list that contains all models
        ALL_E=[];#list that contains the associated LOO-cross validation error
        ALL_P=[];
        #NV=X.shape[0]# Number of kernels in L
        #print(np.shape(L))
        ERRTH=self.error_function(Y)
        #print("Null model error: ",ERRTH)
        # Add the null model to the list of all models
        ALL_M.append([])
        ALL_E.append(ERRTH)
        #if self.params['ALGO']=="SSE_BIC":
        ALL_P.append(np.mean(Y))
        
        
        
        V=list(X) #np.arange(0,NV,1).astype(int) #  all possible variables
        #print all features
        #print('Listing all features....')
        #print(V)
        
        CURR_M=[]; # empty current model
        #t=time.time()
        #(SGM,ERR,MINERR)=self.grow_by_one(CURR_M,V,ERRTH,Y,X)
        AGM=self.grow_by_one(CURR_M,V,ERRTH,Y,X)#AGM=ALL_GROWN_MODELS
        #print ('Took ', time.time()-t, ' seconds to grow all first order models')
        ERRTH=AGM['MINERR'] #Change the error threshold
        #print(np.shape(SGM))
        
        FLAG=0 # Flag for stopping the algorithm
        #print(np.size(ERR))
        if np.size(AGM['SGM'])>0:   #If simgle variable models that are better than the null model are found
            V=[mv for l in AGM['SGM'] for mv in l ] #** Include only those variables which are better than the null model
            #print(V)
            #store the single variable models
            #print(len(AGM['SGM']),AGM['ERR'])
            for k in range(len(AGM['SGM'])):
                ALL_M.append(AGM['SGM'][k]) #Store all the one variable models
                ALL_E.append(AGM['ERR'][k]) #Store all models errors 
                #if self.params['ALGO']=="SSE_BIC":
                ALL_P.append(AGM['BETA'][k])
                    
                    
            
            K1=np.min([self.params['TOP_K'],np.size(AGM['ERR'])]); # find minimum between K and the number of models
            IERR=np.argsort(AGM['ERR'],axis=0)[:K1]# get indexes of K1 models with the lowest errors
            #print(IERR)
            #CURR_M=SGM[IERR] #Select top K1 models for further growth
            CURR_M=[AGM['SGM'][i] for i in IERR]
            V=[V[i] for i in IERR] # Only these many variables will be selected from now on
            #print(np.size(CURR_M))
            #print(V)
            while FLAG==0:
                V1=V # Make a new copy of the indexes of available variables
                MINERR1=10**50 # will be used to store min-error of the k variable models
                i=0 #index of the current model
                MC=0# count number of the grown models
                #N_rows=np.shape(CURR_M)[0]*np.size(V1)#Number of maximum number of models possible in the next round
                #N_cols=np.shape(CURR_M)[1]+1 # Size of the next round of models
                
                #SGM_NEXT=np.zeros((N_rows,N_cols)) #Pre-allocate memory to store the next order models
                #ERR_NEXT=np.zeros(N_rows) # Pre-allocate memory to store the errors of the next set of models
                #print(ERRTH)
                SGM_NEXT=[] # stores the next set of models dynamically
                ERR_NEXT=[] # stores the errors dynamically
                P_NEXT=[] # store model paramters dynamically
                for M in CURR_M: #for each model in the set of current models
                    #print(V1)
                    #print(M)
                    V1=np.setdiff1d(V1,M)
                    #print('Current model is: ',M)
                    #print('Current model is: ',V)
                    #print(V1)
                    if len(V1)>0: # If V1 is non empty                            
                        AGM=self.grow_by_one(M,V1,ERRTH,Y,X)#ALL GROWN MODELS
                        if np.size(AGM['SGM'])>0:# If the grown model set is not empty
                            N_MOD=np.shape(AGM['SGM'])[0] # Number of new models
                            #print(np.shape(SGM))
                            #print(np.shape(SGM_NEXT))
                            for k in range(len(AGM['SGM'])):
                                SGM_NEXT.append(AGM['SGM'][k])# Store the grown models
                                ERR_NEXT.append(AGM['ERR'][k])# Store the error of the grown models                               
                                #if self.params['ALGO']=="SSE_BIC":
                                P_NEXT.append(AGM['BETA'][k])
                            MC=MC+N_MOD # Increase the total model count
                            if AGM['MINERR']<MINERR1:
                                MINERR1=AGM['MINERR']
                        i=i+1 # Increase the index of the current model
                #Out of for loop
                if MC==0: #No model has been grown
                    FLAG=1 # quit
                else: #Some models had been grown 
                    #SGM_NEXT=SGM_NEXT[0:MC:1,:] # Remove the unused storage spaces
                    #ERR_NEXT=ERR_NEXT[0:MC:1] # Remove the unused storage spaces
                   
                    for k in range(len(SGM_NEXT)):
                        ALL_M.append(SGM_NEXT[k]) # Store the grown models 
                        ALL_E.append(ERR_NEXT[k]) # Store  the model errors
                        #if self.params['ALGO']=="SSE_BIC":
                        ALL_P.append(P_NEXT[k]) # Store model paramters
                   
                    K1=np.min([self.params['TOP_K'],MC]) # minimum between total number of grown model and K
                    IERR=np.argsort(ERR_NEXT,axis=0)[:K1] # Take top K1 models for further growing
                    #IERR=np.reshape(IERR,(np.))
                    #CURR_M=SGM_NEXT[IERR] #Assign top K1 models to current models
                    CURR_M=[SGM_NEXT[i] for i in IERR]#Assign top K1 models to current models
                    #print(np.shape(IERR))
                    ERRTH=MINERR1 # Assign error threshold to the current minimum error (so that at the stage we only select models with lower error)
                #print("current model order: ",np.shape(SGM_NEXT)[1])
                 # end of while
        MODELS={'ALL_MODELS':ALL_M,'ALL_ERRS':ALL_E}
        #if self.params['ALGO']=="SSE_BIC":
        MODELS['BETA']=ALL_P
        return MODELS #Return model its paramteres and the errors associated with it       

    #calculate the error
    def error_function(self,Y):
        #Y=observed data
        #Y1=predicted data
        rho=0
        X=np.array([])
        Y=Y.values.reshape(len(Y),1)
        if self.params['ALGO']=="SSE_BIC":            
            rho,beta=self.mse_bic(Y,X)
        elif self.params['ALGO']=="SSE_AIC":            
            rho,beta=self.mse_aic(Y,X)
        elif self.params['ALGO']=="BAYES":            
            rho,beta=self.bayesian_reg(Y,X)
        elif self.params['ALGO']=="LASSO":            
            rho,beta=self.lasso_reg(Y,X)
        elif self.params['ALGO']=="RIDGE":            
            rho,beta=self.ridge_reg(Y,X)
        elif self.params['ALGO']=="ELASTIC_NET":
            rho,beta=self.elastic_net_reg(Y,X)
            
            
        return (rho) #I1=index of non-zero element in Y1
    
    def predict_lin_reg(self,M,beta,D):
        #M=model
        #Beta=regression coeffs
        # D=data
        X=D[M].values # Extract data into a matrix
        if not X.size: #if X is empty return the coefficient
            return beta
        else:
            X1=np.insert(X,0,1,axis=1)  # insert 1s to account for interceptors      
            Y1=np.matmul(X1,beta); # Pretict Y
            return Y1
    
    def predict_avg_lin_reg(self,ALL_M,ALL_E,ALL_B,D,TOP_K):
        #ALL_M= ALL_Models
        #ALL_E=ERRORS ASSOCIATED with the models
        #ALL_B = betas associated with the models
        #D = test data
        #TOP_K=top k model
        if TOP_K<1 or TOP_K>len(ALL_M):
            TOP_K=len(ALL_M) # 
        I=np.argsort(ALL_E)
        Y=[self.predict_lin_reg(ALL_M[I[i]],ALL_B[I[i]],D) for i in range(TOP_K)]
        #E=[np.sum((Y-Y[:i,])**2) for i in range(TOP_K)]
        # create a dictionary for the best model
        variables=ALL_M[I[0]] # predictors of top mode
        betas=ALL_B[I[0]] # coefficients for predictors of top model
        if hasattr(betas, "__len__"):
            top_model={'intercept':betas[0]} # first coefficient is the intercept
            if len(betas)>1:
                for i in range(1,len(betas)):
                    top_model[variables[i-1]]=betas[i]
        else:
            top_model={'intercept':betas}
            
        #print(top_model)
        return (np.mean(Y,axis=0),np.std(Y,axis=0),top_model)
    
        
        
        
        
        
        

        
        
    def optimal_model(self,X,ALL_M):
        #print(X)
        I=X.notnull()# find out features which do not have missing values
        #print(I.shape)
        OPT_M=[]# Optimal model
        for M in ALL_M:
            IM=I[M]# Missing status of Model M
            #print(IM,IM.shape)
            OPT_M=M
            FLAG=True
            for m in M:
                #print(IM[m])
                FLAG=FLAG & IM[m]
            #print('Model: ',M,' Flag: ',FLAG)
            if FLAG:     
                break  # break at the first model
        return(OPT_M)
    
    def get_nonnull(self,X,Y,M):
        # X=data
        # M=model
        I=X.notnull() # not null
        I1=np.ones(np.shape(I)[0],dtype=bool) # initialize boolean index
       #print(GM)
        #GM=list(X)# 
        for gmv in M:
            I1=I1&I[gmv] # take only non null data
        #print(np.sum(I1))
        if len(X.shape)==1:
            return (X.loc[I1].values,Y.loc[I1].values)
        else:
            return (X.loc[I1,:].values,Y.loc[I1].values)
        
    
        


def CV_indexes(I,k):
    #I=indexes
    #k=k fold cross validation
    DN=np.ceil(len(I)/k)
    L=[]
    for i in range(k):
        start=int(i*DN)
        end=int(np.min([(i*DN)+DN,len(I)]))
        L.append(I[start:end:1])
    return L

def cross_validate(df,target,params,k): # calculates cross validation errors
    #df=data frame
    #target = target variable
    #params=paramters
    #k= k fold cross validation
    t=time.time()
    r=regression_with_variable_selection(params)
    feature_names=list(set(list(df))-set([target]))
    L=CV_indexes(df.index,k)
    E=0
    #count=1
    MS=[] # list of top models for each cross-val data
    for l in L:
        #print(l)
        L1=list(set(df.index)-set(l))
        M=r.model_search(df.loc[L1,target],df.loc[L1,feature_names])
        YM,YS,top_model=r.predict_avg_lin_reg(M['ALL_MODELS'],M['ALL_ERRS'],M['BETA'],df.loc[l,feature_names],1)# get the best model
        sse=np.sum((df.loc[l,target].values-YM)**2) # get model error
        E=E+sse #sum of squared error on test data
        #print(top_model)
        MS.append(top_model)
    print('Took ', time.time()-t,' seconds to do cross validation')
    return E,MS

def grid_search(algo,parameters,df,target,k,TOP_K_MODELS):
    #algo=name of the algorithm
    #paramters = names of the parameters
    # df=data frame
    # target = target variable
    #k= k-fold cross validation
    # TOP_K_MODELS = grow only Top K models
    E=[] # list of cross validation errors for each paramter value
    MODELS=[] # list of top models for each paramter value
    beta=[]
    params={'ALGO':algo, 'TOP_K':TOP_K_MODELS}
    for p in parameters:
        params[p]=0.01
    if params['ALGO']=="ELASTIC_NET":
        params['alpha']=1
        beta = np.arange(0.1,0.9,0.1)        
        for b in beta:
            params['L1_RATIO']=b
            E1,M1=cross_validate(df,target,params,k) #overall cross validation error and top models
            E.append(E1)
            MODELS.append(M1)
    elif params['ALGO']=="LASSO" or params['ALGO']=="RIDGE":
        beta=np.arange(0.1,1,0.1) # params b
        for a in beta:
            params['alpha']=a
            E1,M1=cross_validate(df,target,params,k) #overall cross validation error and top models
            E.append(E1)
            MODELS.append(M1)
    return beta, MODELS, E
            
            
    
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
    return df3
        
        
    #df2=pd.concat([df, df1.shift(), df1.shift(2)],axis=1)
    #return df1

def add_interactions(df,features):
    #create separate columns containing interactions between pairs of features in the dataset
    for f1 in features:
        for f2 in features:
            if f1 != f2:
                n_f=f1+"_x_"+f2
                df[n_f]=df[f1]*df[f2]
    return df
                
                
    
    
    
        
                

#Callback class
class CallBack(object):
    completed = defaultdict(int)

    def __init__(self, index, parallel):
        self.index = index
        self.parallel = parallel

    def __call__(self, index):
        CallBack.completed[self.parallel] += 1
        print("done with {}".format(CallBack.completed[self.parallel]))
        if self.parallel._original_iterable:
            self.parallel.dispatch_next()


            
            
##possible algo ALGO=KERNEL_REG, MSE
#D=load_diabetes()
#df=pd.DataFrame(data=D.data,columns=D['feature_names']);
#df['target']=D['target']
##df.to_csv('diabetes.csv')
##df=pd.read_csv(sys.argv[0])
#feature_names=D['feature_names']#list(df)
#
##standardize data
#df=(df-df.mean())/df.std()
#########################################################################
#df1=add_lag(df,feature_names,[1,3,5])
##df2=add_interactions(df1,list(df1))
#df2=df1.dropna()
########################################################################
#params={'ALGO':"SSE_AIC",'TOP_K':5,'alpha':0.1,'L1_RATIO':0.5}
#E=[];
#E.append(cross_validate(df2,'target',params,4))

#params={'ALGO':"LASSO",'TOP_K':10,'alpha':0.1,'L1_RATIO':0.5}
#E.append(cross_validate(df,'target',params,10))
#params={'ALGO':"RIDGE",'TOP_K':10,'alpha':0.1,'L1_RATIO':0.5}
#E.append(cross_validate(df,'target',params,10))
#
#params={'ALGO':"BAYES",'TOP_K':10,'alpha':0.1,'L1_RATIO':0.5}
#E.append(cross_validate(df,'target',params,10))
#
#params={'ALGO':"ELASTIC_NET",'TOP_K':10,'alpha':0.1,'L1_RATIO':0.5}
#E.append(cross_validate(df,'target',params,10))
#########################################################################


