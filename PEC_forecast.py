#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:33:42 2018

@author: pedrocampelo
"""

 
#Bibliotecas Usuais.

import numpy as np
import pandas as pd
from pandas import Series
from pandas import read_csv 
from pandas import DataFrame
from pandas import concat
from pandas.plotting import autocorrelation_plot
import os
import time
import datetime as dt
import re
import matplotlib.pyplot as plt

#Modelos
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss


#import pyramid
#from pyramid.arima import auto_arima


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_absolute_percentage_error
# from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lars  
from sklearn.linear_model import LarsCV  
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


os.chdir('C:/Users/Pichau/Desktop/Work/UnB/Tese/Dados')
cwd = os.getcwd()
cwd


def bootstrap_sample(data, B, w):
    '''
    Bootstrap the input data
    data: input numpy data array
    B: boostrap size
    w: block length of the boostrap
    '''
    t = len(data)
    p = 1 / w
    indices = np.zeros((t, B), dtype=int)
    indices[0, :] = np.ceil(t * rand(1, B))
    select = np.asfortranarray(rand(B, t).T < p)
    vals = np.ceil(rand(1, np.sum(np.sum(select))) * t).astype(int)
    indices_flat = indices.ravel(order="F")
    indices_flat[select.ravel(order="F")] = vals.ravel()
    indices = indices_flat.reshape([B, t]).T
    for i in range(1, t):
        indices[i, ~select[i, :]] = indices[i - 1, ~select[i, :]] + 1
    indices[indices > t] = indices[indices > t] - t
    indices -= 1
    return data[indices]


def compute_dij(losses, bsdata):
    '''Compute the loss difference'''
    t, M0 = losses.shape
    B = bsdata.shape[1]
    dijbar = np.zeros((M0, M0))
    for j in range(M0):
        dijbar[j, :] = np.mean(losses - losses[:, [j]], axis=0)

    dijbarstar = np.zeros((B, M0, M0))
    for b in range(B):
        meanworkdata = np.mean(losses[bsdata[:, b], :], axis=0)
        for j in range(M0):
            dijbarstar[b, j, :] = meanworkdata - meanworkdata[j]

    vardijbar = np.mean((dijbarstar - np.expand_dims(dijbar, 0)) ** 2, axis=0)
    vardijbar += np.eye(M0)

    return dijbar, dijbarstar, vardijbar


def calculate_PvalR(z, included, zdata0):
    '''Calculate the p-value of relative algorithm'''
    empdistTR = np.max(np.max(np.abs(z), 2), 1)
    zdata = zdata0[ix_(included - 1, included - 1)]
    TR = np.max(zdata)
    pval = np.mean(empdistTR > TR)
    return pval


def calculate_PvalSQ(z, included, zdata0):
    '''Calculate the p-value of sequential algorithm'''
    empdistTSQ = np.sum(z ** 2, axis=1).sum(axis=1) / 2
    zdata = zdata0[ix_(included - 1, included - 1)]
    TSQ = np.sum(zdata ** 2) / 2
    pval = np.mean(empdistTSQ > TSQ)
    return pval


def iterate(dijbar, dijbarstar, vardijbar, alpha, algorithm="R"):
    '''Iteratively excluding inferior model'''
    B, M0, _ = dijbarstar.shape
    z0 = (dijbarstar - np.expand_dims(dijbar, 0)) / np.sqrt(
        np.expand_dims(vardijbar, 0)
    )
    zdata0 = dijbar / np.sqrt(vardijbar)

    excludedR = np.zeros([M0, 1], dtype=int)
    pvalsR = np.ones([M0, 1])

    for i in range(M0 - 1):
        included = np.setdiff1d(np.arange(1, M0 + 1), excludedR)
        m = len(included)
        z = z0[ix_(range(B), included - 1, included - 1)]

        if algorithm == "R":
            pvalsR[i] = calculate_PvalR(z, included, zdata0)
        elif algorithm == "SQ":
            pvalsR[i] = calculate_PvalSQ(z, included, zdata0)

        scale = m / (m - 1)
        dibar = np.mean(dijbar[ix_(included - 1, included - 1)], 0) * scale
        dibstar = np.mean(dijbarstar[ix_(range(B), included - 1, included - 1)], 1) * (
            m / (m - 1)
        )
        vardi = np.mean((dibstar - dibar) ** 2, axis=0)
        t = dibar / np.sqrt(vardi)
        modeltoremove = np.argmax(t)
        excludedR[i] = included[modeltoremove]

    maxpval = pvalsR[0]
    for i in range(1, M0):
        if pvalsR[i] < maxpval:
            pvalsR[i] = maxpval
        else:
            maxpval = pvalsR[i]

    excludedR[-1] = np.setdiff1d(np.arange(1, M0 + 1), excludedR)
    pl = np.argmax(pvalsR > alpha)
    includedR = excludedR[pl:]
    excludedR = excludedR[:pl]
    return includedR - 1, excludedR - 1, pvalsR


def MCS(losses, alpha, B, w, algorithm):
    '''Main function of the MCS'''
    t, M0 = losses.shape
    bsdata = bootstrap_sample(np.arange(t), B, w)
    dijbar, dijbarstar, vardijbar = compute_dij(losses, bsdata)
    includedR, excludedR, pvalsR = iterate(
        dijbar, dijbarstar, vardijbar, alpha, algorithm=algorithm
    )
    return includedR, excludedR, pvalsR


class ModelConfidenceSet(object):
    def __init__(self, data, alpha, B, w, algorithm="SQ", names=None):
        """
        Implementation of Econometrica Paper:
        Hansen, Peter R., Asger Lunde, and James M. Nason. "The model confidence set." Econometrica 79.2 (2011): 453-497.

        Input:
            data->pandas.DataFrame or numpy.ndarray: input data, columns are the losses of each model 
            alpha->float: confidence level
            B->int: bootstrap size for computation covariance
            w->int: block size for bootstrap sampling
            algorithm->str: SQ or R, SQ is the first t-statistics in Hansen (2011) p.465, and R is the second t-statistics
            names->list: the name of each model (corresponding to each columns). 

        Method:
            run(self): compute the MCS procedure

        Attributes:
            included: models that are in the model confidence sets at confidence level of alpha
            excluded: models that are NOT in the model confidence sets at confidence level of alpha
            pvalues: the bootstrap p-values of each models
        """

        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.names = data.columns.values if names is None else names
        elif isinstance(data, np.ndarray):
            self.data = data
            self.names = np.arange(data.shape[1]) if names is None else names

        if alpha < 0 or alpha > 1:
            raise ValueError(
                f"alpha must be larger than zero and less than 1, found {alpha}"
            )
        if not isinstance(B, int):
            try:
                B = int(B)
            except Exception as identifier:
                raise RuntimeError(
                    f"Bootstrap size B must be a integer, fail to convert", identifier
                )
        if B < 1:
            raise ValueError(f"Bootstrap size B must be larger than 1, found {B}")
        if not isinstance(w, int):
            try:
                w = int(w)
            except Exception as identifier:
                raise RuntimeError(
                    f"Bootstrap block size w must be a integer, fail to convert",
                    identifier,
                )
        if w < 1:
            raise ValueError(f"Bootstrap block size w must be larger than 1, found {w}")

        if algorithm not in ["R", "SQ"]:
            raise TypeError(f"Only R and SQ algorithm supported, found {algorithm}")

        self.alpha = alpha
        self.B = B
        self.w = w
        self.algorithm = algorithm

    def run(self):
        included, excluded, pvals = MCS(
            self.data, self.alpha, self.B, self.w, self.algorithm
        )

        self.included = self.names[included].ravel().tolist()
        self.excluded = self.names[excluded].ravel().tolist()
        self.pvalues = pd.Series(pvals.ravel(), index=self.excluded + self.included)
        return self


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def create_database():
    
    periodo = pd.date_range('2/1/2017', periods=546)
    treino = pd.date_range('2/1/2017', periods=365)
    teste = pd.date_range('2/1/2018', periods=181)
    
    consumo = read_csv('dadosconsumo.csv', sep=';',header=0, parse_dates=[0], index_col='Data', squeeze=True)
    variaveis = read_csv('dadosx.csv', sep=';',header=0, parse_dates=[0], index_col='Data', squeeze=True, encoding = "ISO-8859-1")


    colunas =['CEE_SECO_01','CEE_SECO_02','CEE_SECO_03','CEE_SECO_04','CEE_SECO_05','CEE_SECO_06','CEE_SECO_07','CEE_SECO_08','CEE_SECO_09','CEE_SECO_10','CEE_SECO_11','CEE_SECO_12', 
            'CEE_SECO_13','CEE_SECO_14','CEE_SECO_15','CEE_SECO_16','CEE_SECO_17','CEE_SECO_18','CEE_SECO_19','CEE_SECO_20','CEE_SECO_21','CEE_SECO_22','CEE_SECO_23','CEE_SECO_24', 
            'CEE_SECO_MED','CEE_SECO_CSO','CEE_SECO_CP','CEE_SECO_IP','CEE_SECO_IND','CEE_SECO_PP','CEE_SECO_RED','CEE_SECO_RR','CEE_SECO_RRA','CEE_SECO_RRI','CEE_SECO_SP1','CEE_SECO_SP2', 
            'CEE_SECO_TOT','CEE_SUL_01','CEE_SUL_02','CEE_SUL_03','CEE_SUL_04','CEE_SUL_05','CEE_SUL_06','CEE_SUL_07','CEE_SUL_08','CEE_SUL_09','CEE_SUL_10','CEE_SUL_11', 
            'CEE_SUL_12','CEE_SUL_13','CEE_SUL_14','CEE_SUL_15','CEE_SUL_16','CEE_SUL_17','CEE_SUL_18','CEE_SUL_19','CEE_SUL_20','CEE_SUL_21','CEE_SUL_22','CEE_SUL_23','CEE_SUL_24' ,
            'CEE_SUL_MED','CEE_SUL_CSO','CEE_SUL_CP','CEE_SUL_IP','CEE_SUL_IND','CEE_SUL_PP','CEE_SUL_RED','CEE_SUL_RR','CEE_SUL_RRA','CEE_SUL_RRI','CEE_SUL_SP1','CEE_SUL_SP2','CEE_SUL_TOT', 
            'CEE_NE_01','CEE_NE_02','CEE_NE_03','CEE_NE_04','CEE_NE_05','CEE_NE_06','CEE_NE_07','CEE_NE_08','CEE_NE_09','CEE_NE_10','CEE_NE_11','CEE_NE_12','CEE_NE_13', 
            'CEE_NE_14','CEE_NE_15','CEE_NE_16','CEE_NE_17','CEE_NE_18','CEE_NE_19','CEE_NE_20','CEE_NE_21','CEE_NE_22','CEE_NE_23','CEE_NE_24','CEE_NE_MED','CEE_NE_CSO', 
            'CEE_NE_CP','CEE_NE_IP','CEE_NE_IND','CEE_NE_PP','CEE_NE_RED','CEE_NE_RR','CEE_NE_RRA','CEE_NE_RRI','CEE_NE_SP1','CEE_NE_SP2','CEE_NE_TOT','CEE_N_01','CEE_N_02', 
            'CEE_N_03','CEE_N_04','CEE_N_05','CEE_N_06','CEE_N_07','CEE_N_08','CEE_N_09','CEE_N_10','CEE_N_11','CEE_N_12','CEE_N_13','CEE_N_14','CEE_N_15',
            'CEE_N_16','CEE_N_17','CEE_N_18','CEE_N_19','CEE_N_20','CEE_N_21','CEE_N_22','CEE_N_23','CEE_N_24','CEE_N_MED','CEE_N_CSO','CEE_N_CP','CEE_N_IP', 
            'CEE_N_IND','CEE_N_PP','CEE_N_RED','CEE_N_RR','CEE_N_RRA','CEE_N_RRI','CEE_N_SP1','CEE_N_SP2','CEE_N_TOT','CEE_BR_01','CEE_BR_02','CEE_BR_03','CEE_BR_04',
            'CEE_BR_05','CEE_BR_06','CEE_BR_07','CEE_BR_08','CEE_BR_09','CEE_BR_10','CEE_BR_11','CEE_BR_12','CEE_BR_13','CEE_BR_14','CEE_BR_15','CEE_BR_16','CEE_BR_17' ,
            'CEE_BR_18','CEE_BR_19','CEE_BR_20','CEE_BR_21','CEE_BR_22','CEE_BR_23','CEE_BR_24','CEE_BR_MED','CEE_BR_CSO','CEE_BR_CP','CEE_BR_IP','CEE_BR_IND','CEE_BR_PP', 
            'CEE_BR_RED','CEE_BR_RR','CEE_BR_RRA','CEE_BR_RRI','CEE_BR_SP1','CEE_BR_SP2','CEE_BR_TOT','DM','DS','MÊS','ANO','ESTAC','FER','NEB_SECO_9','NEB_SECO_15' , 
            'NEB_SECO_21','NEB_SECO_MED','PA_SECO_9','PA_SECO_15','PA_SECO_21','PA_SECO_MED','TEMP_BS_SECO_9','TEMP_BS_SECO_15','TEMP_BS_SECO_21','TEMP_BS_SECO_MED','TEMP_BU_SECO_9','TEMP_BU_SECO_15',
            'TEMP_BU_SECO_21','TEMP_BU_SECO_MED','UMID_SECO_9','UMID_SECO_15','UMID_SECO_21','UMID_SECO_MED','DV_SECO_9','DV_SECO_15','DV_SECO_21','DV_SECO_MED','VV_SECO_9','VV_SECO_15','VV_SECO_21', 
            'VV_SECO_MED','NEB_SUL_9','NEB_SUL_15','NEB_SUL_21','NEB_SUL_MED','PA_SUL_9','PA_SUL_15','PA_SUL_21','PA_SUL_MED','TEMP_BS_SUL_9','TEMP_BS_SUL_15','TEMP_BS_SUL_21','TEMP_BS_SUL_MED', 
            'TEMP_BU_SUL_9','TEMP_BU_SUL_15','TEMP_BU_SUL_21','TEMP_BU_SUL_MED','UMID_SUL_9','UMID_SUL_15','UMID_SUL_21','UMID_SUL_MED','DV_SUL_9','DV_SUL_15','DV_SUL_21','DV_SUL_MED','VV_SUL_9',
            'VV_SUL_15','VV_SUL_21','VV_SUL_MED','NEB_NE_9','NEB_NE_15','NEB_NE_21','NEB_NE_MED','PA_NE_9','PA_NE_15','PA_NE_21','PA_NE_MED','TEMP_BS_NE_9','TEMP_BS_NE_15','TEMP_BS_NE_21','TEMP_BS_NE_MED',
            'TEMP_BU_NE_9','TEMP_BU_NE_15','TEMP_BU_NE_21','TEMP_BU_NE_MED','UMID_NE_9','UMID_NE_15','UMID_NE_21','UMID_NE_MED','DV_NE_9','DV_NE_15','DV_NE_21','DV_NE_MED','VV_NE_9','VV_NE_15','VV_NE_21', 
            'VV_NE_MED','NEB_N_9','NEB_N_15','NEB_N_21','NEB_N_MED','PA_N_9','PA_N_15','PA_N_21','PA_N_MED','TEMP_BS_N_9','TEMP_BS_N_15','TEMP_BS_N_21','TEMP_BS_N_MED','TEMP_BU_N_9','TEMP_BU_N_15','TEMP_BU_N_21', 
            'TEMP_BU_N_MED','UMID_N_9','UMID_N_15','UMID_N_21','UMID_N_MED','DV_N_9','DV_N_15','DV_N_21','DV_N_MED','VV_N_9','VV_N_15','VV_N_21','VV_N_MED','NEB_BR_9','NEB_BR_15','NEB_BR_21',
            'NEB_BR_MED','PA_BR_9','PA_BR_15','PA_BR_21','PA_BR_MED','TEMP_BS_BR_9','TEMP_BS_BR_15','TEMP_BS_BR_21','TEMP_BS_BR_MED','TEMP_BU_BR_9','TEMP_BU_BR_15','TEMP_BU_BR_21','TEMP_BU_BR_MED','UMID_BR_9', 
            'UMID_BR_15','UMID_BR_21','UMID_BR_MED','DV_BR_9','DV_BR_15','DV_BR_21','DV_BR_MED','VV_BR_9','VV_BR_15','VV_BR_21','VV_BR_MED','TAR_SECO_CSO','TAR_SECO_CP','TAR_SECO_IP','TAR_SECO_IND',
            'TAR_SECO_PP','TAR_SECO_RED','TAR_SECO_RR','TAR_SECO_RRA','TAR_SECO_RRI','TAR_SECO_SP1','TAR_SECO_SP2','TAR_SECO_MED','TAR_SUL_CSO','TAR_SUL_CP','TAR_SUL_IP','TAR_SUL_IND','TAR_SUL_PP','TAR_SUL_RED', 
            'TAR_SUL_RR','TAR_SUL_RRA','TAR_SUL_RRI','TAR_SUL_SP1','TAR_SUL_SP2','TAR_SUL_MED','TAR_NE_CSO','TAR_NE_CP','TAR_NE_IP','TAR_NE_IND','TAR_NE_PP','TAR_NE_RED','TAR_NE_RR','TAR_NE_RRA', 
            'TAR_NE_RRI','TAR_NE_SP1','TAR_NE_SP2','TAR_NE_MED','TAR_N_CSO','TAR_N_CP','TAR_N_IP','TAR_N_IND','TAR_N_PP','TAR_N_RED','TAR_N_RR','TAR_N_RRA','TAR_N_RRI','TAR_N_SP1','TAR_N_SP2',
            'TAR_N_MED','TAR_BR_CSO','TAR_BR_CP','TAR_BR_IP','TAR_BR_IND','TAR_BR_PP','TAR_BR_RED','TAR_BR_RR','TAR_BR_RRA','TAR_BR_RRI','TAR_BR_SP1','TAR_BR_SP2','TAR_BR_MED','Meta_Selic', 
            'Taxa_Selic','CDI','DolarC','DolarC_var','DolarV','DolarV_var','EuroC','EuroC_var','EuroV','EuroV_var','IBV_Cot','IBV_min','IBV_max','IBV_varabs','IBV_varperc','IBV_vol','INPC_m',
            'INPC_ac','IPCA_m','IPCA_ac','IPAM_m','IPAM_ac','IPADI_m', 'IPADI_ac' , 'IGPM_m','IGPM_ac','IGPDI_m','IGPDI_ac','PAB_o','PAB_d','TVP_o','TVP_d','PICV_o','ICV_d','CCU_o',
            'CCU_d','CS_o','CS_d','UCPIIT_FGV_o','UCPIIT_FGV_d','CPCIIT_CNI_o','CPCIIT_CNI_d','VIR_o','VIR_d','HTPIT_o','HTPIT_d','SRIT_o','SRIT_d','PPOB','PGN','PIG_o','PIG_d','PIBCa_o','PIBCa_d',
            'PIBI_o','PIBI_d','PIBCo_o','PIBCo_d','PIA_o','PIA_d','ICC','INEC','ICEI','DBNDES','IEG_o','IEG_d','IETIT_o','IETIT_d','IETC_o','IETC_d','IETS_o','IETS_d','IETCV_o','IETCV_d', 'PO','TD','BM','PME']

     
    
    dados = pd.concat([consumo, variaveis], axis=1)
    dados.columns=colunas
    dados.index=periodo
    
    return periodo, treino, teste, dados    


def set_database(forecastHorizon, dados, periodo, train_ratio):
     
    y = dados['CEE_BR_TOT']
    colunas=dados.columns
    X = pd.DataFrame(index=periodo)
    
    
    for i in colunas:
        X[i+'_lag1']=dados[i].shift(round(forecastHorizon))
        
    for i in colunas:
        X[i+'_lag2']=dados[i].shift(round(forecastHorizon*2))
        
    for i in colunas:
        X[i+'_lag3']=dados[i].shift(round(forecastHorizon*3))
 
        
    # for i in colunas:
    #     X[r'{}_lag_{}'.format(i, str(forecastHorizon*1))]=dados[i].shift(round(forecastHorizon))
        
    # for i in colunas:
    #     X[r'{}_lag_{}'.format(i, str(forecastHorizon*2))]=dados[i].shift(round(forecastHorizon))
        
    # for i in colunas:
    #     X[r'{}_lag_{}'.format(i, str(forecastHorizon*3))]=dados[i].shift(round(forecastHorizon))
         
    lenx=len(X)
    X=X.dropna()
    X['constante']=1
    lenx1=len(X)
    dif_len=lenx-lenx1
    y=y[dif_len:]
       
    
    n = len(X)*(1-train_ratio)
    
    treino= periodo[:-round(n)]
    teste=periodo[-round(n):]
    
    
    #Dividindo em teste e treino
    # train_size = int(len(X) * ((len(X)-90)/len(X)))
    train_size = round(len(X)*train_ratio)
    X_treino, X_teste = X[0:train_size], X[train_size:len(X)]
    y_treino, y_teste = y[0:train_size], y[train_size:len(y)]
     
    print('Observations: %d' % (len(X)))
    print('Training Observations: %d' % (len(X_treino)))
    print('Testing Observations: %d' % (len(X_teste)))
    
    

    return y,X, X_treino, X_teste, y_treino, y_teste, treino, teste


def set_adjusted_temperature(X_treino, X_teste):
    

    for col in ['TEMP_BS', 'TEMP_BU' ]:
        for reg in ['SECO', 'SUL','NE','N']:
            for hour in ['9', '15', '21', 'MED']:
                for lag in ['lag1', 'lag2','lag3']:
                
                    X_treino[col.split('_')[0]+'aj_'+col.split('_')[1] + '_'+reg+'_'+hour+'_'+lag]= X_treino[col+'_'+reg+'_'+hour+'_'+lag] - X_treino[col+'_'+reg+'_'+hour+'_'+lag].mean()
                    X_teste[col.split('_')[0]+'aj_'+col.split('_')[1] + '_'+reg+'_'+hour+'_'+lag]= X_teste[col+'_'+reg+'_'+hour+'_'+lag] - X_teste[col+'_'+reg+'_'+hour+'_'+lag].mean()
                
                    print(col+'_'+reg+'_'+hour+'_'+lag)
    
    return X_treino, X_teste


def correlations_plots(dados, y):
    
    #Correlação        
    values = DataFrame(dados['CEE_BR_TOT'].values)
    dataframe = concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t-1','t+1']
    result = dataframe.corr()
    print(result)
       
    dec_seas = seasonal_decompose(y, model='multiplicative')
    fig = dec_seas.plot()
    plt.savefig('CEEBRseas.png')   
    
    
    #Autocorrelação PLOT   
    
    #Brasil
    autocorrelation_plot(dados['CEE_BR_TOT'])
    plt.title('Autocorrelation of Power Eletricity Consumption ')
    plt.savefig('CEEBRautocorr.png')   
    plt.show()
    
    #Agregado
    fig, axes = plt.subplots(4,1, sharey=True, sharex=True)
    fig = autocorrelation_plot(dados['CEE_SECO_TOT'], ax=axes[0])
    fig = autocorrelation_plot(dados['CEE_SUL_TOT'], ax=axes[1])
    fig = autocorrelation_plot(dados['CEE_NE_TOT'], ax=axes[2])
    fig = autocorrelation_plot(dados['CEE_N_TOT'], ax=axes[3])
    #fig.tight_layout()
    #plt.savefig('CEEautocorragreg.png')  
    plt.show()
    
    
    #Lag Plot
    plot_acf(dados['CEE_BR_TOT'], lags=40)
    plt.savefig('CEEBRlag1.png')   
    pyplot.show()
    
      
    fig, axes = plt.subplots(4,1, sharey=True, sharex=True)
    fig = sm.graphics.tsa.plot_acf(dados['CEE_SECO_TOT'], lags= 51, ax=axes[0], title='Sudeste/Centro-Oeste')
    fig = sm.graphics.tsa.plot_acf(dados['CEE_SUL_TOT'], lags= 51, ax=axes[1], title='Sul')
    fig = sm.graphics.tsa.plot_acf(dados['CEE_NE_TOT'], lags= 51, ax=axes[2], title= 'Nordeste')
    fig = sm.graphics.tsa.plot_acf(dados['CEE_N_TOT'], lags= 51, ax=axes[3], title='Norte')
    fig.tight_layout()
    plt.savefig('CEElag1agreg.png')   
    plt.show()
    
    return None

    
def adf_test(y):
    # perform Augmented Dickey Fuller test
    print('Results of Augmented Dickey-Fuller test:')
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    print(dfoutput)
       


def first_diff(y_treino, y_teste):
    
    y_treino_diff_aux = np.diff(y_treino)
    print(y_treino[1:].index)
    treino_diff= pd.date_range('3/6/2017', periods=len(y_treino_diff_aux))
    y_treino_diff = pd.DataFrame(y_treino_diff_aux, index=treino_diff)
 
    
    y_teste_diff_aux=np.diff(y_teste)
    print(y_treino[-1:].index)
    teste_diff= pd.date_range('7/2/2018', periods=len(y_teste_diff_aux))
    y_teste_diff = pd.DataFrame(y_teste_diff_aux, index=teste_diff)

    return y_treino_diff, y_teste_diff


def model_plot(model, y_treino, y_teste, y_predictions):
    
    plt.figure()    
    pyplot.plot(y_treino, label='Train')
    pyplot.plot(y_teste, color='black', label='Test')
    pyplot.plot(y_predictions , label='Forecast')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Power Eletricity Consumption Forecast ('+model+')')
#    plt.grid()
#    plt.savefig(model+'_'+str(forecastHorizon)+'lag.png')  
    pyplot.show()
    
    return None


def make_forecast(X,y,train_size, forecast_horizon, models_list, jobs):
 
    ycol = y.to_frame().columns.item()
    y=y.to_frame()

    df_metrics = pd.DataFrame()    
    X_treino, X_teste = X[0:train_size], X[train_size:len(X)]
    y_treino, y_teste = y[0:train_size], y[train_size:len(y)]    
    alpha_list=[1e-20, 1e-15, 1e-10, 1e-5, 1e-2,1e-1, 1, 5, 10, 100]
    
    coef_dict = {}  
    time_df=pd.DataFrame()           
    for modelname in models_list:
        
        print('###############')
        
        time_model_aux = dt.datetime.now()
        
        df_coef2 = pd.DataFrame(index=X.columns)
        df_alpha = pd.DataFrame()
        df_ratio = pd.DataFrame()
        pred_list = []
        data_list = []
        i=0
        while forecast_horizon+i-1<len(X)-train_size:
            
            X_treino2, X_teste2 = X[i:train_size+i], X[train_size+forecast_horizon+i-1:train_size+forecast_horizon+i]
            y_treino2, y_teste2 = y[i:train_size+i], y[train_size+forecast_horizon+i-1:train_size+forecast_horizon+i]
            
            # print(X_treino2.head(1).index.item(),X_treino2.tail(1).index.item() )
            # print(X_teste2.head(1).index.item(),X_teste2.tail(1).index.item() )
            
            if modelname=='LASSO':
            
                print ('Rodando ', modelname)
                model = LassoCV(cv=3,copy_X=True, fit_intercept=True, verbose=True, normalize=True, alphas = alpha_list, n_jobs=jobs)
                model_fit = model.fit(X_treino2, y_treino2)   
    
                data_pred = y_teste2.index.item().strftime('%d%m%Y')
                print('Data prevista é:', data_pred)
                df_coef2[data_pred] = [x for x in model_fit.coef_] 
                df_alpha = df_alpha.append(pd.DataFrame({'alpha':model_fit.alpha_},index=[data_pred]))   
    
                coef_dict[modelname]={'coef':df_coef2, 'alpha':df_alpha}
    
    
    
            elif modelname=='LARS':
                
                print ('Rodando ', modelname)
                model = LarsCV(cv=5,copy_X=True, fit_intercept=True, max_iter=1000, verbose=True, normalize=True, n_jobs=jobs)
                model_fit = model.fit(X_treino2, y_treino2)   
    
                data_pred = y_teste2.index.item().strftime('%d%m%Y')
                print('Data prevista é:', data_pred)
                df_coef2[data_pred] = [x for x in model_fit.coef_] 
                df_alpha = df_alpha.append(pd.DataFrame({'alpha':model_fit.alpha_},index=[data_pred]))   
    
                coef_dict[modelname]={'coef':df_coef2, 'alpha':df_alpha}
    
                
            elif modelname=='LASSOLARS':
      
                print ('Rodando ', modelname)
                model = LassoLarsCV(cv=5,copy_X=True, fit_intercept=True, max_iter=1000, verbose=True, normalize=True,n_jobs=jobs)
                model_fit = model.fit(X_treino2, y_treino2)             
    
                data_pred = y_teste2.index.item().strftime('%d%m%Y')
                print('Data prevista é:', data_pred)
                df_coef2[data_pred] = [x for x in model_fit.coef_] 
                df_alpha = df_alpha.append(pd.DataFrame({'alpha':model_fit.alpha_},index=[data_pred]))   
    
                coef_dict[modelname]={'coef':df_coef2, 'alpha':df_alpha}
                            
    
            elif modelname=='RIDGE':
    
                print ('Rodando ', modelname)
                model = RidgeCV(cv=5,alphas=alpha_list, fit_intercept=True, normalize=True)
                model_fit = model.fit(X_treino2, y_treino2)             
                
                data_pred = y_teste2.index.item().strftime('%d%m%Y')
                print('Data prevista é:', data_pred)
                df_coef2[data_pred] = model_fit.coef_.tolist()[0]
                df_alpha = df_alpha.append(pd.DataFrame({'alpha':model_fit.alpha_},index=[data_pred]))   
    
                coef_dict[modelname]={'coef':df_coef2, 'alpha':df_alpha}
               
                
               
            elif modelname=='EN':    
    
                print ('Rodando ', modelname)
                model = ElasticNetCV(cv=5,copy_X=True, fit_intercept=True, max_iter=1000, verbose=True, normalize=True,n_jobs=jobs,
                                       alphas = alpha_list, l1_ratio = [ratio for ratio in np.arange(0,1,.05)])
                model_fit = model.fit(X_treino2, y_treino2)             
                
                data_pred = y_teste2.index.item().strftime('%d%m%Y')
                print('Data prevista é:', data_pred)
                df_coef2[data_pred] = [x for x in model_fit.coef_] 
                df_alpha = df_alpha.append(pd.DataFrame({'alpha':model_fit.alpha_},index=[data_pred]))   
                df_ratio = df_ratio.append(pd.DataFrame({'l1ration':model_fit.l1_ratio_},index=[data_pred]))   
    
                coef_dict[modelname]={'coef':df_coef2, 'alpha':df_alpha, 'l1ratio':df_ratio}

    
            elif modelname=='OLS':
    
                print ('Rodando ', modelname)
                model = LinearRegression(copy_X=True, fit_intercept=True, normalize=True)
                model_fit = model.fit(X_treino2, y_treino2)             
                data_pred = y_teste2.index.item().strftime('%d%m%Y')
                print('Data prevista é:', data_pred)
                
                df_coef2[data_pred] = model_fit.coef_.tolist()[0]
                # df_alpha = df_alpha.append(pd.DataFrame({'alpha':model_fit.alpha_},index=[data_pred]))   
    
                coef_dict[modelname]={'coef':df_coef2, 'alpha':df_alpha}
            
            
            elif modelname=='ARIMA':
    
                print ('Rodando ', modelname)
                
                # yDiff_df = pd.DataFrame({'y_orig':y_treino2.Milk,
                                          # 'y_diff':[np.nan] + [el for el in np.diff(y_treino2.Milk)]},
                             # index = y_treino2.index) 

                # yDiff_df2 = yDiff_df['y_diff'].copy().dropna()
                # resDiff = sm.tsa.arma_order_select_ic(y_treino2, max_ar=6, max_ma=6, ic='aic')
                

                model_fit = ARIMA(y_treino2, order=(12,1,1)).fit()
                data_pred = y_teste2.index.item().strftime('%d%m%Y')
                print('Data prevista é:', data_pred)                   

                # plt.plot(yDiff_df.index.get_level_values('Data'), yDiff_df.y_diff)
                # plt.plot(a.index.get_level_values('Data'), model_fit.fittedvalues, color='red')
            

            elif modelname=='SVR':
    
                print ('Rodando ', modelname)
                
                # yDiff_df = pd.DataFrame({'y_orig':y_treino2.Milk,
                                          # 'y_diff':[np.nan] + [el for el in np.diff(y_treino2.Milk)]},
                             # index = y_treino2.index) 

                # yDiff_df2 = yDiff_df['y_diff'].copy().dropna()
                # resDiff = sm.tsa.arma_order_select_ic(y_treino2, max_ar=6, max_ma=6, ic='aic')
                

                
                param_grid = {'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
                         'C' :  [0.001, 0.1, 1, 10],
                         'degree' : [3,8],
                         'coef0' : [0.01,10,0.5],
                         'gamma' : [0.001, 0.01, 0.1, 1]}
    
                svr_grid = GridSearchCV(SVR(),param_grid,cv=5, verbose = 2).fit(X_treino2, y_treino2)
                best_params = svr_grid.best_params_
                
                model = SVR(best_params)
                model_fit = model.fit(X_treino2, y_treino2)
                
                # print('SVR demorou ', dt.datetime.now() - a, 'para rodar SVR para a data_prevista em:',data_pred)

                print('Data prevista é:', data_pred)                   

                # plt.plot(yDiff_df.index.get_level_values('Data'), yDiff_df.y_diff)
                # plt.plot(a.index.get_level_values('Data'), model_fit.fittedvalues, color='red')

     
            elif modelname=='RF':
    
                print ('Rodando ', modelname)
                param_grid = {
                             # 'max_features': ['auto', 'sqrt', 'log2'],
                               'bootstrap': [True],
                              'n_estimators': [10,50,100, 500]}
        
                rf_grid = GridSearchCV(RandomForestRegressor(oob_score=False, random_state=None, 
                                    verbose=True, warm_start=False, n_jobs=jobs),
                                   param_grid, cv=4).fit(X_treino2, y_treino2)
       
      
                n_estimators = rf_grid.best_estimator_.n_estimators
                # max_features = rf_grid.best_estimator_.max_features        
                # bootstrap = rf_grid.best_estimator_.bootstrap

                model = RandomForestRegressor(oob_score=False, random_state=None, 
                                      verbose=True, warm_start=False,
                                      bootstrap=True, n_estimators=n_estimators)  
                model_fit = model.fit(X_treino2, y_treino2)
                
                data_pred = y_teste2.index.item().strftime('%d%m%Y')
                print('Data prevista é:', data_pred)
                try:
                    df_coef2[data_pred] = [x for x in model_fit.feature_importances_] 
                    coef_dict[modelname]={'coef':df_coef2, 'estimadores':df_alpha}

                except:
                    coef_dict[modelname]={'coef':df_coef2, 'estimadores':df_alpha}
                
                df_alpha = df_alpha.append(pd.DataFrame({'estimadores':n_estimators},index=[data_pred]))   
    
    
            if modelname=='ARIMA':
                pred_list.append(model_fit.forecast()[0][0])
            elif modelname=='SARIMAX':
                pred_list.append(model_fit.forecast())
                
            elif modelname!='RW':
                pred_list.append(model_fit.predict(X_teste2))
            
            data_list.append(y_teste2.index.item())
            i+=1
            print(i)
    
  
        # coef_dict[modelname]={'coef':df_coef2, 'alpha':df_alpha, 'l1ratio':df_ratio}  
        if modelname=='ARIMA':
            pred_list = {'Data':data_list,
                         'Pred':pred_list}
                      
        else:
            pred_list = {'Data':data_list,
                         'Pred':[x[0] for x in pred_list]}
            
        if modelname!='RW':
            y_teste_aux = pd.DataFrame(pred_list)
            y_teste_aux.columns = ['Data', 'Ypred_'+modelname]
            y_teste_aux = y_teste_aux.set_index('Data')
   
            df_types = y_teste_aux.dtypes.to_frame().reset_index()
            df_types.columns = ['var','type']
            models_to_drop = df_types[df_types['type']=='object']['var'].to_list()
    
            for var in models_to_drop:
                print(var)  
                
                y_teste_aux[var] = [x[0] for x in y_teste_aux[var].to_list() ]    
            
        else:
            y_teste_filter = y[y.index>=y_treino.index[-1]]
            y_teste_aux = y_teste_filter.shift(forecast_horizon).rename(columns={'CEE_BR_TOT':'Ypred_RW'})[1:]

            
        y_teste = y_teste.merge(y_teste_aux, left_index=True, right_index=True, how='left')    
        df_metrics_aux = pd.DataFrame({
                                    'R2_test':  r2_score(y_teste.dropna()[ycol], y_teste.dropna()['Ypred_'+modelname]),
                                    'MSE_test':  mean_squared_error(y_teste.dropna()[ycol], y_teste.dropna()['Ypred_'+modelname]),
                                    'MAE_test':  mean_absolute_error(y_teste.dropna()[ycol], y_teste.dropna()['Ypred_'+modelname]),
                                    'MAPE_test':  mean_absolute_percentage_error(y_teste.dropna()[ycol], y_teste.dropna()['Ypred_'+modelname])},
                                  index=[modelname])
        
        df_metrics_aux['RMSE'] = np.sqrt(df_metrics_aux['MSE_test'])
        df_metrics = df_metrics.append(df_metrics_aux)  
 
        time_model = dt.datetime.now() - time_model_aux
        print('O modelo demorou', time_model, 'para rodar o modelo', modelname)
        
        time_df_aux = pd.DataFrame({'Model':modelname, 'Time':time_model}, index=[modelname])
        
        time_df = time_df.append(time_df_aux)
        
    # df_types = y_teste.dtypes.to_frame().reset_index()
    # df_types.columns = ['var','type']
    # models_to_drop = df_types[df_types['type']=='object']['var'].to_list()
    
    # y_teste = y_teste.dropna(subset=models_to_drop)
    # for var in models_to_drop:
    #     print(var)  
        
    #     y_teste[var] = [x[0] for x in y_teste[var].to_list() ]    


    # fig, ax = plt.subplots(figsize=(8, 4))
    # # ax.plot(y_treino.index.get_level_values('Data'),y_treino[ycol], label="Original")
    # ax.plot(y_teste.index.get_level_values('Data'),y_teste[ycol], label="REAL")
    
    # for col in models_list:
    #     ax.plot(y_teste.index.get_level_values('Data'),y_teste['Ypred_' +col], label=col, alpha=.6)
    
    # # ax.plot(y_teste.index.get_level_values('Data'),y_teste['Ypred_lasso'], label="Previsao - lasso")

    # # ax.plot(y_teste.index.get_level_values('Data'),y_teste['Milk_Pred_CV'], label="Previsao - CV")
    # # for alpha in alpha_list:
    # #     ax.plot(y_teste.index.get_level_values('Data'),y_teste['Milk_Pred' +str(alpha)], label="Previsao - "+str(alpha), alpha=.6)

    # ax.legend(loc='upper left')
    # ax.set_ylabel('em R$ por Litro') 
    # ax.set_title('Previsão Preço do Leite')
    
    # plt.xticks(rotation=30)
    # # plt.savefig('Milk Forecast - GLM.png', bbox_inches='tight') 
    # plt.show()
    
    
    return X_treino, X_teste,  y_treino, y_teste, df_metrics, coef_dict, time_df



def saveResults(df_metrics , y_teste, coef_dict,forecastHorizon, models_list, join):
    
    load_results = {'metricas': df_metrics, 'previsao': y_teste, 'parametros':coef_dict}
    
    if join:
        a_file = open("Results - "+'-'.join(models_list)+" - "+str(forecastHorizon)+".pkl", "wb")
        pickle.dump(load_results, a_file)
        a_file.close()

    else:
        a_file = open("Results - "+models_list[0]+" - "+str(forecastHorizon)+".pkl", "wb")
        pickle.dump(load_results, a_file)
        a_file.close()

    
    return 'Dados Salvos'



colunas2 = ['DM','DS','MÊS','ANO','ESTAC','FER','NEB_BR_9','NEB_BR_15','NEB_BR_21',
'NEB_BR_MED','PA_BR_9','PA_BR_15','PA_BR_21','PA_BR_MED','TEMP_BS_BR_9','TEMP_BS_BR_15','TEMP_BS_BR_21','TEMP_BS_BR_MED','TEMP_BU_BR_9','TEMP_BU_BR_15','TEMP_BU_BR_21','TEMP_BU_BR_MED','UMID_BR_9', 
'UMID_BR_15','UMID_BR_21','UMID_BR_MED','DV_BR_9','DV_BR_15','DV_BR_21','DV_BR_MED','VV_BR_9','VV_BR_15','VV_BR_21','VV_BR_MED','TAR_BR_CSO','TAR_BR_CP','TAR_BR_IP','TAR_BR_IND','TAR_BR_PP','TAR_BR_RED',
'TAR_BR_RR','TAR_BR_RRA','TAR_BR_RRI','TAR_BR_SP1','TAR_BR_SP2','TAR_BR_MED','Meta_Selic', 'Taxa_Selic','CDI','DolarC','DolarC_var','DolarV','DolarV_var','EuroC','EuroC_var','EuroV','EuroV_var','IBV_Cot',
'IBV_min','IBV_max','IBV_varabs','IBV_varperc','IBV_vol','INPC_m','INPC_ac','IPCA_m','IPCA_ac','IPAM_m','IPAM_ac','IPADI_m', 'IPADI_ac' , 'IGPM_m','IGPM_ac','IGPDI_m','IGPDI_ac','PAB_o','PAB_d',
'TVP_o','TVP_d','PICV_o','ICV_d','CCU_o','CCU_d','CS_o','CS_d','UCPIIT_FGV_o','UCPIIT_FGV_d','CPCIIT_CNI_o','CPCIIT_CNI_d','VIR_o','VIR_d','HTPIT_o','HTPIT_d','SRIT_o','SRIT_d','PPOB','PGN','PIG_o','PIG_d','PIBCa_o','PIBCa_d',
'PIBI_o','PIBI_d','PIBCo_o','PIBCo_d','PIA_o','PIA_d','ICC','INEC','ICEI','DBNDES','IEG_o','IEG_d','IETIT_o','IETIT_d','IETC_o','IETC_d','IETS_o','IETS_d','IETCV_o','IETCV_d', 'PO','TD','BM','PME',
'TEMPaj_BS_BR_9','TEMPaj_BS_BR_15','TEMPaj_BS_BR_21','TEMPaj_BS_BR_MED','TEMPaj_BU_N_MED','TEMPaj_BU_BR_9','TEMPaj_BU_BR_15','TEMPaj_BU_BR_21','TEMPaj_BU_BR_MED'] 

 
def sum_df(df_coef):
    
    #Construir df das somas
    coef_cce_list =['CEE_SECO_01','CEE_SECO_02','CEE_SECO_03','CEE_SECO_04','CEE_SECO_05','CEE_SECO_06','CEE_SECO_07','CEE_SECO_08','CEE_SECO_09','CEE_SECO_10','CEE_SECO_11','CEE_SECO_12', 
                'CEE_SECO_13','CEE_SECO_14','CEE_SECO_15','CEE_SECO_16','CEE_SECO_17','CEE_SECO_18','CEE_SECO_19','CEE_SECO_20','CEE_SECO_21','CEE_SECO_22','CEE_SECO_23','CEE_SECO_24', 
                'CEE_SECO_MED','CEE_SECO_CSO','CEE_SECO_CP','CEE_SECO_IP','CEE_SECO_IND','CEE_SECO_PP','CEE_SECO_RED','CEE_SECO_RR','CEE_SECO_RRA','CEE_SECO_RRI','CEE_SECO_SP1','CEE_SECO_SP2', 
                'CEE_SECO_TOT','CEE_SUL_01','CEE_SUL_02','CEE_SUL_03','CEE_SUL_04','CEE_SUL_05','CEE_SUL_06','CEE_SUL_07','CEE_SUL_08','CEE_SUL_09','CEE_SUL_10','CEE_SUL_11', 
                'CEE_SUL_12','CEE_SUL_13','CEE_SUL_14','CEE_SUL_15','CEE_SUL_16','CEE_SUL_17','CEE_SUL_18','CEE_SUL_19','CEE_SUL_20','CEE_SUL_21','CEE_SUL_22','CEE_SUL_23','CEE_SUL_24' ,
                'CEE_SUL_MED','CEE_SUL_CSO','CEE_SUL_CP','CEE_SUL_IP','CEE_SUL_IND','CEE_SUL_PP','CEE_SUL_RED','CEE_SUL_RR','CEE_SUL_RRA','CEE_SUL_RRI','CEE_SUL_SP1','CEE_SUL_SP2','CEE_SUL_TOT', 
                'CEE_NE_01','CEE_NE_02','CEE_NE_03','CEE_NE_04','CEE_NE_05','CEE_NE_06','CEE_NE_07','CEE_NE_08','CEE_NE_09','CEE_NE_10','CEE_NE_11','CEE_NE_12','CEE_NE_13', 
                'CEE_NE_14','CEE_NE_15','CEE_NE_16','CEE_NE_17','CEE_NE_18','CEE_NE_19','CEE_NE_20','CEE_NE_21','CEE_NE_22','CEE_NE_23','CEE_NE_24','CEE_NE_MED','CEE_NE_CSO', 
                'CEE_NE_CP','CEE_NE_IP','CEE_NE_IND','CEE_NE_PP','CEE_NE_RED','CEE_NE_RR','CEE_NE_RRA','CEE_NE_RRI','CEE_NE_SP1','CEE_NE_SP2','CEE_NE_TOT','CEE_N_01','CEE_N_02', 
                'CEE_N_03','CEE_N_04','CEE_N_05','CEE_N_06','CEE_N_07','CEE_N_08','CEE_N_09','CEE_N_10','CEE_N_11','CEE_N_12','CEE_N_13','CEE_N_14','CEE_N_15',
                'CEE_N_16','CEE_N_17','CEE_N_18','CEE_N_19','CEE_N_20','CEE_N_21','CEE_N_22','CEE_N_23','CEE_N_24','CEE_N_MED','CEE_N_CSO','CEE_N_CP','CEE_N_IP', 
                'CEE_N_IND','CEE_N_PP','CEE_N_RED','CEE_N_RR','CEE_N_RRA','CEE_N_RRI','CEE_N_SP1','CEE_N_SP2','CEE_N_TOT','CEE_BR_01','CEE_BR_02','CEE_BR_03','CEE_BR_04',
                'CEE_BR_05','CEE_BR_06','CEE_BR_07','CEE_BR_08','CEE_BR_09','CEE_BR_10','CEE_BR_11','CEE_BR_12','CEE_BR_13','CEE_BR_14','CEE_BR_15','CEE_BR_16','CEE_BR_17' ,
                'CEE_BR_18','CEE_BR_19','CEE_BR_20','CEE_BR_21','CEE_BR_22','CEE_BR_23','CEE_BR_24','CEE_BR_MED','CEE_BR_CSO','CEE_BR_CP','CEE_BR_IP','CEE_BR_IND','CEE_BR_PP', 
                'CEE_BR_RED','CEE_BR_RR','CEE_BR_RRA','CEE_BR_RRI','CEE_BR_SP1','CEE_BR_SP2','CEE_BR_TOT']
    
    coef_vc_list =['DM','DS','MÊS','ANO','ESTAC','FER']
    
    coef_vm_list=['NEB_SECO_9','NEB_SECO_15','NEB_SECO_21','NEB_SECO_MED','PA_SECO_9','PA_SECO_15','PA_SECO_21','PA_SECO_MED','TEMP_BS_SECO_9','TEMP_BS_SECO_15','TEMP_BS_SECO_21','TEMP_BS_SECO_MED','TEMP_BU_SECO_9','TEMP_BU_SECO_15',
                'TEMP_BU_SECO_21','TEMP_BU_SECO_MED','UMID_SECO_9','UMID_SECO_15','UMID_SECO_21','UMID_SECO_MED','DV_SECO_9','DV_SECO_15','DV_SECO_21','DV_SECO_MED','VV_SECO_9','VV_SECO_15','VV_SECO_21', 
                'VV_SECO_MED','NEB_SUL_9','NEB_SUL_15','NEB_SUL_21','NEB_SUL_MED','PA_SUL_9','PA_SUL_15','PA_SUL_21','PA_SUL_MED','TEMP_BS_SUL_9','TEMP_BS_SUL_15','TEMP_BS_SUL_21','TEMP_BS_SUL_MED', 
                'TEMP_BU_SUL_9','TEMP_BU_SUL_15','TEMP_BU_SUL_21','TEMP_BU_SUL_MED','UMID_SUL_9','UMID_SUL_15','UMID_SUL_21','UMID_SUL_MED','DV_SUL_9','DV_SUL_15','DV_SUL_21','DV_SUL_MED','VV_SUL_9',
                'VV_SUL_15','VV_SUL_21','VV_SUL_MED','NEB_NE_9','NEB_NE_15','NEB_NE_21','NEB_NE_MED','PA_NE_9','PA_NE_15','PA_NE_21','PA_NE_MED','TEMP_BS_NE_9','TEMP_BS_NE_15','TEMP_BS_NE_21','TEMP_BS_NE_MED',
                'TEMP_BU_NE_9','TEMP_BU_NE_15','TEMP_BU_NE_21','TEMP_BU_NE_MED','UMID_NE_9','UMID_NE_15','UMID_NE_21','UMID_NE_MED','DV_NE_9','DV_NE_15','DV_NE_21','DV_NE_MED','VV_NE_9','VV_NE_15','VV_NE_21', 
                'VV_NE_MED','NEB_N_9','NEB_N_15','NEB_N_21','NEB_N_MED','PA_N_9','PA_N_15','PA_N_21','PA_N_MED','TEMP_BS_N_9','TEMP_BS_N_15','TEMP_BS_N_21','TEMP_BS_N_MED','TEMP_BU_N_9','TEMP_BU_N_15','TEMP_BU_N_21', 
                'TEMP_BU_N_MED','UMID_N_9','UMID_N_15','UMID_N_21','UMID_N_MED','DV_N_9','DV_N_15','DV_N_21','DV_N_MED','VV_N_9','VV_N_15','VV_N_21','VV_N_MED','NEB_BR_9','NEB_BR_15','NEB_BR_21',
                'NEB_BR_MED','PA_BR_9','PA_BR_15','PA_BR_21','PA_BR_MED','TEMP_BS_BR_9','TEMP_BS_BR_15','TEMP_BS_BR_21','TEMP_BS_BR_MED','TEMP_BU_BR_9','TEMP_BU_BR_15','TEMP_BU_BR_21','TEMP_BU_BR_MED','UMID_BR_9', 
                'UMID_BR_15','UMID_BR_21','UMID_BR_MED','DV_BR_9','DV_BR_15','DV_BR_21','DV_BR_MED','VV_BR_9','VV_BR_15','VV_BR_21','VV_BR_MED','TEMPaj_BS_SECO_9','TEMPaj_BS_SECO_15','TEMPaj_BS_SECO_21','TEMPaj_BS_SECO_MED',
                'TEMPaj_BS_SUL_9','TEMPaj_BS_SUL_15','TEMPaj_BS_SUL_21','TEMPaj_BS_SUL_MED','TEMPaj_BS_NE_9','TEMPaj_BS_NE_15','TEMPaj_BS_NE_21','TEMPaj_BS_NE_MED','TEMPaj_BS_N_9','TEMPaj_BS_N_15','TEMPaj_BS_N_21','TEMPaj_BS_N_MED',
                'TEMPaj_BS_BR_9','TEMPaj_BS_BR_15','TEMPaj_BS_BR_21','TEMPaj_BS_BR_MED','TEMPaj_BU_SECO_9','TEMPaj_BU_SECO_15','TEMPaj_BU_SECO_21','TEMPaj_BU_SECO_MED','TEMPaj_BU_SUL_9','TEMPaj_BU_SUL_15','TEMPaj_BU_SUL_21',
                'TEMPaj_BU_SUL_MED','TEMPaj_BU_NE_9','TEMPaj_BU_NE_15','TEMPaj_BU_NE_21','TEMPaj_BU_NE_MED','TEMPaj_BU_N_9','TEMPaj_BU_N_15','TEMPaj_BU_N_21','TEMPaj_BU_N_MED','TEMPaj_BU_BR_9','TEMPaj_BU_BR_15','TEMPaj_BU_BR_21','TEMPaj_BU_BR_MED']
                
    coef_vtar_list=['TAR_SECO_CSO','TAR_SECO_CP','TAR_SECO_IP','TAR_SECO_IND','TAR_SECO_PP','TAR_SECO_RED','TAR_SECO_RR','TAR_SECO_RRA','TAR_SECO_RRI','TAR_SECO_SP1','TAR_SECO_SP2','TAR_SECO_MED','TAR_SUL_CSO',
                'TAR_SUL_CP','TAR_SUL_IP','TAR_SUL_IND','TAR_SUL_PP','TAR_SUL_RED', 'TAR_SUL_RR','TAR_SUL_RRA','TAR_SUL_RRI','TAR_SUL_SP1','TAR_SUL_SP2','TAR_SUL_MED','TAR_NE_CSO','TAR_NE_CP','TAR_NE_IP','TAR_NE_IND','TAR_NE_PP','TAR_NE_RED','TAR_NE_RR','TAR_NE_RRA', 
                'TAR_NE_RRI','TAR_NE_SP1','TAR_NE_SP2','TAR_NE_MED','TAR_N_CSO','TAR_N_CP','TAR_N_IP','TAR_N_IND','TAR_N_PP','TAR_N_RED','TAR_N_RR','TAR_N_RRA','TAR_N_RRI','TAR_N_SP1','TAR_N_SP2',
                'TAR_N_MED','TAR_BR_CSO','TAR_BR_CP','TAR_BR_IP','TAR_BR_IND','TAR_BR_PP','TAR_BR_RED','TAR_BR_RR','TAR_BR_RRA','TAR_BR_RRI','TAR_BR_SP1','TAR_BR_SP2','TAR_BR_MED']
               
    coef_ve_list=['Meta_Selic', 'Taxa_Selic','CDI','DolarC','DolarC_var','DolarV','DolarV_var','EuroC','EuroC_var','EuroV','EuroV_var','IBV_Cot','IBV_min','IBV_max','IBV_varabs','IBV_varperc','IBV_vol','INPC_m',
                'INPC_ac','IPCA_m','IPCA_ac','IPAM_m','IPAM_ac','IPADI_m', 'IPADI_ac' , 'IGPM_m','IGPM_ac','IGPDI_m','IGPDI_ac','PAB_o','PAB_d','TVP_o','TVP_d','PICV_o','ICV_d','CCU_o',
                'CCU_d','CS_o','CS_d','UCPIIT_FGV_o','UCPIIT_FGV_d','CPCIIT_CNI_o','CPCIIT_CNI_d','VIR_o','VIR_d','HTPIT_o','HTPIT_d','SRIT_o','SRIT_d','PPOB','PGN','PIG_o','PIG_d','PIBCa_o','PIBCa_d',
                'PIBI_o','PIBI_d','PIBCo_o','PIBCo_d','PIA_o','PIA_d','ICC','INEC','ICEI','DBNDES','IEG_o','IEG_d','IETIT_o','IETIT_d','IETC_o','IETC_d','IETS_o','IETS_d','IETCV_o','IETCV_d', 'PO','TD','BM','PME'] 
    
    
    lista1=[]
    lista2=[]
    lista3=[]
    lista4=[]
    lista5=[]
    
    for i in list(range(1,4)):
        for j in coef_cce_list:
            lista1.append(j+'_lag'+str(i))
        for k in coef_vc_list:
            lista2.append(k+'_lag'+str(i))
        for l in coef_vm_list:
            lista3.append(l+'_lag'+str(i))
        for m in coef_vtar_list:
            lista4.append(m+'_lag'+str(i))
        for n in coef_ve_list:
            lista5.append(n+'_lag'+str(i))        
    
    
    coef_cce_list=coef_cce_list+lista1
    coef_vc_list=coef_vc_list+lista2
    coef_vm_list=coef_vm_list+lista3
    coef_vtar_list=coef_vtar_list+lista4
    coef_ve_list=coef_ve_list+lista5
    
    
    rest_list1=list(set(df_coef.index)-set(coef_cce_list))
    coef_cce_df=df_coef.copy().drop(rest_list1)
    coef_cce_sum_list=[]
    for i in coef_cce_df.columns:
        coef_cce_sum_list.append(sum(abs(coef_cce_df[i])))
        
    rest_list2=list(set(df_coef.index)-set(coef_vc_list))
    coef_vc_df=df_coef.copy().drop(rest_list2)
    coef_vc_sum_list=[]
    for i in coef_vc_df.columns:
        coef_vc_sum_list.append(sum(abs(coef_vc_df[i])))
    
    rest_list3=list(set(df_coef.index)-set(coef_vm_list))
    coef_vm_df=df_coef.copy().drop(rest_list3)
    coef_vm_sum_list=[]
    for i in coef_vm_df.columns:
        coef_vm_sum_list.append(sum(abs(coef_vm_df[i])))
    
    rest_list4=list(set(df_coef.index)-set(coef_vtar_list))
    coef_vtar_df=df_coef.copy().drop(rest_list4)
    coef_vtar_sum_list=[]
    for i in coef_vtar_df.columns:
        coef_vtar_sum_list.append(sum(abs(coef_vtar_df[i])))
    
    rest_list5=list(set(df_coef.index)-set(coef_ve_list))
    coef_ve_df=df_coef.copy().drop(rest_list5)
    coef_ve_sum_list=[]
    for i in coef_ve_df.columns:
        coef_ve_sum_list.append(sum(abs(coef_ve_df[i])))
        
        
    listao=[]
    listao.append(coef_cce_sum_list)
    listao.append(coef_vc_sum_list)
    listao.append(coef_vm_sum_list)
    listao.append(coef_vtar_sum_list)
    listao.append(coef_ve_sum_list)
    
    coef_sum_df=pd.DataFrame(data=listao,columns=df_coef.columns, index=['CEE Sum','Cal Var Sum' ,'Met Var Sum','Tar Var Sum','Econ Var Sum'])

    return coef_sum_df



def previsao_df(R2_list,EQM_list,resid_list,accuracy_list,R2_test_list):
    
    index=['Modelos','R2', 'EQM', 'Resíduo','Accuracy', 'R2 Teste']    
    colunas3 = ['AR','Random Walk','Linear Regression','Lasso','Lars','Lasso Lars', 'Ridge','Elastic Net','Random Forest']
    
    previsao = pd.DataFrame([colunas3,R2_list, EQM_list, resid_list, accuracy_list,R2_test_list ],index=index, columns=list(range(1,10)))
    previsao=previsao.transpose()
    previsao=previsao.sort_values(by='Resíduo', ascending=True)
    print(previsao)
    
    
    #Plot4
    print(previsao[['Modelos','Resíduo']])
    
    #previsao.sort_values(by='Resíduo', ascending=True)[0:1].index.item()
    #previsao.sort_values(by='Resíduo', ascending=True)[0:1]['Modelos'].item()
    
    previsao_aux=pd.DataFrame(data=y_predictions1)
    previsao_aux[1]=y_predictions2
    previsao_aux[2]=y_predictions3
    previsao_aux[3]=y_predictions4
    previsao_aux[4]=y_predictions5
    previsao_aux[5]=y_predictions6
    previsao_aux[6]=y_predictions8
    previsao_aux[7]=y_predictions9
    previsao_aux[8]=y_predictions10
    previsao_aux.columns = [list(range(1,10))]
    
    return previsao, previsao_aux


def plot_forecast(y, y_treino,y_teste,forecast_horizon, df_metrics):
    
    y_aux = y[y.index.year>2017]  
    y_treino_aux = y_treino[y_treino.index.year>2017]  

    x_label=[]
    term=30
    while term<len(y_aux.index):   
        x_label.append(y_aux.index[term])
        term=term+round(len(y_aux.index)/8)


    best_model = df_metrics.model.to_list()[:4]
    target = 'CEE_BR_TOT'
    

    n=4
    fig2, axes = plt.subplots(nrows=n, ncols=1, figsize=(7, 7))
    for i in range(n):
        axes[i].set_title(best_model[i])
        axes[i].plot(y_treino_aux/1e6, label='Train')
        axes[i].plot(pd.to_datetime(y_teste.index),y_teste[target]/1e6, color='black', label='Test')
        axes[i].plot(pd.to_datetime(y_teste.index),y_teste['Ypred_'+best_model[i]]/1e6, label='Forecast')
        axes[i].legend(loc='lower left')
        axes[i].set_ylabel("mWh")
        # axes[i].set_xlim(y.index[0]-10, y.index[-1]+5)
        # axes[i].set_xticks(x_label)
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
        axes[i].xaxis.set_minor_formatter(mdates.DateFormatter("%m-%Y"))
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(8))


        #axes[0].set_rotation(30)
    

    fig2.tight_layout()
    plt.savefig('modelo'+str(forecast_horizon)+'dias.png') 
    plt.show()
    
    return None



def save_results(y_teste, coef_sum_df,df_metrics, forecast_horizon):
    
    writer = pd.ExcelWriter('dados_fh'+str(forecast_horizon)+'.xlsx')
    y_teste.to_excel(writer,'previsao_lag'+str(forecast_horizon))
    coef_sum_df.to_excel(writer,'coeficientes_lag'+str(forecast_horizon))
    df_metrics.to_excel(writer,'metricas_lag'+str(forecast_horizon))
    writer.save()
    
    return None

def open_results():
    
    # dict_results = {}
    df_pred_agg = pd.DataFrame()
    df_metrics_agg = pd.DataFrame()
    df_metrics_agg2 = pd.DataFrame()
    df_coeff_agg = pd.DataFrame()
    
    for file in [x for x in os.listdir() if 'dados_fh' in x]:
        
        fh = re.findall('\d{2}|\d{1}', file)[0]
        # print(fh)
        
        dict_r = pd.read_excel(file, sheet_name=None)
        
        df_pred_aux = dict_r[[x for x in dict_r.keys() if 'previsao' in x][0]].copy()
        df_metrics_aux = dict_r[[x for x in dict_r.keys() if 'metricas' in x][0]].copy()
        df_coeff_aux = dict_r[[x for x in dict_r.keys() if 'coeficientes' in x][0]].copy()
        
        
        df_pred_aux['fh'] = int(fh)
        df_metrics_aux['fh'] = int(fh)
        df_coeff_aux['fh'] = int(fh)

        df_metrics_aux2 = df_metrics_aux[:4]


        df_pred_agg = df_pred_agg.append(df_pred_aux)
        df_metrics_agg = df_metrics_agg.append(df_metrics_aux)
        df_metrics_agg2 = df_metrics_agg2.append(df_metrics_aux2)
        df_coeff_agg = df_coeff_agg.append(df_coeff_aux)
   

    df_pred_agg = df_pred_agg.sort_values('fh')
    df_metrics_agg = df_metrics_agg.sort_values(['fh','mcs','MSE_test'], ascending=[True, False, True])
    
    df_coeff_agg = df_coeff_agg#.sort_values('fh')    
    df_coeff_agg = df_coeff_agg.rename(columns={'Unnamed: 0':'var'})


    
    df_metrics_agg2 = df_metrics_agg2[[ 'model','fh', 'mcs','RMSE', 'MAE_test', 'MAPE_test','R2_test']]
    df_metrics_agg2.columns = [ 'Model','forecast_horizon', 'MCS','RMSE', 'MAE', 'MAPE','R2']
    
    for col in df_metrics_agg2.columns:
        if df_metrics_agg2[col].dtype=='int64' or df_metrics_agg2[col].dtype=='float64':
            if col=='R2':
                df_metrics_agg2[col] = df_metrics_agg2[col].apply(lambda x: round(x,2))
            elif col=='MAPE':
                df_metrics_agg2[col] = df_metrics_agg2[col].apply(lambda x: round(x,1))
            else:
                df_metrics_agg2[col] = df_metrics_agg2[col].apply(lambda x: round(x))


    dict_cond ={'LASSOLARS': 'Lasso Lars',
                 'RF': 'Random Forest',
                 'LASSO': 'Lasso',
                 'EN': 'Elastic Net',
                 'ARIMA': 'Arima',
                 'RIDGE': 'Ridge',
                 'RW': 'Random Walk'}
    conditions = []
    choices = []
    for key,value in dict_cond.items():
        conditions.append(df_metrics_agg2['Model']==key)
        choices.append(value)

    df_metrics_agg2['Model'] = np.select(conditions,choices)
    df_metrics_agg2 = df_metrics_agg2.sort_values(['forecast_horizon','MCS','RMSE'], ascending=[True, False, True])        
 
    cols = list(df_coeff_agg.columns[1:5])
    df_metrics_agg2['combined'] = df_metrics_agg2[[x for x in df_metrics_agg2.columns if 'forecast' not in x]].apply(lambda row: '&'.join(row.values.astype(str)), axis=1)


  
    bm =  'RF'
    bm2 = 'LASSOLARS'
    bm3 = 'EN'
    bm4 = 'RIDGE'

    df_coeff_agg_filter  = df_coeff_agg[['var','fh']+[bm]]
    df_coeff_bm = df_coeff_agg_filter.pivot(index="fh", columns="var", values=bm).reset_index()

    df_coeff_agg_filter2 = df_coeff_agg[['var','fh']+[bm2]]
    df_coeff_bm2 = df_coeff_agg_filter2.pivot(index="fh", columns="var", values=bm2).reset_index()

    df_coeff_agg_filter3  = df_coeff_agg[['var','fh']+[bm3]]
    df_coeff_bm3 = df_coeff_agg_filter3.pivot(index="fh", columns="var", values=bm3).reset_index()

    df_coeff_agg_filter4  = df_coeff_agg[['var','fh']+[bm4]]
    df_coeff_bm4 = df_coeff_agg_filter4.pivot(index="fh", columns="var", values=bm4).reset_index()    
    
    vars_list = list(df_coeff_bm2.columns[1:])
    vars_list = ['CEE Sum', 'Cal Var Sum', 'Met Var Sum', 'Tar Var Sum', 'Econ Var Sum']

    var_dict = {'CEE Sum':'PEC Variables',
                'Cal Var Sum': 'Calendar Variables',
                'Econ Var Sum':'Economic Variables',
                'Met Var Sum':'Weather Variables',
                'Tar Var Sum':'Price Variables'
                }   
    
    for i,bm in enumerate(['RF','LASSOLARS', 'EN','RIDGE']):
        print(i)
    
        df_coeff_agg_filter  = df_coeff_agg[['var','fh']+[bm]]
        df_coeff_bm = df_coeff_agg_filter.pivot(index="fh", columns="var", values=bm).reset_index()
        
        n=len(vars_list)
        fig2, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 10), sharex=True)
        for j in range(n):
            axes[j].set_title(var_dict[vars_list[i]])
            axes[j].plot(df_coeff_bm['fh'],df_coeff_bm[vars_list[i]], label=bm)
            # axe2 = axes[i].twinx()
            # axe2.plot(df_coeff_bm2['fh'],df_coeff_bm2[vars_list[i]], color='orange', label = bm2)
            # axes[i].legend(loc='lower left')
            axes[j].set_ylabel("coff sum")
            axes[j].set_xticks([1,7,15,30,60,90])
    
        axes[4].set_xlabel("forecast horizon")        
        plt.suptitle(bm+'\n', size=16)
        fig2.tight_layout()

        plt.savefig('coef_sum_agg_bm{}.png'.format(str(i))) 
        plt.show()


        

        df_coef_pivoted =  df_coeff_agg[['var',bm,'fh']].pivot(index='fh',columns='var', values=bm)


        # df_coef_pivoted=(df_coef_pivoted-df_coef_pivoted.min())/(df_coef_pivoted.max()-df_coef_pivoted.min())
        # df_pivoted_norm = df_coef_pivoted =  df_coeff_agg[['var',bm,'fh']].pivot(index='fh',columns='var', values=bm)
        df_pivoted_norm = df_coef_pivoted / df_coef_pivoted.sum()
        df_pivoted_norm=df_pivoted_norm#.reset_index()


        # We need to transform the data from raw data to percentage (fraction)
        data_perc = df_pivoted_norm.divide(df_pivoted_norm.sum(axis=1), axis=0)
         
        
        # Make the plot
        fig2, axes = plt.subplots(figsize=(10, 7))
    
        axes.stackplot(data_perc.index,  data_perc["CEE Sum"],  data_perc["Cal Var Sum"], data_perc["Met Var Sum"],data_perc["Tar Var Sum"],  data_perc["Econ Var Sum"],
                      labels=['CEE Sum', 'Cal Var Sum', 'Met Var Sum','Tar Var Sum' ,'Econ Var Sum'], alpha=.7)
        axes.legend(loc='center left', bbox_to_anchor=(1.05, .5))
        plt.margins(0,0)
        axes.set_ylabel("%")
        axes.set_xticks([1,7,15,30,60,90])
        
        plt.title(bm.upper()+' variables')
        fig2.tight_layout()
        plt.savefig('stacked_model_{}.png'.format(str(bm))) 
        plt.show()



    fig, axs = plt.subplots(nrows=4, figsize=(8, 10), sharex=True)
    for i,bm in enumerate(['RF','LASSOLARS', 'EN','RIDGE']):
        
        print(bm)
        
        df_coef_pivoted =  df_coeff_agg[['var',bm,'fh']].pivot(index='fh',columns='var', values=bm)

        print(df_coef_pivoted)

        # df_coef_pivoted=(df_coef_pivoted-df_coef_pivoted.min())/(df_coef_pivoted.max()-df_coef_pivoted.min())
        # df_pivoted_norm = df_coef_pivoted =  df_coeff_agg[['var',bm,'fh']].pivot(index='fh',columns='var', values=bm)
        df_pivoted_norm = df_coef_pivoted / df_coef_pivoted.sum()
        df_pivoted_norm=df_pivoted_norm#.reset_index()


        # We need to transform the data from raw data to percentage (fraction)
        data_perc = df_pivoted_norm.divide(df_pivoted_norm.sum(axis=1), axis=0)
        axs[i].set_title(bm)
        axs[i].stackplot(data_perc.index,  data_perc["CEE Sum"],  data_perc["Cal Var Sum"], data_perc["Met Var Sum"],data_perc["Tar Var Sum"],  data_perc["Econ Var Sum"],
                         labels=['CEE Sum', 'Cal Var Sum', 'Met Var Sum','Tar Var Sum' ,'Econ Var Sum'], alpha=.7)
        
        # axes[i].legend(loc='center left', bbox_to_anchor=(1.05, .5))
        axs[i].margins(0,0)
        # axes[i].set_ylabel("%")
        # axes[i].set_xticks([1,7,15,30,60,90])
        
        # axe2 = axes[i].twinx()
        # axe2.plot(df_coeff_bm2['fh'],df_coeff_bm2[vars_list[i]], color='orange', label = bm2)
        # axes[i].legend(loc='lower left')
        # axs[i].set_ylabel("coff sum")
        # axs[i].set_xticks([1,7,15,30,60,90])
    
    
    
    # axes[2].legend(loc='lower left')

    axs[3].set_xlabel("forecast horizon")  
    axs[3].set_xticks([1,7,15,30,60,90])
      
    # plt.suptitle(bm+'\n', size=16)
    # plt.margins(0,0)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [var_dict[key] for key  in labels] 
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc=10,  bbox_to_anchor=(1.2, 2.4))
    
    # fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)  
    fig.tight_layout()

    plt.savefig('stacked_agg.png'.format(str(i))) 
    plt.show()

    
    
    writer = pd.ExcelWriter('coeficientes.xlsx')
    df_coef_pivoted.to_excel(writer,'agregado', index=False)
    df_pivoted_norm.to_excel(writer,'percentual', index=False)
    data_perc.to_excel(writer,'soma_um', index=False)

    writer.save()
    




    fh_list = df_metrics_agg.fh.drop_duplicates().to_list()
    
    rows = 3
    columns = 2
  
    fig2, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(8, 10))
    i=0
    for row in range(rows):
        for col in range(columns):
            
            metrics_filter = df_metrics_agg[df_metrics_agg.fh==fh_list[i]]
            # print(metrics_filter)
            
            metrics_filter['index'] = metrics_filter['RMSE']/metrics_filter[metrics_filter['model']=='RF']['RMSE'].item()
            # metrics_filter=metrics_filter.sort_values('index')
            # print(metrics_filter)
    
            x=np.arange(3)
            y1 = metrics_filter[metrics_filter['model']=='ARIMA']['index'].item()
            y2 = metrics_filter[metrics_filter['model']=='RW']['index'].item()
    
            axes[row,col].set_title('Forecast Horizon: {}'.format(str(fh_list[i])))
            axes[row,col].bar(x[0], (1), 0.35, color=['blue'], label='RF')
            axes[row,col].bar(x[1], (y1), 0.35, color=['red'], label='ARIMA')
            axes[row,col].bar(x[2], (y2), 0.35, color=['red'], label='RW')
            axes[row,col].set_ylim(0, max(1,y1,y2)+.3)
            axes[row,col].legend(loc='upper left')
            axes[row,col].set_ylabel('RMSE Index')
            axes[row,col].set_xticks(x)    
            axes[row,col].set_xticklabels(['RF','ARIMA','RW'])
                        
            print('fh: {} - arima({}) e rw ({})'.format(fh_list[i],y1,y2))            
            i+=1
            
    fig2.tight_layout()
    plt.savefig('barplot.png') 
    plt.show()        
    
    
    
    for fh in fh_list:
        print(fh)
        
        
    return df_pred_agg, df_metrics_agg, df_coeff_agg


def barplot_single(previsao, forecast_horizon):
        
    x=np.arange(3)
    y=[previsao[previsao.Modelos=='AR'].Resíduo.item(), previsao[0:1].Resíduo.item(), previsao[1:2].Resíduo.item()]
    
    plt.bar(x, (y), 0.35, color=['red', 'blue', 'blue'])
    
    plt.ylabel('RMSE')
    plt.title('Benchmark vs Best Models')
    plt.xticks(np.arange(3),['Benchmark', previsao[0:1].Modelos.item(), previsao[1:2].Modelos.item()])
    #plt.yticks(np.arange(0,previsao[previsao.Modelos=='AR'].Resíduo.item()+10000 , 10000))
    plt.ylim(0, previsao[previsao.Modelos=='AR'].Resíduo.item()+150000)
    plt.legend()
    
    
    for i,j in enumerate(y):
        plt.text(x = i-0.15, y = j+25000, s = str(int(j)), size = 10)
    
    plt.savefig('barplot_fh'+str(forecast_horizon)+'.png')   
    plt.show()
    
    return None


#Colocar Label no grafixo
#Fazer opcao com ARIMA
def barplot_aggregate(flag=0):
    
    results = read_csv('barplot.csv', sep=';',header=0, parse_dates=[0], index_col=False, squeeze=True)

    if flag==0:

        x=np.arange(3)
        y1_1=[results[results.FH=='1'].RW.item()/results[results.FH=='1'].RW.item()]
        y1_2=[results[results.FH=='1'].FBM_result.item()/results[results.FH=='1'].RW.item(), results[results.FH=='1'].SBM_result.item()/results[results.FH=='1'].RW.item()]

        y2_1=[results[results.FH=='7'].RW.item()/results[results.FH=='7'].RW.item()]
        y2_2=[results[results.FH=='7'].FBM_result.item()/results[results.FH=='7'].RW.item(), results[results.FH=='7'].SBM_result.item()/results[results.FH=='7'].RW.item()]

        y3_1=[results[results.FH=='15'].RW.item()/results[results.FH=='15'].RW.item()]
        y3_2=[results[results.FH=='15'].FBM_result.item()/results[results.FH=='15'].RW.item(), results[results.FH=='15'].SBM_result.item()/results[results.FH=='15'].RW.item()]

        y4_1=[results[results.FH=='30'].RW.item()/results[results.FH=='30'].RW.item()]
        y4_2=[results[results.FH=='30'].FBM_result.item()/results[results.FH=='30'].RW.item(), results[results.FH=='30'].SBM_result.item()/results[results.FH=='30'].RW.item()]

        y5_1=[results[results.FH=='60'].RW.item()/results[results.FH=='60'].RW.item()]
        y5_2=[results[results.FH=='60'].FBM_result.item()/results[results.FH=='60'].RW.item(), results[results.FH=='60'].SBM_result.item()/results[results.FH=='60'].RW.item()]

        y6_1=[results[results.FH=='90'].RW.item()/results[results.FH=='90'].RW.item()]
        y6_2=[results[results.FH=='90'].FBM_result.item()/results[results.FH=='90'].RW.item(), results[results.FH=='90'].SBM_result.item()/results[results.FH=='90'].RW.item()]
      


        fig2, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))
        
        axes[0,0].set_title('Forecast Horizon: 1')
        axes[0,0].bar(x[0], (y1_1), 0.35, color=['red'], label='Benchmark')
        axes[0,0].bar(x[1:], (y1_2), 0.35, color=['blue'], label='Best Models')
        axes[0,0].set_ylim(0, 1.05)
        axes[0,0].legend(loc='best')
        axes[0,0].set_ylabel('RMSE Index')
        axes[0,0].set_xticks(x)    
        axes[0,0].set_xticklabels(['RW', results[results.FH=='1'].FBM_nickname.item(), results[results.FH=='1'].SBM_nickname.item()])
              
        axes[0,1].set_title('Forecast Horizon: 7')
        axes[0,1].bar(x[0], (y2_1), 0.35, color=['red'], label='Benchmark')
        axes[0,1].bar(x[1:], (y2_2), 0.35, color=['blue'], label='Best Models')
        axes[0,1].set_ylim(0, 1.05)
        axes[0,1].legend(loc='best')
        axes[0,1].set_ylabel('RMSE Index')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(['RW', results[results.FH=='7'].FBM_nickname.item(), results[results.FH=='7'].SBM_nickname.item()])
        
        axes[1,0].set_title('Forecast Horizon: 15')
        axes[1,0].bar(x[0], (y3_1), 0.35, color=['red'], label='Benchmark')
        axes[1,0].bar(x[1:], (y3_2), 0.35, color=['blue'],label='Best Models')
        axes[1,0].set_ylim(0, 1.05)
        axes[1,0].legend(loc='best')
        axes[1,0].set_ylabel('RMSE Index')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(['RW', results[results.FH=='15'].FBM_nickname.item(), results[results.FH=='15'].SBM_nickname.item()])
    
        axes[1,1].set_title('Forecast Horizon: 30')
        axes[1,1].bar(x[0], (y4_1), 0.35, color=['red'], label='Benchmark')
        axes[1,1].bar(x[1:], (y4_2), 0.35, color=['blue'],label='Best Models')
        axes[1,1].set_ylim(0, 1.05)
        axes[1,1].legend(loc='best')
        axes[1,1].set_ylabel('$RMSE/RMSE_{LASSO}$')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(['RW', results[results.FH=='30'].FBM_nickname.item(), results[results.FH=='30'].SBM_nickname.item()])
    
        axes[2,0].set_title('Forecast Horizon: 60')
        axes[2,0].bar(x[0], (y5_1), 0.35, color=['red'], label='Benchmark')
        axes[2,0].bar(x[1:], (y5_2), 0.35, color=['blue'],label='Best Models')
        axes[2,0].set_ylim(0, 1.05)
        axes[2,0].legend(loc='best')
        axes[2,0].set_ylabel('RMSE Index')
        axes[2,0].set_xticks(x)
        axes[2,0].set_xticklabels(['RW', results[results.FH=='60'].FBM_nickname.item(), results[results.FH=='60'].SBM_nickname.item()])
    
        axes[2,1].set_title('Forecast Horizon: 90')
        axes[2,1].bar(x[0], (y6_1), 0.35, color=['red'], label='Benchmark')
        axes[2,1].bar(x[1:], (y6_2), 0.35, color=['blue'],label='Best Models')
        axes[2,1].set_ylim(0, 1.05)
        axes[2,1].legend(loc='best')
        axes[2,1].set_ylabel('RMSE Index')
        axes[2,1].set_xticks(x)
        axes[2,1].set_xticklabels(['RW', results[results.FH=='90'].FBM_nickname.item(), results[results.FH=='90'].SBM_nickname.item()])
        fig2.tight_layout()
    
        plt.savefig('barplot1_v2.png') 
#        plt.show()
        
    elif flag==1:
         
        x=np.arange(4)
#        y1_1=[results[results.FH=='1'].ARIMA.item()/results[results.FH=='1'].FBM_result.item(),results[results.FH=='1'].RW.item()/results[results.FH=='1'].FBM_result.item()]
#        y1_2=[results[results.FH=='1'].FBM_result.item()/results[results.FH=='1'].FBM_result.item(), results[results.FH=='1'].SBM_result.item()/results[results.FH=='1'].FBM_result.item()]
#
#        y2_1=[results[results.FH=='7'].ARIMA.item()/results[results.FH=='7'].FBM_result.item(),results[results.FH=='7'].RW.item()/results[results.FH=='7'].FBM_result.item()]
#        y2_2=[results[results.FH=='7'].FBM_result.item()/results[results.FH=='7'].FBM_result.item(), results[results.FH=='7'].SBM_result.item()/results[results.FH=='7'].FBM_result.item()]
#
#        y3_1=[results[results.FH=='15'].ARIMA.item()/results[results.FH=='15'].FBM_result.item(),results[results.FH=='15'].RW.item()/results[results.FH=='15'].FBM_result.item()]
#        y3_2=[results[results.FH=='15'].FBM_result.item()/results[results.FH=='15'].FBM_result.item(), results[results.FH=='15'].SBM_result.item()/results[results.FH=='15'].FBM_result.item()]
#
#        y4_1=[results[results.FH=='30'].ARIMA.item()/results[results.FH=='30'].FBM_result.item(),results[results.FH=='30'].RW.item()/results[results.FH=='30'].FBM_result.item()]
#        y4_2=[results[results.FH=='30'].FBM_result.item()/results[results.FH=='30'].FBM_result.item(), results[results.FH=='30'].SBM_result.item()/results[results.FH=='30'].FBM_result.item()]
#
#        y5_1=[results[results.FH=='60'].ARIMA.item()/results[results.FH=='60'].FBM_result.item(),results[results.FH=='60'].RW.item()/results[results.FH=='60'].FBM_result.item()]
#        y5_2=[results[results.FH=='60'].FBM_result.item()/results[results.FH=='60'].FBM_result.item(), results[results.FH=='60'].SBM_result.item()/results[results.FH=='60'].FBM_result.item()]
#
#        y6_1=[results[results.FH=='90'].ARIMA.item()/results[results.FH=='90'].FBM_result.item(),results[results.FH=='90'].RW.item()/results[results.FH=='90'].FBM_result.item()]
#        y6_2=[results[results.FH=='90'].FBM_result.item()/results[results.FH=='90'].FBM_result.item(), results[results.FH=='90'].SBM_result.item()/results[results.FH=='90'].FBM_result.item()]
# 
        
        y1_1=[results[results.FH=='1'].ARIMA.item()/results[results.FH=='1'].ARIMA.item(),results[results.FH=='1'].RW.item()/results[results.FH=='1'].ARIMA.item()]
        y1_2=[results[results.FH=='1'].FBM_result.item()/results[results.FH=='1'].ARIMA.item(), results[results.FH=='1'].SBM_result.item()/results[results.FH=='1'].ARIMA.item()]

        y2_1=[results[results.FH=='7'].ARIMA.item()/results[results.FH=='7'].ARIMA.item(),results[results.FH=='7'].RW.item()/results[results.FH=='7'].ARIMA.item()]
        y2_2=[results[results.FH=='7'].FBM_result.item()/results[results.FH=='7'].ARIMA.item(), results[results.FH=='7'].SBM_result.item()/results[results.FH=='7'].ARIMA.item()]

        y3_1=[results[results.FH=='15'].ARIMA.item()/results[results.FH=='15'].ARIMA.item(),results[results.FH=='15'].RW.item()/results[results.FH=='15'].ARIMA.item()]
        y3_2=[results[results.FH=='15'].FBM_result.item()/results[results.FH=='15'].ARIMA.item(), results[results.FH=='15'].SBM_result.item()/results[results.FH=='15'].ARIMA.item()]

        y4_1=[results[results.FH=='30'].ARIMA.item()/results[results.FH=='30'].ARIMA.item(),results[results.FH=='30'].RW.item()/results[results.FH=='30'].ARIMA.item()]
        y4_2=[results[results.FH=='30'].FBM_result.item()/results[results.FH=='30'].ARIMA.item(), results[results.FH=='30'].SBM_result.item()/results[results.FH=='30'].ARIMA.item()]

        y5_1=[results[results.FH=='60'].ARIMA.item()/results[results.FH=='60'].ARIMA.item(),results[results.FH=='60'].RW.item()/results[results.FH=='60'].ARIMA.item()]
        y5_2=[results[results.FH=='60'].FBM_result.item()/results[results.FH=='60'].ARIMA.item(), results[results.FH=='60'].SBM_result.item()/results[results.FH=='60'].ARIMA.item()]

        y6_1=[results[results.FH=='90'].ARIMA.item()/results[results.FH=='90'].ARIMA.item(),results[results.FH=='90'].RW.item()/results[results.FH=='90'].ARIMA.item()]
        y6_2=[results[results.FH=='90'].FBM_result.item()/results[results.FH=='90'].ARIMA.item(), results[results.FH=='90'].SBM_result.item()/results[results.FH=='90'].ARIMA.item()]
         
        fig2, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))
        
        axes[0,0].set_title('Forecast Horizon: 1')
        axes[0,0].bar(x[:2], (y1_1), 0.35, color=['red'], label='Benchmark')
        axes[0,0].bar(x[2:], (y1_2), 0.35, color=['blue'], label='Best Models')
#        axes[0,0].set_ylim(0, 42)
        axes[0,0].legend(loc='best')
        axes[0,0].set_ylabel('RMSE Index')
        axes[0,0].set_xticks(x)    
        axes[0,0].set_xticklabels(['ARIMA','RW', results[results.FH=='1'].FBM_nickname.item(), results[results.FH=='1'].SBM_nickname.item()])
              
        axes[0,1].set_title('Forecast Horizon: 7')
        axes[0,1].bar(x[:2], (y2_1), 0.35, color=['red'], label='Benchmark')
        axes[0,1].bar(x[2:], (y2_2), 0.35, color=['blue'], label='Best Models')
#        axes[0,1].set_ylim(0, 42)
        axes[0,1].legend(loc='best')
        axes[0,1].set_ylabel('RMSE Index')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(['ARIMA','RW', results[results.FH=='7'].FBM_nickname.item(), results[results.FH=='7'].SBM_nickname.item()])
        
        axes[1,0].set_title('Forecast Horizon: 15')
        axes[1,0].bar(x[:2], (y3_1), 0.35, color=['red'], label='Benchmark')
        axes[1,0].bar(x[2:], (y3_2), 0.35, color=['blue'],label='Best Models')
#        axes[1,0].set_ylim(0, 22)
        axes[1,0].legend(loc='best')
        axes[1,0].set_ylabel('RMSE Index')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(['ARIMA','RW', results[results.FH=='15'].FBM_nickname.item(), results[results.FH=='15'].SBM_nickname.item()])
    
        axes[1,1].set_title('Forecast Horizon: 30')
        axes[1,1].bar(x[:2], (y4_1), 0.35, color=['red'], label='Benchmark')
        axes[1,1].bar(x[2:], (y4_2), 0.35, color=['blue'],label='Best Models')
#        axes[1,1].set_ylim(0, 22)
        axes[1,1].legend(loc='best')
        axes[1,1].set_ylabel('RMSE Index')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(['ARIMA','RW', results[results.FH=='30'].FBM_nickname.item(), results[results.FH=='30'].SBM_nickname.item()])
    
        axes[2,0].set_title('Forecast Horizon: 60')
        axes[2,0].bar(x[:2], (y5_1), 0.35, color=['red'], label='Benchmark')
        axes[2,0].bar(x[2:], (y5_2), 0.35, color=['blue'],label='Best Models')
#        axes[2,0].set_ylim(0, 22)
        axes[2,0].legend(loc='best')
        axes[2,0].set_ylabel('RMSE Index')
        axes[2,0].set_xticks(x)
        axes[2,0].set_xticklabels(['ARIMA','RW', results[results.FH=='60'].FBM_nickname.item(), results[results.FH=='60'].SBM_nickname.item()])
    
        axes[2,1].set_title('Forecast Horizon: 90')
        axes[2,1].bar(x[:2], (y6_1), 0.35, color=['red'], label='Benchmark')
        axes[2,1].bar(x[2:], (y6_2), 0.35, color=['blue'],label='Best Models')
#        axes[2,1].set_ylim(0, 22)
        axes[2,1].legend(loc='best')
        axes[2,1].set_ylabel('RMSE Index')
        axes[2,1].set_xticks(x)
        axes[2,1].set_xticklabels(['ARIMA','RW', results[results.FH=='90'].FBM_nickname.item(), results[results.FH=='90'].SBM_nickname.item()])
    
        fig2.tight_layout()
#        plt.savefig('barplot2_v2.png') 
        plt.show()
        
    
    return None



if __name__== "__main3__":
   
    
    #correlations_plots(dados, y)    
    #adf_test(y_treino)
    #adf_test(y_teste)
    #adf_test(y)
    
    """
    Não podemos rejeitas a hipótese nula a 10%, 
    então a série é não estacionária (série toda e separada).
    """
    
    #y_treino_diff, y_teste_diff = first_diff(y_treino, y_teste)
    
    #adf_test(y_treino_diff_aux)     
    #adf_test(y_teste_diff_aux)     
    
    """
    1a diferença é estacionária -> i=1
    """
  
 
    periodo, treino, teste, dados = create_database()
    
    
    
    #make correlation and summary analysis
    
    # cor_df = dados[['CEE_BR_TOT'] + [col for col in dados.columns if 'CEE_BR_TOT' not in col]].corr().reset_index().rename(columns = {'index':'VAR'})#.dropna(axis=1, how='all')
    # cor_df.to_excel('Correlation Matrix.xlsx', index=False)


    # summary_df = dados[['CEE_BR_TOT'] + [col for col in dados.columns if 'CEE_BR_TOT' not in col]].describe().reset_index().rename(columns = {'index':'STATISTIC'})
    # summary_df.to_excel('Summary Statistics.xlsx', index=False)

    
    train_ratio=.7
    forecast_horizon=1
    
        

    y, X, X_treino, X_teste, y_treino, y_teste, treino, teste = set_database(forecast_horizon, dados, periodo, train_ratio)
    X_treino, X_teste = set_adjusted_temperature(X_treino, X_teste)
    
        
    models_list = ['LASSO', 'LARS', 'LASSOLARS', 'RIDGE', 'EN','OLS','ARIMA','RF', 'RW']
    
    # models_list = ['LASSO','OLS','RW']
    
    train_size = round(len(X)*train_ratio)
    X_treino, X_teste,  y_treino, y_teste, df_metrics, coef_dict, df_temp  = make_forecast(X,y,train_size, forecast_horizon, models_list, jobs=-2)


    for forecast_horizon2 in [1,7,15,30,60,90]:
        y, X, X_treino, X_teste, y_treino, y_teste, treino, teste = set_database(forecast_horizon2, dados, periodo, train_ratio)        
        if forecast_horizon==90:
        
            df_temp['fh_{}'.format(str(forecast_horizon2))] = 10/162
        else:
            df_temp['fh_{}'.format(str(forecast_horizon2))] = (len(y_teste) - forecast_horizon2)/162
        df_temp['time_{}'.format(str(forecast_horizon2))] = df_temp['Time'] * df_temp['fh_{}'.format(str(forecast_horizon2))]

    
    def sum_time(x):    
        return x['time_1'] + x['time_7'] + x['time_15'] + x['time_30'] + x['time_60'] + x['time_90']
    
    
    df_temp['time_tot'] = df_temp.apply(lambda x: sum_time(x), axis=1)
    df_temp_final = df_temp[[x for x in df_temp.columns if 'time_' in x  or 'Model' in x]]
    for col in [x for x in df_temp_final.columns if 'Model' not in x] :
        fh = col.split('_')[-1]
        df_temp_final['time_hour_{}'.format(fh)] = df_temp[col].apply(lambda x: x/np.timedelta64(1, 'h'))
        df_temp_final['time_minute_{}'.format(fh)] = df_temp[col].apply(lambda x: x/np.timedelta64(1, 'm'))


    df_temp_final = df_temp_final[[x for x in df_temp_final.columns if 'time_hour' in x or 'time_minute' in x or 'Model' in x]]
        
    df_temp2 = pd.DataFrame({'Tempo Total (horas)' : [df_temp['time_tot'].sum()/np.timedelta64(1, 'h')],
                             'Tempo Total (minutos)' : [df_temp['time_tot'].sum()/np.timedelta64(1, 'm')]})

    df_temp3 = df_temp.copy()
    for col in [x for x in df_temp3.columns if 'time_' in x]:
        df_temp3[col] = df_temp3[col].astype(str)
        
    df_temp3 = df_temp3[[x for x in df_temp3.columns if 'time_' in x or 'Model' in x]]

    
    writer = pd.ExcelWriter('tempo_modelos.xlsx')
    df_temp_final.to_excel(writer,'modelos', index=False)
    df_temp2.to_excel(writer,'agregado', index=False)
    df_temp3.to_excel(writer,'tempo alternativo', index=False)

    writer.save()
    

    df_metrics = df_metrics.reset_index().rename(columns = {'index':'model'})
    
    # Wrap data and compute the Mean Absolute Error
    data = pd.DataFrame()
    for model in models_list:
        if model!='RW':
            data[model] = np.abs(y_teste['Ypred_'+model] - y_teste['CEE_BR_TOT'])
        
    
    mcs = ModelConfidenceSet(data, 0.1,3, 1000).run()
    mcs.included
    mcs_df=mcs.pvalues.to_frame().reset_index().rename(columns={0:'mcs', 'index':'model'})
    df_metrics = df_metrics.merge(mcs_df, how='left', on='model')
    
    df_metrics = df_metrics.fillna(0)
    df_metrics = df_metrics.sort_values(['mcs','MSE_test'], ascending=[False, True])
    # df_metrics = df_metrics.sort_values(['MSE_test','mcs'], ascending=[True, False])

    df_coef = pd.DataFrame()
    for i,model in enumerate(models_list):
        if model in coef_dict.keys():
            df_coef_aux = coef_dict[model]['coef'].mean(axis=1).to_frame().rename(columns={0:model})
            if i==0:
                df_coef = df_coef_aux.copy()
            else:
                df_coef = df_coef.merge(df_coef_aux, how='left',left_index=True,right_index=True)
        
    coef_sum_df=sum_df(df_coef)
    coef_sum_df = coef_sum_df[[x for x in df_metrics.model.to_list() if x in coef_sum_df.columns]]
    
    cols = list(coef_sum_df.columns[:4])
    for col in cols:
        coef_sum_df[cols]= coef_sum_df[cols].apply(lambda x: round(x,2))
    coef_sum_df['combined'] = coef_sum_df[cols].apply(lambda row: '&'.join(row.values.astype(str)), axis=1)

    print(cols)
    print(coef_sum_df['combined'])
    
    plot_forecast(y, y_treino,y_teste,forecast_horizon, df_metrics)
    # barplot_aggregate(flag=0)
    
    dados=dados.reset_index()
    dados['year'] = dados['index'].dt.year.astype(str)
    dados['month'] = dados['index'].dt.month.astype(str)

    dados1 = dados[dados.year=='2017'].rename(columns={'CEE_BR_TOT':'mWh'})
    dados1['mWh'] = dados1['mWh']/1e6
    dados2 = dados[dados.year=='2018'].rename(columns={'CEE_BR_TOT':'mWh'})
    dados2['mWh'] = dados2['mWh']/1e6
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    sns.boxplot(ax=axs[0], x="month", y="mWh", data=dados1)
    axs[0].set_title('Average consumption 2017')
    sns.boxplot(ax=axs[1], x="month", y="mWh", data=dados2)
    axs[1].set_title('Average consumption 2018')
    plt.tight_layout()
    plt.savefig('boxplot.png')   

    plt.show()
    
    
    save_results(y_teste, coef_sum_df,df_metrics, forecast_horizon)



    # 1) LASSO LARS / RF / ELASTIC NET / LASSO
    # 2) RF/ ELASTIC NET/ RIDGE / LASSO LARS
    # 3) RF / RIDGE/ ELASTIC NET /LASSO LARS / 


   # R2_list = [R21,0,R23,R24, R25,R26,R28,R29,R210] 
    # EQM_list = [EQM1,EQM2,EQM3,EQM4,EQM5,EQM6,EQM8,EQM9,EQM10]
    # resid_list = [resid1,resid2,resid3,resid4, resid5,resid6,resid8,resid9,resid10]
    # accuracy_list = [accuracy_1,accuracy_2,accuracy_3,accuracy_4,accuracy_5,accuracy_6,accuracy_8,accuracy_9,accuracy_10]
    # R2_test_list = [R2_1_teste,0,R2_3_teste,R2_4_teste, R2_5_teste,R2_6_teste,R2_8_teste,R2_9_teste,R2_10_teste]


    # previsao, previsao_aux=previsao_df(R2_list,EQM_list,resid_list,accuracy_list,R2_test_list)





# def arima_classificator(y_treino, y_teste): 
    
#     arima_dict={}
#     for p in range (1,6):
#         arima_dict.update({p:{}})
#         for q in range (1,6):
 
#             try:
#                 model = ARIMA(y_treino, order=(p,1,q))
#                 model_fit = model.fit()
            
#                 y_predictions = model_fit.predict(start=len(y_treino), end=len(y_treino)+len(y_teste)-1, dynamic=False)
#                 EQM= mean_squared_error(y_teste, y_predictions)
#                 resid = np.sqrt(EQM) 
                
#                 arima_dict[p].update({q:resid}) 
#             except:
#                 pass
           
#     arima_min_list=[]
#     for i in  arima_dict.keys():
#         arima_min_list.append((i,min(arima_dict[i].items(), key=lambda x: x[1])))
              
#         q=min(arima_min_list, key=lambda x: x[1][1])[1][0]
#         p=min(arima_min_list, key=lambda x: x[1][1])[0]
           
#     print('ARIMA tem os parametros p,q:', p,q) 
    
#     return p,q



# def run_arima(p,q, y_treino, y_teste):
    
#     try:
#         model1 = ARIMA(y_treino, order=(p,1,q))
#         model_fit1 = model1.fit()
#     except:
#         model1 = ARIMA(y_treino, order=(p-1,1,q-1))
#         model_fit1 = model1.fit()    
    
    
#     print('Lag: %s' % model_fit1.k_ar)
#     print('Coefficients: %s' % model_fit1.params)
#     R21=r2_score(y_treino, model_fit1.predict(start=1, end=len(y_treino), dynamic=False))
    
#     # make predictions
#     y_predictions1 = model_fit1.predict(start=len(y_treino), end=len(y_treino)+len(y_teste)-1, dynamic=False)
#     EQM1= mean_squared_error(y_teste, y_predictions1)
#     resid1 = np.sqrt(EQM1)   
#     print('Test MSE: %.3f' % EQM1, resid1)
#     print(model_fit1.summary())
    
#     accuracy_1 = r2_score(y_teste, y_predictions1)
#     R2_1_teste = accuracy_1
#     print ('accuracy, R2_teste: %.3f' % accuracy_1, R2_1_teste) 

#     return R21, y_predictions1, EQM1, resid1, accuracy_1, R2_1_teste



# def auto_arima(y_treino, y_teste):
    
#     from pyramid.arima import auto_arima
#     model1 = auto_arima(y_treino, start_p=1, start_q=1,
#                                max_p=3, max_q=3, m=12,
#                                start_P=0, seasonal=True,
#                                d=1, D=1, trace=True,
#                                error_action='ignore',  
#                                suppress_warnings=True, 
#                                stepwise=True)
    
#     model_fit1=model1.fit(y_treino)
#     print('Lag: %s' % model_fit1.k_ar)
#     print('Coefficients: %s' % model_fit1.params)
#     R21=r2_score(y_treino, model_fit1.predict(start=1, end=len(y_treino), dynamic=False))
    
#     y_predictions1 = model_fit1.predict(start=len(y_treino), end=len(y_treino)+len(y_teste)-1, dynamic=False)
#     EQM1= mean_squared_error(y_teste, y_predictions1)
#     resid1 = np.sqrt(EQM1)   
#     print('Test MSE: %.3f' % EQM1, resid1)
#     print(model_fit1.summary())
    
#     accuracy_1 = r2_score(y_teste, y_predictions1)
#     R2_1_teste = accuracy_1
#     print ('accuracy, R2_teste: %.3f' % accuracy_1, R2_1_teste) 
    
#     return R21, y_predictions1, EQM1, resid1, accuracy_1, R2_1_teste


# #Random Walk (2)
    
# def run_randomwalk(y_treino, y_teste, teste, forecastHorizon):
    
#     y_pred2=[]    
#     for i in range(int(len(y_teste)/forecastHorizon)):
#         for j in y_treino[-forecastHorizon:]:              
#             y_pred2.append(j)
                                  
#     if (len(y_teste)%forecastHorizon!=0):
#         y_pred2_aux=y_pred2[:len(y_teste)%forecastHorizon]
#         for i in y_pred2_aux:
#             y_pred2.append(i)

#     y_predictions2 = pd.DataFrame(data=y_pred2, index=teste, columns=['y_pred'])
    
#     EQM2 = mean_squared_error(y_teste, y_predictions2) #EQM
#     resid2 = np.sqrt(EQM2) #Resíduo
#     print('Test MSE, Residual: %.3f' % EQM2, resid2)
    
#     accuracy_2 = r2_score(y_teste, y_predictions2)
#     R2_2_teste = sm.OLS(y_teste,X_teste).fit().rsquared
#     print ('accuracy, R2_teste: %.3f' % accuracy_2, R2_2_teste)
 
 
#     return y_predictions2, EQM2, resid2, accuracy_2, R2_2_teste



# #OLS    (3)            

# def run_OLS(X_treino, X_teste, y_treino, y_teste, treino, teste, indice):
    
#     model3 = sm.OLS(y_treino,X_treino)                  #modelo
#     model_fit3 = model3.fit() 
#     print (model_fit3.summary())                        #sumário do modelo
#     coef3=model_fit3.params
    
#     R23=model_fit3.rsquared
        
#     # make predictions
#     y_predictions3 = pd.DataFrame(index=teste, columns=['Data','y_pred'])
#     ypred3=[]    
#     for i in range(int(len(teste)/forecastHorizon)):
#         ypred3.append(model_fit3.predict(X_teste[i*forecastHorizon:(i+1)*forecastHorizon]))   
#     if (len(teste)%forecastHorizon!=0):        
#         ypred3.append(model_fit3.predict(X_teste[(int(len(teste)/forecastHorizon)*forecastHorizon):len(teste)]))      
    
#     y_predictions3=pd.concat(ypred3)
#     y_predictions3=pd.DataFrame({'y_pred':y_predictions3}, index=indice)
#     y_predictions3['Data']=indice
#     y_predictions3=y_predictions3.set_index('Data')
    
    
#     EQM3 = mean_squared_error(y_teste, y_predictions3)    #EQM
#     resid3 = np.sqrt(EQM3)                                #Resíduo
    
#     print('Test MSE, Residual: %.3f' % EQM3, resid3)
        
#     accuracy_3 = r2_score(y_teste, y_predictions3)
#     R2_3_teste = sm.OLS(y_teste,X_teste).fit().rsquared
#     print ('accuracy, R2_teste: %.3f' % accuracy_3, R2_3_teste)
    
#     return coef3, R23, y_predictions1, EQM1, resid1, accuracy_1, R2_1_teste




# #2)Linear Regression (30)
 
# def run_linearregression(X_treino, X_teste, y_treino, y_teste, treino, teste, indice):
    
#     model30= LinearRegression()
#     model30_fit=model30.fit(X_treino, y_treino)
#     print(model30.score(X_treino, y_treino))                        #R2 fora da amostra
#     print(model30.coef_)                                            #coeficientes
#     coef30=np.transpose(model30.coef_)
    
#     R230=model30.score(X_treino, y_treino)    
        
#     # make predictions
#     y_predictions30 = pd.DataFrame(index=teste, columns=['y_pred'])
#     ypred30=[]    
#     for i in range(int(len(teste)/forecastHorizon)):
#         ypred30.append(pd.Series(model30_fit.predict(X_teste[i*forecastHorizon:(i+1)*forecastHorizon])))   
#     if (len(teste)%forecastHorizon!=0):        
#         ypred30.append(pd.Series(model30_fit.predict(X_teste[(int(len(teste)/forecastHorizon)*forecastHorizon):len(teste)])))      
    
#     y_predictions30=pd.concat(ypred30)
#     y_predictions30=pd.DataFrame({'y_pred':y_predictions30})
#     y_predictions30['Data']=indice
#     y_predictions30=y_predictions30.set_index('Data')
    
    
#     EQM30 = mean_squared_error(y_teste, y_predictions30)      #EQM
#     resid30 = np.sqrt(EQM30)                                #Residuo
#     print('Test MSE, residuo: %.3f' % EQM30,resid30)
    
#     accuracy_30 = r2_score(y_teste, y_predictions30)
#     R2_30_teste = model30.score(X_teste, y_teste)  
#     print ('accuracy, R2_teste: %.3f' % accuracy_30, R2_30_teste)
    
#     return coef30, R230, y_predictions30, EQM30, resid30, accuracy_30, R2_30_teste


# #1)Lasso normal - (4)



# def lasso_classificator(X_treino, X_teste, y_treino, y_teste): 
    
#     alpha_lasso_lista = [1e-15, 1e-10, 1e-8, 1e-5, 1e-3,1e-2,1e-1, 1, 5, 10,20]
#     lasso_dict={}
#     for alpha in alpha_lasso_lista:    
#         model = linear_model.Lasso(alpha=alpha, copy_X=True, fit_intercept=True, max_iter=1000,
#                                    normalize=True, positive=False, precompute=False, random_state=None,
#                                    selection='cyclic', tol=0.0001, warm_start=False)
#         model_fit=model.fit(X_treino,y_treino)
        
#         y_predictions = model_fit.predict(X_teste)
#         lasso_dict.update({alpha:np.sqrt(mean_squared_error(y_teste, y_predictions))})

#     alpha=min(lasso_dict.items(), key=lambda x: x[1])[0]
          
#     print('O alpha que apresente o menor erro de previsão é igual a %.1f' % alpha) 
    
#     return alpha



# def run_lasso(alpha, X_treino, X_teste, y_treino, y_teste, treino, teste, indice):
    
#     print(Lasso().get_params())
    
#     model4 = linear_model.Lasso(alpha=alpha, copy_X=True, fit_intercept=True, max_iter=1000,
#                                 normalize=True, positive=False, precompute=False, random_state=None,
#                                 selection='cyclic', tol=0.0001, warm_start=False)
#     model_fit4=model4.fit(X_treino,y_treino)
    
#     coef4=model4.coef_
#     R24 = model_fit4.score(X_treino,y_treino)     
    
#     # make predictions
#     y_predictions4 = pd.DataFrame(index=teste, columns=['y_pred'])
#     ypred4=[]    
#     for i in range(int(len(teste)/forecastHorizon)):
#         ypred4.append(pd.Series(model_fit4.predict(X_teste[i*forecastHorizon:(i+1)*forecastHorizon])))   
#     if (len(teste)%forecastHorizon!=0):        
#         ypred4.append(pd.Series(model_fit4.predict(X_teste[(int(len(teste)/forecastHorizon)*forecastHorizon):len(teste)])))      
    
#     y_predictions4=pd.concat(ypred4)
#     y_predictions4=pd.DataFrame({'y_pred':y_predictions4})
#     y_predictions4['Data']=indice
#     y_predictions4=y_predictions4.set_index('Data')
    
#     EQM4 = mean_squared_error(y_teste, y_predictions4)    #EQM
#     resid4 = np.sqrt(EQM4)    
#     print('Test MSE, residuo: %.3f' % EQM4,resid4)
    
#     accuracy_4 = r2_score(y_teste, y_predictions4)
#     R2_4_teste = model_fit4.score(X_teste, y_teste) 
#     print ('accuracy, R2_teste: %.3f' % accuracy_4, R2_4_teste) 
    
#     return coef4, R24, y_predictions4, EQM4, resid4, accuracy_4, R2_4_teste



# #2) Lasso CV - (40)

# def lassoCV_classificator(X_treino, X_teste, y_treino, y_teste): 
  
#     start_time=time.time()
    
#     eps_LassoCV_list = [1e-10,1e-5,1,5]
#     cv_LassoCV_list=[3,10]    

#     lassoCV_dict={}
#     for eps in eps_LassoCV_list:
#         lassoCV_dict.update({eps:{}})
#         for cv in cv_LassoCV_list:
            
#             model= LassoCV(fit_intercept=True, verbose=False, max_iter=500, 
#                        normalize=True, cv=cv, eps=eps, copy_X=True, 
#                        positive=False) 
#             model_fit=model.fit(X_treino, y_treino)

#             y_predictions = model_fit.predict(X_teste)
            
#             lassoCV_dict[eps].update({cv:np.sqrt(mean_squared_error(y_teste, y_predictions))})
           
#     lassoCV_min_list=[]
#     for i in  lassoCV_dict.keys():
#         lassoCV_min_list.append((i,min(lassoCV_dict[i].items(), key=lambda x: x[1])))

#         eps=min(lassoCV_min_list, key=lambda x: x[1][1])[0]              
#         cv=min(lassoCV_min_list, key=lambda x: x[1][1])[1][0]
             
#     print('O epsilson e o cv que apresentam o menor erro de previsão são iguais a %.1f' % eps, cv) 
#     print ("My program took", time.time() - start_time, "seconds to run") 
   
#     return eps, cv



# def run_lassoCV(eps, cv, X_treino, X_teste, y_treino, y_teste, treino, teste, indice):
    
#     print(LassoCV().get_params()) 

#     model40= LassoCV(fit_intercept=True, verbose=False, max_iter=500, normalize=True, 
#                     cv=cv, eps=eps, copy_X=True, positive=False) 
    
#     model40_fit = model40.fit(X_treino,y_treino)
    
#     print(model40_fit.coef_) 
#     coef40=model40_fit.coef_ 
#     R240 = model40_fit.score(X_treino, y_treino) 
    
    
#     # make predictions
#     y_predictions40 = pd.DataFrame(index=teste, columns=['y_pred'])
#     ypred40=[]    
#     for i in range(int(len(teste)/forecastHorizon)):
#         ypred40.append(pd.Series(model40_fit.predict(X_teste[i*forecastHorizon:(i+1)*forecastHorizon])))   
#     if (len(teste)%forecastHorizon!=0):        
#         ypred40.append(pd.Series(model40_fit.predict(X_teste[(int(len(teste)/forecastHorizon)*forecastHorizon):len(teste)])))      
    
#     y_predictions40=pd.concat(ypred40)
#     y_predictions40=pd.DataFrame({'y_pred':y_predictions40})
#     y_predictions40['Data']=indice
#     y_predictions40=y_predictions40.set_index('Data')
    
#     EQM40 = mean_squared_error(y_teste, y_predictions40)
#     resid40 = np.sqrt(EQM40)
#     print('Test MSE: %.3f' % EQM40,resid40)
    
#     accuracy_40 = r2_score(y_teste, y_predictions40)
#     R2_40_teste = model40_fit.score(X_teste, y_teste) 
#     print ('accuracy, R2_teste: %.3f' % accuracy_40, R2_40_teste)

#     return coef40, R240, y_predictions40, EQM40, resid40, accuracy_40, R2_40_teste

       


# #1) Lars - 5


# def lars_classificator(X_treino, X_teste, y_treino, y_teste): 
  
#     start_time=time.time()
    
#     eps_list = [1e-15, 1e-10, 1e-8, 1e-5,1e-3,1e-2, 1,2,5, 10,20]
#     nzero_coef_list=[1,10,50,100,500]    

#     lars_dict={}
#     for eps in eps_list:
#         lars_dict.update({eps:{}})
#         for nzero in nzero_coef_list:
            
#             model=Lars(fit_intercept=True, verbose=False, normalize=True, 
#                        n_nonzero_coefs=nzero, eps=eps, 
#                        copy_X=True, fit_path=True, positive=False) 
#             model_fit=model.fit(X_treino, y_treino)
        
#             y_predictions = model_fit.predict(X_teste)
            
#             lars_dict[eps].update({nzero:np.sqrt(mean_squared_error(y_teste, y_predictions))})
           
#     lars_min_list=[]
#     for i in  lars_dict.keys():
#         lars_min_list.append((i,min(lars_dict[i].items(), key=lambda x: x[1])))
              
#         eps=min(lars_min_list, key=lambda x: x[1][1])[0]
#         nZeroCoef=min(lars_min_list, key=lambda x: x[1][1])[1][0]
             
#     print('O epsilson e o cv que apresentam o menor erro de previsão são iguais a %.1f' % eps, cv) 
#     print ("My program took", time.time() - start_time, "seconds to run") 
   
#     return eps, nZeroCoef


# def run_lars(eps, nZeroCoef, X_treino, X_teste, y_treino, y_teste, treino, teste, indice):

#     print(Lars().get_params())

#     model5=Lars(fit_intercept=True, verbose=False, normalize=True, 
#                 n_nonzero_coefs=int(nZeroCoef), eps=eps, copy_X=True, fit_path=True, 
#                 positive=False) 
#     model5_fit=model5.fit(X_treino, y_treino)    
#     coef5=model5_fit.coef_ 
#     R25 = model5_fit.score(X_treino, y_treino) 
    
#     # make predictions
#     y_predictions5 = pd.DataFrame(index=teste, columns=['y_pred'])
#     ypred5=[]    
#     for i in range(int(len(teste)/forecastHorizon)):
#         ypred5.append(pd.Series(model5_fit.predict(X_teste[i*forecastHorizon:(i+1)*forecastHorizon])))   
#     if (len(teste)%forecastHorizon!=0):        
#         ypred5.append(pd.Series(model5_fit.predict(X_teste[(int(len(teste)/forecastHorizon)*forecastHorizon):len(teste)])))      
    
#     y_predictions5=pd.concat(ypred5)
#     y_predictions5=pd.DataFrame({'y_pred':y_predictions5})
#     y_predictions5['Data']=indice
#     y_predictions5=y_predictions5.set_index('Data')
    
#     EQM5 = mean_squared_error(y_teste, y_predictions5)
#     resid5 = np.sqrt(EQM5)
#     print('Test MSE: %.3f' % EQM5,resid5)
    
#     accuracy_5 = r2_score(y_teste, y_predictions5)
#     R2_5_teste = model5_fit.score(X_teste, y_teste) 
#     print ('accuracy, R2_teste: %.3f' % accuracy_5, R2_5_teste) 

#     return coef5, R25, y_predictions5, EQM5, resid5, accuracy_5, R2_5_teste


    
# #2) Lasso Lars - (6))

# def lassoLars_classificator(X_treino, X_teste, y_treino, y_teste): 
  
#     start_time=time.time()
    
#     eps_LL_list=[1e-15, 1e-8,1e-3,1,20]
#     alpha_LL_lista=[1e-10,1e-8,1,10]    

#     LL_dict={}
#     for eps in eps_LL_list:
#         LL_dict.update({eps:{}})
#         for alpha in alpha_LL_lista:
            
#             model=LassoLars(alpha=alpha,fit_intercept=True, verbose=False, normalize=True, 
#                             max_iter=500, eps=eps, copy_X=True,fit_path=True, positive=False) 
#             model_fit=model.fit(X_treino, y_treino)
        
#             y_predictions = model_fit.predict(X_teste)
            
#             LL_dict[eps].update({alpha:np.sqrt(mean_squared_error(y_teste, y_predictions))})
           
#     LL_min_list=[]
#     for i in  LL_dict.keys():
#         LL_min_list.append((i,min(LL_dict[i].items(), key=lambda x: x[1])))
              
#         eps=min(LL_min_list, key=lambda x: x[1][1])[0]
#         alpha=min(LL_min_list, key=lambda x: x[1][1])[1][0]
             
#     print('O epsilson e o cv que apresentam o menor erro de previsão são iguais a %.1f' % eps, cv) 
#     print ("My program took", time.time() - start_time, "seconds to run") 
   
#     return eps, alpha


# def run_lassoLars(eps, alpha, X_treino, X_teste, y_treino, y_teste, treino, teste, indice):
    
#     print(LassoLars().get_params()) 

#     model6=LassoLars( alpha=alpha, fit_intercept=True, verbose=False, normalize=True, 
#                      max_iter=500, eps=eps, copy_X=True,fit_path=True, positive=False) 
#     model6_fit=model6.fit(X_treino, y_treino)
#     coef6=model6_fit.coef_
    
#     R26 = model6_fit.score(X_treino, y_treino) 
    
    
#     # make predictions
#     y_predictions6 = pd.DataFrame(index=teste, columns=['y_pred'])
#     ypred6=[]    
#     for i in range(int(len(teste)/forecastHorizon)):
#         ypred6.append(pd.Series(model6_fit.predict(X_teste[i*forecastHorizon:(i+1)*forecastHorizon])))   
#     if (len(teste)%forecastHorizon!=0):        
#         ypred6.append(pd.Series(model6_fit.predict(X_teste[(int(len(teste)/forecastHorizon)*forecastHorizon):len(teste)])))      
    
#     y_predictions6=pd.concat(ypred6)
#     y_predictions6=pd.DataFrame({'y_pred':y_predictions6})
#     y_predictions6['Data']=indice
#     y_predictions6=y_predictions6.set_index('Data')
    
#     EQM6 = mean_squared_error(y_teste, y_predictions6)
#     resid6 = np.sqrt(EQM6)
#     print('Test MSE: %.3f' % EQM6,resid6)
    
#     accuracy_6 = r2_score(y_teste, y_predictions6)
#     R2_6_teste = model6_fit.score(X_teste, y_teste) 
#     print ('accuracy, R2_teste: %.3f' % accuracy_6, R2_6_teste) 

#     return coef6, R26, y_predictions6, EQM6, resid6, accuracy_6, R2_6_teste




# #3) Lasso Lars CV - (7)

  
# def lassoLarsCV_classificator(X_treino, X_teste, y_treino, y_teste): 
  
#     start_time=time.time()
    
#     eps_LassoLarsCV_list = [1e-8, 1, 10]
#     cv_lassoLarsCV_list = [3,10,30]   

#     LLCV_dict={}
#     for eps in eps_LassoLarsCV_list:
#         LLCV_dict.update({eps:{}})
#         for cv in cv_lassoLarsCV_list:
            
#             model= LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, 
#                                normalize=True, cv=cv, max_n_alphas=1000, eps=eps, copy_X=True, 
#                                positive=False) 
#             model_fit=model.fit(X_treino, y_treino)
        
#             y_predictions = model_fit.predict(X_teste)
            
#             LLCV_dict[eps].update({cv:np.sqrt(mean_squared_error(y_teste, y_predictions))})
           
#     LLCV_min_list=[]
#     for i in  LLCV_dict.keys():
#         LLCV_min_list.append((i,min(LLCV_dict[i].items(), key=lambda x: x[1])))
              
#         eps=min(LLCV_min_list, key=lambda x: x[1][1])[0]
#         cv=min(LLCV_min_list, key=lambda x: x[1][1])[1][0]
             
#     print('O epsilson e o cv que apresentam o menor erro de previsão são iguais a %.1f' % eps, cv) 
#     print ("My program took", time.time() - start_time, "seconds to run") 
   
#     return eps, cv


# def run_lassoLarsCV(eps, cv, X_treino, X_teste, y_treino, y_teste, treino, teste, indice):
    
#     print(LassoLarsCV().get_params())

#     model7= LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, 
#                         normalize=True, cv=cv, max_n_alphas=1000, eps=eps, 
#                         copy_X=True, positive=False) 
#     model7_fit = model7.fit(X_treino,y_treino)
#     print(model7_fit.coef_) 
#     coef7=model7_fit.coef_ 
#     R27 = model7_fit.score(X_treino, y_treino) 
    
#     # make predictions
#     y_predictions7 = pd.DataFrame(index=teste, columns=['y_pred'])
#     ypred7=[]    
#     for i in range(int(len(teste)/forecastHorizon)):
#         ypred7.append(pd.Series(model7_fit.predict(X_teste[i*forecastHorizon:(i+1)*forecastHorizon])))   
#     if (len(teste)%forecastHorizon!=0):        
#         ypred7.append(pd.Series(model7_fit.predict(X_teste[(int(len(teste)/forecastHorizon)*forecastHorizon):len(teste)])))      
    
#     y_predictions7=pd.concat(ypred7)
#     y_predictions7=pd.DataFrame({'y_pred':y_predictions7})
#     y_predictions7['Data']=indice
#     y_predictions7=y_predictions7.set_index('Data')
    
    
#     EQM7 = mean_squared_error(y_teste, y_predictions7)
#     resid7 = np.sqrt(EQM7)
#     print('Test MSE: %.3f' % EQM7,resid7)
    
#     accuracy_7 = r2_score(y_teste, y_predictions7)
#     R2_7_teste = model7_fit.score(X_teste, y_teste) 
#     print ('accuracy, R2_teste: %.3f' % accuracy_7, R2_7_teste)
    
#     return coef7, R27, y_predictions7, EQM7, resid7, accuracy_7, R2_7_teste





# #Ridge Regression - (8)

# def ridge_classificator(X_treino, X_teste, y_treino, y_teste): 
    
#     alpha_ridge_lista = [1e-15, 1e-10, 1e-8, 1e-5, 1e-3,1e-2,1e-1, 1, 5, 10,20]
#     ridge_dict={}
#     for alpha in alpha_ridge_lista:    
#         model = Ridge(alpha=alpha, fit_intercept=True, normalize=True, copy_X=True, 
#                       max_iter=None, tol=0.001, random_state=None)
#         model_fit=model.fit(X_treino,y_treino)
    
#         y_predictions = model_fit.predict(X_teste)
#         ridge_dict.update({alpha:np.sqrt(mean_squared_error(y_teste, y_predictions))})

#     alpha=min(ridge_dict.items(), key=lambda x: x[1])[0]
          
#     print('O alpha que apresente o menor erro de previsão é igual a %.1f' % alpha) 
    
#     return alpha


# def run_ridge(alpha, X_treino, X_teste, y_treino, y_teste, treino, teste, indice):
    
#     print(Ridge().get_params())
      
#     model8 = Ridge(alpha=alpha, fit_intercept=True, normalize=True, copy_X=True, 
#                    max_iter=None, tol=0.001, random_state=None)
#     model8_fit=model8.fit(X_treino, y_treino)
    
#     coef8=np.transpose(model8_fit.coef_)
#     R28 = model8_fit.score(X_treino, y_treino) 
    
#     # make predictions
#     y_predictions8 = pd.DataFrame(index=teste, columns=['y_pred'])
#     ypred8=[]    
#     for i in range(int(len(teste)/forecastHorizon)):
#         ypred8.append(pd.Series(model8_fit.predict(X_teste[i*forecastHorizon:(i+1)*forecastHorizon])))   
#     if (len(teste)%forecastHorizon!=0):        
#         ypred8.append(pd.Series(model8_fit.predict(X_teste[(int(len(teste)/forecastHorizon)*forecastHorizon):len(teste)])))      
    
#     y_predictions8=pd.concat(ypred8)
#     y_predictions8=pd.DataFrame({'y_pred':y_predictions8})
#     y_predictions8['Data']=indice
#     y_predictions8=y_predictions8.set_index('Data')
    
    
#     EQM8 = mean_squared_error(y_teste, y_predictions8)
#     resid8 = np.sqrt(EQM8)
#     print('Test MSE: %.3f' % EQM8,resid8)
    
#     accuracy_8 = r2_score(y_teste, y_predictions8)
#     R2_8_teste = model8_fit.score(X_teste, y_teste) 
#     print ('accuracy, R2_teste: %.3f' % accuracy_8, R2_8_teste) 

#     return coef8, R28, y_predictions8, EQM8, resid8, accuracy_8, R2_8_teste


# #ElasticNet 
# #1) ElasticNet sem CV - (90)


# def ElasticNet_classificator(X_treino, X_teste, y_treino, y_teste): 
  
#     start_time=time.time()
    
#     alpha_EN_list = [1e-15, 1e-5,1e-2, 1, 5,20]
#     l1_ratio_list=[0.01,0.25,0.5,0.75,0.99]   

#     elasticNet_dict={}
#     for alpha in alpha_EN_list:
#         elasticNet_dict.update({alpha:{}})
#         for l1 in l1_ratio_list:
            
#             model=ElasticNet(alpha=alpha, l1_ratio=l1, fit_intercept=True, normalize=False, 
#                              precompute=False, max_iter=1000, copy_X=True, tol=0.0001, 
#                              warm_start=False, positive=False, random_state=None) 
#             model_fit=model.fit(X_treino, y_treino)
        
#             y_predictions = model_fit.predict(X_teste)
            
#             elasticNet_dict[alpha].update({l1:np.sqrt(mean_squared_error(y_teste, y_predictions))})
           
#     elasticNet_min_list=[]
#     for i in  elasticNet_dict.keys():
#         elasticNet_min_list.append((i,min(elasticNet_dict[i].items(), key=lambda x: x[1])))
              
#         alpha=min(elasticNet_min_list, key=lambda x: x[1][1])[0]
#         l1=min(elasticNet_min_list, key=lambda x: x[1][1])[1][0]
             
#     print('O epsilson e o cv que apresentam o menor erro de previsão são iguais a %.1f' % eps, cv) 
#     print ("My program took", time.time() - start_time, "seconds to run") 
   
#     return alpha, l1


# def run_elasticNet(alpha, l1, X_treino, X_teste, y_treino, y_teste, treino, teste, indice):
    
#     print(ElasticNet().get_params())

#     model90=ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, normalize=False, 
#                        precompute=False, max_iter=1000, copy_X=True, tol=0.0001, 
#                        warm_start=False, positive=False, random_state=None) 
#     model90_fit=model90.fit(X_treino, y_treino)
    
    
#     R290 = model90.score(X_treino, y_treino) 
#     coef90=model90.coef_
    
#     # make predictions
#     y_predictions90 = pd.DataFrame(index=teste, columns=['y_pred'])
#     ypred90=[]    
#     for i in range(int(len(teste)/forecastHorizon)):
#         ypred90.append(pd.Series(model90_fit.predict(X_teste[i*forecastHorizon:(i+1)*forecastHorizon])))   
#     if (len(teste)%forecastHorizon!=0):        
#         ypred90.append(pd.Series(model90_fit.predict(X_teste[(int(len(teste)/forecastHorizon)*forecastHorizon):len(teste)])))      
    
#     y_predictions90=pd.concat(ypred90)
#     y_predictions90=pd.DataFrame({'y_pred':y_predictions90})
#     y_predictions90['Data']=y_predictions3.index
#     y_predictions90=y_predictions90.set_index('Data')
    
#     EQM90 = mean_squared_error(y_teste, y_predictions90)
#     resid90 = np.sqrt(EQM90)
#     print('Test MSE: %.3f' % EQM90,resid90)
    
#     accuracy_90 = r2_score(y_teste, y_predictions90)
#     R2_90_teste = model90_fit.score(X_teste, y_teste) 
#     print ('accuracy, R2_teste: %.3f' % accuracy_90, R2_90_teste) 

#     return coef90, R290, y_predictions90, EQM90, resid90, accuracy_90, R2_90_teste



# #2) ElasticNetCV - (9)


# def ElasticNetCV_classificator(X_treino, X_teste, y_treino, y_teste): 
  
#     start_time=time.time()
    
#     l1_ratio_list=[0.25,0.5,0.75]
#     eps_ENCV_list = [1e-8,1,10]  

#     elasticNetCV_dict={}
#     for l1 in l1_ratio_list:
#         elasticNetCV_dict.update({l1:{}})
#         for eps in eps_ENCV_list:
            
#             model=ElasticNetCV(alphas=None, copy_X=True, eps=eps, fit_intercept=True,
#                                l1_ratio=l1, max_iter=1000, n_alphas=100, n_jobs=None,
#                                normalize=True, positive=False, precompute='auto', random_state=0,
#                                selection='cyclic', tol=0.0001, verbose=0) 
        
#             model_fit=model.fit(X_treino, y_treino)
        
#             y_predictions = model_fit.predict(X_teste)
            
#             elasticNetCV_dict[l1].update({eps:np.sqrt(mean_squared_error(y_teste, y_predictions))})
           
#     elasticNetCV_min_list=[]
#     for i in  elasticNetCV_dict.keys():
#         elasticNetCV_min_list.append((i,min(elasticNetCV_dict[i].items(), key=lambda x: x[1])))
              
#         l1=min(elasticNetCV_min_list, key=lambda x: x[1][1])[0]
#         eps=min(elasticNetCV_min_list, key=lambda x: x[1][1])[1][0]
             
#     print('O epsilson e o cv que apresentam o menor erro de previsão são iguais a %.1f' % eps, cv) 
#     print ("My program took", time.time() - start_time, "seconds to run") 
   
#     return l1, eps


# def run_elasticNetCV(l1, eps, X_treino, X_teste, y_treino, y_teste, treino, teste, indice):
    
#     print(ElasticNetCV().get_params())

#     model9 = ElasticNetCV(alphas=None, copy_X=True, eps=eps, fit_intercept=True,
#                           l1_ratio=l1_ratio, max_iter=1000, n_alphas=100, n_jobs=None,
#                           normalize=True, positive=False, precompute='auto', random_state=0,
#     selection='cyclic', tol=0.0001, verbose=0).fit(X_treino,y_treino)
    
#     model9_fit=model9.fit(X_treino,y_treino)
#     print(model9_fit.coef_) 
    
#     R29 = model9_fit.score(X_treino, y_treino) 
#     coef9=model9_fit.coef_
    
#     # make predictions
#     y_predictions9 = pd.DataFrame(index=teste, columns=['y_pred'])
#     ypred9=[]    
#     for i in range(int(len(teste)/forecastHorizon)):
#         ypred9.append(pd.Series(model9_fit.predict(X_teste[i*forecastHorizon:(i+1)*forecastHorizon])))   
#     if (len(teste)%forecastHorizon!=0):        
#         ypred9.append(pd.Series(model9_fit.predict(X_teste[(int(len(teste)/forecastHorizon)*forecastHorizon):len(teste)])))      
    
#     y_predictions9=pd.concat(ypred9)
#     y_predictions9=pd.DataFrame({'y_pred':y_predictions9})
#     y_predictions9['Data']=y_predictions3.index
#     y_predictions9=y_predictions9.set_index('Data')
    
#     EQM9 = mean_squared_error(y_teste, y_predictions9)
#     resid9 = np.sqrt(EQM9)
#     print('Test MSE: %.3f' % EQM9,resid9)
    
#     accuracy_9 = r2_score(y_teste, y_predictions9)
#     R2_9_teste = model9_fit.score(X_teste, y_teste) 
#     print ('accuracy, R2_teste: %.3f' % accuracy_9, R2_9_teste)
#     return coef9, R29, y_predictions9, EQM9, resid9, accuracy_9, R2_9_teste




# #Random Forest - (10)

# def RandomForest_classificator(X_treino, X_teste, y_treino, y_teste): 
    
#     n_estimators_list=[10,50,100,1000]
#     rf_dict={}
#     for n_estimator in n_estimators_list:    
#         model=RandomForestRegressor(n_estimators=n_estimator, max_depth=None, min_samples_split=2, 
#                                     min_samples_leaf=1, min_weight_fraction_leaf=0, max_leaf_nodes=None, 
#                                     bootstrap=True, oob_score=False,n_jobs=1, random_state=None, 
#                                     verbose=0, warm_start=False, max_features=None)
#         model_fit=model.fit(X_treino, y_treino)
#         y_predictions = model_fit.predict(X_teste)
#         rf_dict.update({n_estimator:np.sqrt(mean_squared_error(y_teste, y_predictions))})

#     n_estimator=min(rf_dict.items(), key=lambda x: x[1])[0]
          
#     print('O numero de estimatores que apresenta o menor erro de previsão é igual a %.1f' % n_estimator) 
    
#     return n_estimator


# def run_randomforest(n_estimator, X_treino, X_teste, y_treino, y_teste, treino, teste, indice):
    
#     print(RandomForestRegressor().get_params())

#     model10=RandomForestRegressor(n_estimators=n_estimator, max_depth=None, min_samples_split=2, 
#                                   min_samples_leaf=1, min_weight_fraction_leaf=0, max_leaf_nodes=None, 
#                                   bootstrap=True, oob_score=False,n_jobs=1, random_state=None, 
#                                   verbose=0, warm_start=False, max_features=None)
    
#     model10_fit=model10.fit(X_treino, y_treino)
    
#     print(model10_fit.feature_importances_)
#     coef10=model10_fit.feature_importances_
    
#     R210 = model10_fit.score(X_treino, y_treino) 
    
    
#     # make predictions
#     y_predictions10 = pd.DataFrame(index=teste, columns=['y_pred'])
#     ypred10=[]    
#     for i in range(int(len(teste)/forecastHorizon)):
#         ypred10.append(pd.Series(model10_fit.predict(X_teste[i*forecastHorizon:(i+1)*forecastHorizon])))   
#     if (len(teste)%forecastHorizon!=0):        
#         ypred10.append(pd.Series(model10_fit.predict(X_teste[(int(len(teste)/forecastHorizon)*forecastHorizon):len(teste)])))      
    
#     y_predictions10=pd.concat(ypred10)
#     y_predictions10=pd.DataFrame({'y_pred':y_predictions10})
#     y_predictions10['Data']=indice
#     y_predictions10=y_predictions10.set_index('Data')
    
#     EQM10 = mean_squared_error(y_teste, y_predictions10)
#     resid10 = np.sqrt(EQM10)
#     print('Test MSE: %.3f' % EQM10,resid10)
    
#     accuracy_10 = r2_score(y_teste, y_predictions10)
#     R2_10_teste = model10_fit.score(X_teste, y_teste) 
#     print ('accuracy, R2_teste: %.3f' % accuracy_10, R2_10_teste) 


#     return coef10, R210, y_predictions10, EQM10, resid10, accuracy_10, R2_10_teste




#     p,q=arima_classificator(y_treino, y_teste)
#     R21, y_predictions1, EQM1, resid1, accuracy_1, R2_1_teste = run_arima(p,q, y_treino, y_teste)
# #    arima_plot=model_plot('ARIMA', y_treino_diff, y_teste_diff, y_predictions1)
# #    R21, y_predictions1, EQM1, resid1, accuracy_1, R2_1_teste = auto_arima(p,q, y_treino, y_teste)
    
#     y_predictions2, EQM2, resid2, accuracy_2, R2_2_teste = run_randomwalk(y_treino, y_teste, teste, forecast_horizon)
#     model_plot('Random Walk', y_treino, y_teste, y_predictions2)
    
#     coef3, R23, y_predictions3, EQM3, resid3, accuracy_3, R2_3_teste= run_OLS(X_treino, X_teste, y_treino, y_teste, treino, teste, teste)
#     model_plot('OLS', y_treino, y_teste, y_predictions3)
#     indice=y_predictions3.index
    
#     coef30, R230, y_predictions30, EQM30, resid30, accuracy_30, R2_30_teste= run_linearregression(X_treino, X_teste, y_treino, y_teste, treino, teste, teste)
#     model_plot('Linear Regression', y_treino, y_teste, y_predictions30)
        
#     alpha = lasso_classificator(X_treino, X_teste, y_treino, y_teste)
#     coef4, R24, y_predictions4, EQM4, resid4, accuracy_4, R2_4_teste = run_lasso(alpha, X_treino, X_teste, y_treino, y_teste, treino, teste, indice)
#     model_plot('Lasso', y_treino, y_teste, y_predictions4)

#     eps, cv = lassoCV_classificator(X_treino, X_teste, y_treino, y_teste)
#     coef40, R240, y_predictions40, EQM40, resid40, accuracy_40, R2_40_teste = run_lassoCV(eps, cv, X_treino, X_teste, y_treino, y_teste, treino, teste, indice)
#     model_plot('Lasso CV', y_treino, y_teste, y_predictions40)    
    
#     eps, nZeroCoef = lars_classificator(X_treino, X_teste, y_treino, y_teste)    
#     coef5, R25, y_predictions5, EQM5, resid5, accuracy_5, R2_5_teste = run_lars(eps, nZeroCoef, X_treino, X_teste, y_treino, y_teste, treino, teste, indice)
#     model_plot('Lars', y_treino, y_teste, y_predictions5)    
    
#     eps, alpha =lassoLars_classificator(X_treino, X_teste, y_treino, y_teste)
#     coef6, R26, y_predictions6, EQM6, resid6, accuracy_6, R2_6_teste = run_lassoLars(eps, alpha, X_treino, X_teste, y_treino, y_teste, treino, teste, indice)
#     model_plot('Lasso Lars', y_treino, y_teste, y_predictions6)    
    
#     eps, cv = lassoLarsCV_classificator(X_treino, X_teste, y_treino, y_teste)
#     coef7, R27, y_predictions7, EQM7, resid7, accuracy_7, R2_7_teste = run_lassoLarsCV(eps, cv, X_treino, X_teste, y_treino, y_teste, treino, teste, indice)
#     model_plot('Lasso Lars CV' , y_treino, y_teste, y_predictions7)    
    
#     alpha = ridge_classificator(X_treino, X_teste, y_treino, y_teste)
#     coef8, R28, y_predictions8, EQM8, resid8, accuracy_8, R2_8_teste = run_ridge(alpha, X_treino, X_teste, y_treino, y_teste, treino, teste, indice)
#     model_plot('Ridge', y_treino, y_teste, y_predictions8)    
    
#     alpha, l1_ratio = ElasticNet_classificator(X_treino, X_teste, y_treino, y_teste)
#     coef90, R290, y_predictions90, EQM90, resid90, accuracy_90, R2_90_teste = run_elasticNet(alpha, l1_ratio, X_treino, X_teste, y_treino, y_teste, treino, teste, indice)
#     model_plot('Elastic Net', y_treino, y_teste, y_predictions90)    
    
#     l1_ratio, eps = ElasticNetCV_classificator(X_treino, X_teste, y_treino, y_teste)
#     coef9, R29, y_predictions9, EQM9, resid9, accuracy_9, R2_9_teste = run_elasticNetCV(l1_ratio, eps, X_treino, X_teste, y_treino, y_teste, treino, teste, indice)
#     model_plot('Elastic Net CV', y_treino, y_teste, y_predictions9)   
    
#     n_estimator = RandomForest_classificator(X_treino, X_teste, y_treino, y_teste)
#     coef10, R210, y_predictions10, EQM10, resid10, accuracy_10, R2_10_teste = run_randomforest(n_estimator, X_treino, X_teste, y_treino, y_teste, treino, teste, indice)
#     model_plot('Random Forest', y_treino, y_teste, y_predictions10)
    
#     EQM3,coef3,resid3,accuracy_3,R2_3_teste,EQM4,coef4,resid4,accuracy_4,R2_4_teste,EQM6,coef6,resid6,accuracy_6,R2_6_teste,EQM9,coef9,resid9,accuracy_9,R2_9_teste = model_classificator(EQM3,coef3,resid3,accuracy_3,R2_3_teste,EQM30,coef30,resid30,accuracy_30,R2_30_teste,EQM4,coef4,resid4,accuracy_4,R2_4_teste,EQM40,coef40,resid40,accuracy_40,R2_40_teste,EQM6,coef6,resid6,accuracy_6,R2_6_teste,EQM7,coef7,resid7,accuracy_7,R2_7_teste,EQM9,coef9,resid9,accuracy_9,R2_9_teste,EQM90,coef90,resid90,accuracy_90,R2_90_teste)

