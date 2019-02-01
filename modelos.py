#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:33:42 2018

@author: pedrocampelo
"""

 
#Bibliotecas Usuais
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot
from plotly.plotly import plot_mpl
from pprint import pprint   

import pandas as pd
from pandas import Series
from pandas import read_csv 
from pandas import DataFrame
from pandas import concat
from pandas.tools.plotting import autocorrelation_plot

import datetime as dt 
import time

#Modelos
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

import sklearn 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lars    
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor

os.chdir('/Users/pedrocampelo/Desktop/Work/UnB/Tese/Dados')
cwd = os.getcwd()
cwd
 
periodo = pd.date_range('2/1/2017', periods=546)
treino = pd.date_range('2/1/2017', periods=365)
teste = pd.date_range('2/1/2018', periods=181)

consumo = read_csv('dadosconsumo.csv', sep=';',header=0, parse_dates=[0], index_col='Data', squeeze=True)
variaveis = read_csv('dadosx.csv', sep=';',header=0, parse_dates=[0], index_col='Data', squeeze=True, encoding = "ISO-8859-1")

  
#Ajustando as variáveis de temperatura
#BS
variaveis['TEMPaj_BS_SECO_9']=variaveis['TEMP_BS_SECO_9'] - variaveis['TEMP_BS_SECO_9'].mean()
variaveis['TEMPaj_BS_SECO_15']=variaveis['TEMP_BS_SECO_15'] - variaveis['TEMP_BS_SECO_15'].mean()
variaveis['TEMPaj_BS_SECO_21']=variaveis['TEMP_BS_SECO_21'] - variaveis['TEMP_BS_SECO_21'].mean()
variaveis['TEMPaj_BS_SECO_MED']=variaveis['TEMP_BS_SECO_MED'] - variaveis['TEMP_BS_SECO_MED'].mean()

variaveis['TEMPaj_BS_SUL_9']=variaveis['TEMP_BS_SUL_9'] - variaveis['TEMP_BS_SUL_9'].mean()
variaveis['TEMPaj_BS_SUL_15']=variaveis['TEMP_BS_SUL_15'] - variaveis['TEMP_BS_SUL_15'].mean()
variaveis['TEMPaj_BS_SUL_21']=variaveis['TEMP_BS_SUL_21'] - variaveis['TEMP_BS_SUL_21'].mean()
variaveis['TEMPaj_BS_SUL_MED']=variaveis['TEMP_BS_SUL_MED'] - variaveis['TEMP_BS_SUL_MED'].mean()

variaveis['TEMPaj_BS_NE_9']=variaveis['TEMP_BS_NE_9'] - variaveis['TEMP_BS_NE_9'].mean()
variaveis['TEMPaj_BS_NE_15']=variaveis['TEMP_BS_NE_15'] - variaveis['TEMP_BS_NE_15'].mean()
variaveis['TEMPaj_BS_NE_21']=variaveis['TEMP_BS_NE_21'] - variaveis['TEMP_BS_NE_21'].mean()
variaveis['TEMPaj_BS_NE_MED']=variaveis['TEMP_BS_NE_MED'] - variaveis['TEMP_BS_NE_MED'].mean()

variaveis['TEMPaj_BS_N_9']=variaveis['TEMP_BS_N_9'] - variaveis['TEMP_BS_N_9'].mean()
variaveis['TEMPaj_BS_N_15']=variaveis['TEMP_BS_N_15'] - variaveis['TEMP_BS_N_15'].mean()
variaveis['TEMPaj_BS_N_21']=variaveis['TEMP_BS_N_21'] - variaveis['TEMP_BS_N_21'].mean()
variaveis['TEMPaj_BS_N_MED']=variaveis['TEMP_BS_N_MED'] - variaveis['TEMP_BS_N_MED'].mean()

variaveis['TEMPaj_BS_BR_9']=variaveis['TEMP_BS_BR_9'] - variaveis['TEMP_BS_BR_9'].mean()
variaveis['TEMPaj_BS_BR_15']=variaveis['TEMP_BS_BR_15'] - variaveis['TEMP_BS_BR_15'].mean()
variaveis['TEMPaj_BS_BR_21']=variaveis['TEMP_BS_BR_21'] - variaveis['TEMP_BS_BR_21'].mean()
variaveis['TEMPaj_BS_BR_MED']=variaveis['TEMP_BS_BR_MED'] - variaveis['TEMP_BS_BR_MED'].mean()

#Ajustando as variáveis de temperatura
#BU
variaveis['TEMPaj_BU_SECO_9']=variaveis['TEMP_BU_SECO_9'] - variaveis['TEMP_BU_SECO_9'].mean()
variaveis['TEMPaj_BU_SECO_15']=variaveis['TEMP_BU_SECO_15'] - variaveis['TEMP_BU_SECO_15'].mean()
variaveis['TEMPaj_BU_SECO_21']=variaveis['TEMP_BU_SECO_21'] - variaveis['TEMP_BU_SECO_21'].mean()
variaveis['TEMPaj_BU_SECO_MED']=variaveis['TEMP_BU_SECO_MED'] - variaveis['TEMP_BU_SECO_MED'].mean()

variaveis['TEMPaj_BU_SUL_9']=variaveis['TEMP_BU_SUL_9'] - variaveis['TEMP_BU_SUL_9'].mean()
variaveis['TEMPaj_BU_SUL_15']=variaveis['TEMP_BU_SUL_15'] - variaveis['TEMP_BU_SUL_15'].mean()
variaveis['TEMPaj_BU_SUL_21']=variaveis['TEMP_BU_SUL_21'] - variaveis['TEMP_BU_SUL_21'].mean()
variaveis['TEMPaj_BU_SUL_MED']=variaveis['TEMP_BU_SUL_MED'] - variaveis['TEMP_BU_SUL_MED'].mean()

variaveis['TEMPaj_BU_NE_9']=variaveis['TEMP_BU_NE_9'] - variaveis['TEMP_BU_NE_9'].mean()
variaveis['TEMPaj_BU_NE_15']=variaveis['TEMP_BU_NE_15'] - variaveis['TEMP_BU_NE_15'].mean()
variaveis['TEMPaj_BU_NE_21']=variaveis['TEMP_BU_NE_21'] - variaveis['TEMP_BU_NE_21'].mean()
variaveis['TEMPaj_BU_NE_MED']=variaveis['TEMP_BU_NE_MED'] - variaveis['TEMP_BU_NE_MED'].mean()

variaveis['TEMPaj_BU_N_9']=variaveis['TEMP_BU_N_9'] - variaveis['TEMP_BU_N_9'].mean()
variaveis['TEMPaj_BU_N_15']=variaveis['TEMP_BU_N_15'] - variaveis['TEMP_BU_N_15'].mean()
variaveis['TEMPaj_BU_N_21']=variaveis['TEMP_BU_N_21'] - variaveis['TEMP_BU_N_21'].mean()
variaveis['TEMPaj_BU_N_MED']=variaveis['TEMP_BU_N_MED'] - variaveis['TEMP_BU_N_MED'].mean()

variaveis['TEMPaj_BU_BR_9']=variaveis['TEMP_BU_BR_9'] - variaveis['TEMP_BU_BR_9'].mean()
variaveis['TEMPaj_BU_BR_15']=variaveis['TEMP_BU_BR_15'] - variaveis['TEMP_BU_BR_15'].mean()
variaveis['TEMPaj_BU_BR_21']=variaveis['TEMP_BU_BR_21'] - variaveis['TEMP_BU_BR_21'].mean()
variaveis['TEMPaj_BU_BR_MED']=variaveis['TEMP_BU_BR_MED'] - variaveis['TEMP_BU_BR_MED'].mean()




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
            'PIBI_o','PIBI_d','PIBCo_o','PIBCo_d','PIA_o','PIA_d','ICC','INEC','ICEI','DBNDES','IEG_o','IEG_d','IETIT_o','IETIT_d','IETC_o','IETC_d','IETS_o','IETS_d','IETCV_o','IETCV_d', 'PO','TD','BM','PME',
            'TEMPaj_BS_SECO_9','TEMPaj_BS_SECO_15','TEMPaj_BS_SECO_21','TEMPaj_BS_SECO_MED','TEMPaj_BS_SUL_9','TEMPaj_BS_SUL_15','TEMPaj_BS_SUL_21','TEMPaj_BS_SUL_MED','TEMPaj_BS_NE_9','TEMPaj_BS_NE_15','TEMPaj_BS_NE_21',
            'TEMPaj_BS_NE_MED','TEMPaj_BS_N_9','TEMPaj_BS_N_15','TEMPaj_BS_N_21','TEMPaj_BS_N_MED','TEMPaj_BS_BR_9','TEMPaj_BS_BR_15','TEMPaj_BS_BR_21','TEMPaj_BS_BR_MED','TEMPaj_BU_SECO_9','TEMPaj_BU_SECO_15',
            'TEMPaj_BU_SECO_21','TEMPaj_BU_SECO_MED','TEMPaj_BU_SUL_9','TEMPaj_BU_SUL_15','TEMPaj_BU_SUL_21','TEMPaj_BU_SUL_MED','TEMPaj_BU_NE_9','TEMPaj_BU_NE_15','TEMPaj_BU_NE_21','TEMPaj_BU_NE_MED','TEMPaj_BU_N_9',
            'TEMPaj_BU_N_15','TEMPaj_BU_N_21','TEMPaj_BU_N_MED','TEMPaj_BU_BR_9','TEMPaj_BU_BR_15','TEMPaj_BU_BR_21','TEMPaj_BU_BR_MED'] 


#    label = read_csv('label.csv', sep=';',header=0, dtype={'label':str})
#    label2 = pd.DataFrame(label, colunas)

#agregando os dados de consumo de EE com as demais variáveis


import numpy as np  

agregado = np.hstack((consumo, variaveis))
dados = pd.DataFrame(agregado, columns = colunas , index=periodo)  
del consumo, variaveis, agregado

 #Rodar30
#Definir Horizonte de Previsão de X dias:    
forecastHorizon=1

#A partir dos dois gráficos podemos decidir quantos lags usar
#Vamos usar 3 lags para rodar os modelos

y = dados['CEE_BR_TOT']
colunas=dados.columns
X = pd.DataFrame(index=periodo)

#    #modelo1
#modo=1
#for i in colunas:
#    X[i+'lag1']=dados[i].shift(forecastHorizon)
#    
#for i in colunas:
#    X[i+'lag2']=dados[i].shift(forecastHorizon+1)
#    
#for i in colunas:
#    X[i+'lag3']=dados[i].shift(forecastHorizon+2)
#   
 
#modelo2
modo=2
for i in colunas:
    X[i+'lag1']=dados[i].shift(round(forecastHorizon))
    
for i in colunas:
    X[i+'lag2']=dados[i].shift(round(forecastHorizon*2))
    
for i in colunas:
    X[i+'lag3']=dados[i].shift(round(forecastHorizon*3))

#modelo3
#modo=3
#for i in colunas:
#    X[i+'lag1']=dados[i].shift(round(forecastHorizon))
#    
#for i in colunas:
#    X[i+'lag2']=dados[i].shift(round(forecastHorizon*(3/2)))
#    
#for i in colunas:
#    X[i+'lag3']=dados[i].shift(round(forecastHorizon*(9/4)))

#modelo4
#modo=4
#for i in colunas:
#    X[i+'lag1']=dados[i].shift(round(forecastHorizon))
#    
#for i in colunas:
#    X[i+'lag2']=dados[i].shift(round(forecastHorizon+7))
#    
#for i in colunas:
#    X[i+'lag3']=dados[i].shift(round(forecastHorizon+14))  

    

X=X.apply(pd.to_numeric,errors='coerce')
 
lenx=len(X)
X=X.dropna()
X['constante']=1
lenx1=len(X)
dif_len=lenx-lenx1
y=y[dif_len:]
   
treino= periodo[:-forecastHorizon]
teste=periodo[-forecastHorizon:]


#Dividindo em teste e treino
train_size = int(len(X) * ((len(X)-forecastHorizon)/len(X)))
X_treino, X_teste = X[0:train_size], X[train_size:len(X)]
y_treino, y_teste = y[0:train_size], y[train_size:len(y)]
 
print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(X_treino)))
print('Testing Observations: %d' % (len(X_teste)))



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
#plt.savefig('CEElag1agreg.png')   
plt.show()



#Checando estacionariedade da série de consumo de EE    
#    from statsmodels.tsa.stattools import adfuller, kpss
#    
#    def adf_test(y):
#        # perform Augmented Dickey Fuller test
#        print('Results of Augmented Dickey-Fuller test:')
#        dftest = adfuller(y, autolag='AIC')
#        dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
#        for key, value in dftest[4].items():
#            dfoutput['Critical Value ({})'.format(key)] = value
#        print(dfoutput)
#       
#     
#    adf_test(y_treino)
#    adf_test(y_teste)
#    adf_test(y)
#    

#Não podemos rejeitas a hipótese nula a 10%, então a série é não estacionária (série toda e separada).

#1st difference
#Olhar qual sera a data
#    y_treino_diff_aux = np.diff(y_treino)
#    print(y_treino[1:].index)
#    treino_diff= pd.date_range('3/6/2017', periods=len(y_treino_diff_aux))
#    y_treino_diff = pd.DataFrame(y_treino_diff_aux, index=treino_diff)
#    adf_test(y_treino_diff_aux)     
#    
#    y_teste_diff_aux=np.diff(y_teste)
#    print(y_treino[-1:].index)
#    teste_diff= pd.date_range('7/2/2018', periods=len(y_teste_diff_aux))
#    y_teste_diff = pd.DataFrame(y_teste_diff_aux, index=teste_diff)
#    adf_test(y_teste_diff_aux)     

#1a diferença é estacionária


#AR - (1)
 
#outro modelo arima (consegue decidir o numero de lags)

model1 = ARIMA(y_treino, order=(15,1,0))
model_fit1 = model1.fit()
print('Lag: %s' % model_fit1.k_ar)
print('Coefficients: %s' % model_fit1.params)
R21=r2_score(y_treino, model_fit1.predict(start=1, end=len(y_treino), dynamic=False))

# make predictions
y_predictions1 = model_fit1.predict(start=len(y_treino), end=len(y_treino)+len(y_teste)-1, dynamic=False)
EQM1= mean_squared_error(y_teste, y_predictions1)
resid1 = np.sqrt(EQM1)   
print('Test MSE: %.3f' % EQM1, resid1)
print(model_fit1.summary())

accuracy_1 = r2_score(y_teste, y_predictions1)
R2_1_teste = accuracy_1
print ('accuracy, R2_teste: %.3f' % accuracy_1, R2_1_teste) 

#    if forecastHorizon>1: 
#        plt.figure()    
#        pyplot.plot(y_treino_diff, label='Train')
#        pyplot.plot(y_teste_diff, color='black', label='Test')
#        pyplot.plot(y_predictions1 , label='Forecast')
#        #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
#        plt.legend(loc='best')
#        plt.ylabel('KW/h')
#        plt.xticks(rotation=30)
#        plt.title('Power Eletricity Consumption Forecast  (AR)')
#        #plt.grid()
#        plt.savefig('modelBR1_'+str(forecastHorizon)+'lag.png')  
#        pyplot.show()
#



#Random Walk (2)

y_predictions2 = pd.DataFrame(data=list(y_treino.tail(forecastHorizon)), index=teste) #previsão

EQM2 = mean_squared_error(y_teste, y_predictions2) #EQM
resid2 = np.sqrt(EQM2) #Resíduo
print('Test MSE, Residual: %.3f' % EQM2, resid2)

accuracy_2 = r2_score(y_teste, y_predictions2)
R2_2_teste = sm.OLS(y_teste,X_teste).fit().rsquared
print ('accuracy, R2_teste: %.3f' % accuracy_2, R2_2_teste)

if forecastHorizon>1: 
    plt.figure() 
    pyplot.plot(y_treino, label='Train')
    pyplot.plot(y_teste, color='black', label='Test')
    pyplot.plot(y_predictions2, label='Forecast')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Power Eletricity Consumption Forecast  (Random Walk)')
    #plt.grid()
    plt.savefig('modelBR2_'+str(forecastHorizon)+'lag.png') 
    pyplot.show() 




                           #OLS    (3)            

model3 = sm.OLS(y_treino,X_treino)                  #modelo
model_fit3 = model3.fit() 
print (model_fit3.summary())                        #sumário do modelo
coef3=model_fit3.params

R23=model_fit3.rsquared

    
# make predictions
y_predictions3 = model_fit3.predict(X_teste)          #previsão

EQM3 = mean_squared_error(y_teste, y_predictions3)    #EQM
resid3 = np.sqrt(EQM3)                                #Resíduo

print('Test MSE, Residual: %.3f' % EQM3, resid3)
    
accuracy_3 = r2_score(y_teste, y_predictions3)
R2_3_teste = sm.OLS(y_teste,X_teste).fit().rsquared
print ('accuracy, R2_teste: %.3f' % accuracy_3, R2_3_teste)

if forecastHorizon>1: 
    plt.figure()    
    pyplot.plot(y_treino, label='Train')
    pyplot.plot(y_teste, color='black', label='Test')
    pyplot.plot(y_predictions3 , label='Forecast')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Power Eletricity Consumption Forecast  (OLS)')
    #plt.grid()
    plt.savefig('modelBR3_'+str(forecastHorizon)+'lag.png')   
    pyplot.show()   




#2)Linear Regression (4)
 
       
model30= LinearRegression().fit(X_treino, y_treino)
print(model30.score(X_treino, y_treino))                        #R2 fora da amostra
print(model30.coef_)                                            #coeficientes
coef30=np.transpose(model30.coef_)

R230=model30.score(X_treino, y_treino)    
    
predictions30 = model30.predict(X_teste)
y_predictions30= pd.DataFrame(predictions30, index=teste)   #previsão
    
EQM30 = mean_squared_error(y_teste, y_predictions30)      #EQM
resid30 = np.sqrt(EQM30)                                #Residuo
print('Test MSE, residuo: %.3f' % EQM30,resid30)

accuracy_30 = r2_score(y_teste, y_predictions30)
R2_30_teste = model30.score(X_teste, y_teste)  
print ('accuracy, R2_teste: %.3f' % accuracy_30, R2_30_teste)

if forecastHorizon>1:     
    plt.figure()    
    pyplot.plot(y_treino, label='Train')
    pyplot.plot(y_teste, color='black', label='Test')
    pyplot.plot(y_predictions30 , label='Forecast')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Power Eletricity Consumption Forecast  (Linear Regression)')
    #plt.grid()
    plt.savefig('modelBR30_'+str(forecastHorizon)+'lag.png')   
    pyplot.show()  







#1)Lasso normal - (4)


#Olhar Default 
print(Lasso().get_params())

def lasso_classificator(X_treino, X_teste, y_treino, y_teste): 
    alpha_lasso_lista = [1e-15, 1e-10, 1e-8, 1e-5, 1e-3,1e-2,1e-1, 1, 5, 10,20]
    EQM_lista=[]
    for alpha in alpha_lasso_lista:
        model = linear_model.Lasso(alpha=alpha, copy_X=True, fit_intercept=True, max_iter=1000,
                                   normalize=True, positive=False, precompute=False, random_state=None,
                                   selection='cyclic', tol=0.0001, warm_start=False)
        model_fit=model.fit(X_treino,y_treino)
        
        y_predictions = model_fit.predict(X_teste)
        EQM_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        # print(EQM_lista)
    
    EQM=pd.DataFrame(columns=['alpha', 'resíduo'], index=list(range(len(alpha_lasso_lista))))
    EQM['alpha']=alpha_lasso_lista
    EQM['resíduo']=EQM_lista
    
    print(EQM.sort_values(by='resíduo', ascending=True)[0:5])
    alpha=EQM.sort_values(by='resíduo', ascending=True)[0:1]['alpha'].item()
    
    print('O alpha que apresente o menor erro de previsão é igual a %.1f' % alpha) 
    
    return alpha

alpha = lasso_classificator(X_treino, X_teste, y_treino, y_teste)


model4 = linear_model.Lasso(alpha=alpha, copy_X=True, fit_intercept=True, max_iter=1000,
                            normalize=True, positive=False, precompute=False, random_state=None,
                            selection='cyclic', tol=0.0001, warm_start=False)
model_fit4=model4.fit(X_treino,y_treino)

coef4=model4.coef_
R24 = model_fit4.score(X_treino,y_treino) 
# make predictions
y_predictions4 = model_fit4.predict(X_teste)
y_predictions4= pd.DataFrame(y_predictions4, index=teste) #previsão 
EQM4 = mean_squared_error(y_teste, y_predictions4)
resid4 = np.sqrt(EQM4)
print('Test MSE, residuo: %.3f' % EQM4,resid4)

accuracy_4 = r2_score(y_teste, y_predictions4)
R2_4_teste = model_fit4.score(X_teste, y_teste) 
print ('accuracy, R2_teste: %.3f' % accuracy_4, R2_4_teste)


if forecastHorizon>1: 
    plt.figure() 
    pyplot.plot(y_treino, label='Train')
    pyplot.plot(y_teste, color='black', label='Test')
    pyplot.plot(y_predictions4, label='Forecast')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Power Eletricity Consumption Forecast  (Lasso)')
    #plt.grid()
    plt.savefig('modelBR4_'+str(forecastHorizon)+'lag.png') 
    pyplot.show() 




#2) Lasso CV - (40)

from sklearn.linear_model import LassoCV 

print(LassoCV().get_params()) 

def lassoCV_classificator(X_treino, X_teste, y_treino, y_teste):
    eps_LassoCV_list = [1e-8,1,5]
    cv_LassoCV_list=[3,10]
    EQM3_lista=[]
    EQM10_lista=[]
    
    for eps in eps_LassoCV_list:
        start_time = time.time()
        model= LassoCV(fit_intercept=True, verbose=False, max_iter=500, 
                       normalize=True, cv=cv_LassoCV_list[0], eps=eps, copy_X=True, 
                       positive=False) 
        model_fit=model.fit(X_treino, y_treino)
    
        y_predictions = model_fit.predict(X_teste)
        EQM3_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM3_lista)
        print ("My program took", time.time() - start_time, "seconds to run")
    
    for eps in eps_LassoCV_list:
        start_time = time.time()
        model= LassoCV(fit_intercept=True, verbose=False, max_iter=500, 
                       normalize=True, cv=cv_LassoCV_list[1], eps=eps, copy_X=True, 
                       positive=False) 
        model_fit=model.fit(X_treino, y_treino)
        
        y_predictions = model_fit.predict(X_teste)
        EQM10_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM10_lista)
        print ("My program took", time.time() - start_time, "seconds to run")
    
    
    EQM1=pd.DataFrame(columns=['EQM3', 'EQM10'], index=eps_LassoCV_list)
    EQM1['EQM3']=EQM3_lista
    EQM1['EQM10']=EQM10_lista
    
    menorEQM1_list=[sorted(EQM3_lista)[0],sorted(EQM10_lista)[0]] 
    menorEQM1=sorted(menorEQM1_list)[0]
    
    position1=menorEQM1_list.index(menorEQM1)
    menorEQM_col=EQM1.columns[position1]
    
    cv=cv_LassoCV_list[position1]
    eps=EQM1.loc[EQM1[menorEQM_col].isin([menorEQM1])].index[0].item() 
    
    print ("My program took", time.time() - start_time, "seconds to run") 
    return cv, eps

cv, eps = lassoCV_classificator(X_treino, X_teste, y_treino, y_teste)


model40= LassoCV(fit_intercept=True, verbose=False, max_iter=500, normalize=True, 
                cv=cv, eps=eps, copy_X=True, positive=False) 

model40_fit = model40.fit(X_treino,y_treino)

print(model40_fit.coef_) 
coef40=model40_fit.coef_ 
R240 = model40_fit.score(X_treino, y_treino) 

y_predictions40 = model40_fit.predict(X_teste)
y_predictions40= pd.DataFrame(y_predictions40, index=teste) #previsão

EQM40 = mean_squared_error(y_teste, y_predictions40)
resid40 = np.sqrt(EQM40)
print('Test MSE: %.3f' % EQM40,resid40)

accuracy_40 = r2_score(y_teste, y_predictions40)
R2_40_teste = model40_fit.score(X_teste, y_teste) 
print ('accuracy, R2_teste: %.3f' % accuracy_40, R2_40_teste)


if forecastHorizon>1: 
    plt.figure() 
    pyplot.plot(y_treino, label='Train')
    pyplot.plot(y_teste, color='black', label='Test')
    pyplot.plot(y_predictions40 , label='Forecast')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Power Eletricity Consumption Forecast  (Lasso CV)')
    plt.grid()
    plt.savefig('modelBR40_'+str(forecastHorizon)+'lag.png') 
    pyplot.show()
   
    
    
    


#1) Lars - 5

#Olhar Default 
print(Lars().get_params())

def lars_classificator(X_treino, X_teste, y_treino, y_teste):
    start_time=time.time()
    eps_list = [1e-15, 1e-10, 1e-8, 1e-5,1e-3,1e-2, 1,2,5, 10,20]
    nzero_coef_list=[1,10,50,100,500]
    EQM1_lista=[]
    EQM10_lista=[]
    EQM50_lista=[]
    EQM100_lista=[]
    EQM500_lista=[]
    
    for eps in eps_list:
        model=Lars(fit_intercept=True, verbose=False, normalize=True, 
                   n_nonzero_coefs=nzero_coef_list[0], eps=eps, 
                   copy_X=True, fit_path=True, positive=False) 
        model_fit=model.fit(X_treino, y_treino)
    
        y_predictions = model_fit.predict(X_teste)
        EQM1_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
    
    for eps in eps_list:
        model=Lars(fit_intercept=True, verbose=False, normalize=True, 
                   n_nonzero_coefs=nzero_coef_list[1], eps=eps, 
                   copy_X=True, fit_path=True, positive=False) 
        model_fit=model.fit(X_treino, y_treino)
    
        y_predictions = model_fit.predict(X_teste)
        EQM10_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
    
    for eps in eps_list:
        model=Lars(fit_intercept=True, verbose=False, normalize=True, 
                   n_nonzero_coefs=nzero_coef_list[2], eps=eps, 
                   copy_X=True, fit_path=True, positive=False) 
        model_fit=model.fit(X_treino, y_treino)
    
        y_predictions = model_fit.predict(X_teste)
        EQM50_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
    
    
    for eps in eps_list:
        model=Lars(fit_intercept=True, verbose=False, normalize=True, 
                   n_nonzero_coefs=nzero_coef_list[3], eps=eps, 
        copy_X=True, fit_path=True, positive=False) 
        model_fit=model.fit(X_treino, y_treino)
        
        y_predictions = model_fit.predict(X_teste)
        EQM100_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
    
    for eps in eps_list:
        model=Lars(fit_intercept=True, verbose=False, normalize=True, 
                   n_nonzero_coefs=nzero_coef_list[4], eps=eps, 
                   copy_X=True, fit_path=True, positive=False) 
        model_fit=model.fit(X_treino, y_treino)
    
        y_predictions = model_fit.predict(X_teste)
        EQM500_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
    
    
    EQM1=pd.DataFrame(columns=['EQM1', 'EQM10','EQM50', 'EQM100','EQM500'], index=eps_list)
    EQM1['EQM1']=EQM1_lista
    EQM1['EQM10']=EQM10_lista
    EQM1['EQM50']=EQM50_lista
    EQM1['EQM100']=EQM100_lista
    EQM1['EQM500']=EQM500_lista
    
    menorEQM1_list=[sorted(EQM1_lista)[0],sorted(EQM10_lista)[0],
    sorted(EQM50_lista)[0],sorted(EQM100_lista)[0],sorted(EQM500_lista)[0]] 
    menorEQM1 = sorted(menorEQM1_list)[0]
    
    position1=menorEQM1_list.index(menorEQM1)
    menorEQM_col=EQM1.columns[position1]
    
    nZeroCoef=nzero_coef_list[position1]
    eps=EQM1.loc[EQM1[menorEQM_col].isin([menorEQM1])].index[0].item() 
    
    print ("My program took", time.time() - start_time, "seconds to run") 
    print('O epsilon e o número de coeficientes diferente de zero que apresentam'
    'o menor erro de previsão são iguais a %.1f' % eps,nZeroCoef) 
    return nZeroCoef, eps

nZeroCoef,eps = lars_classificator(X_treino, X_teste, y_treino, y_teste)

model5=Lars(fit_intercept=True, verbose=False, normalize=True, 
            n_nonzero_coefs=int(nZeroCoef), eps=eps, copy_X=True, fit_path=True, 
            positive=False) 
model5_fit=model5.fit(X_treino, y_treino)    
coef5=model5_fit.coef_ 
R25 = model5_fit.score(X_treino, y_treino) 

# make predictions
y_predictions5 = model5_fit.predict(X_teste)
y_predictions5= pd.DataFrame(y_predictions5, index=teste) #previsão


EQM5 = mean_squared_error(y_teste, y_predictions5)
resid5 = np.sqrt(EQM5)
print('Test MSE: %.3f' % EQM5,resid5)

accuracy_5 = r2_score(y_teste, y_predictions5)
R2_5_teste = model5_fit.score(X_teste, y_teste) 
print ('accuracy, R2_teste: %.3f' % accuracy_5, R2_5_teste) 


if forecastHorizon>1: 
    plt.figure() 
    pyplot.plot(y_treino, label='Train')
    pyplot.plot(y_teste, color='black', label='Test')
    pyplot.plot(y_predictions5 , label='Forecast')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Power Eletricity Consumption Forecast  (Lars)')
    plt.grid()
    plt.savefig('modelBR5_'+str(forecastHorizon)+'lag.png') 
    pyplot.show()
        

    
#2) Lasso Lars - 6

#Olhar Default 
print(LassoLars().get_params()) 

def lassoLars_classificator(X_treino, X_teste, y_treino, y_teste): 
    # alpha_lassoLars_lista = [1e-15, 1e-8, 1e-5,1e-1,1, 5,20]
    start_time=time.time()
    eps_LL_list=[1e-15, 1e-8,1e-3,1,20]
    alpha_LL_lista=[1e-10,1e-8,1,10]
    EQM1_lista=[]
    EQM2_lista=[]
    EQM3_lista=[]
    EQM4_lista=[]
    
    
    for eps in eps_LL_list:
        model=LassoLars(alpha=alpha_LL_lista[0],fit_intercept=True, verbose=False, normalize=True, 
                        max_iter=500, eps=eps, copy_X=True,fit_path=True, positive=False) 
        model_fit=model.fit(X_treino, y_treino)
        
        y_predictions = model_fit.predict(X_teste)
        EQM1_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
    
    for eps in eps_LL_list:
        model=LassoLars(alpha=alpha_LL_lista[1],fit_intercept=True, verbose=False, normalize=True, 
                        max_iter=500, eps=eps, copy_X=True,fit_path=True, positive=False) 
        model_fit=model.fit(X_treino, y_treino)
        
        y_predictions = model_fit.predict(X_teste)
        EQM2_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
    
    for eps in eps_LL_list:
        model=LassoLars(alpha=alpha_LL_lista[2],fit_intercept=True, verbose=False, normalize=True, 
                        max_iter=500, eps=eps, copy_X=True,fit_path=True, positive=False) 
        model_fit=model.fit(X_treino, y_treino)
        
        y_predictions = model_fit.predict(X_teste)
        EQM3_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions))) 
    
    for eps in eps_LL_list:
        model=LassoLars(alpha=alpha_LL_lista[1],fit_intercept=True, verbose=False, normalize=True, 
                        max_iter=500, eps=eps, copy_X=True,fit_path=True, positive=False) 
        model_fit=model.fit(X_treino, y_treino)
    
        y_predictions = model_fit.predict(X_teste)
        EQM4_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
    
    
    EQM1=pd.DataFrame(columns=['EQM1', 'EQM2','EQM3', 'EQM4'], index=eps_LL_list)
    EQM1['EQM1']=EQM1_lista
    EQM1['EQM2']=EQM2_lista
    EQM1['EQM3']=EQM3_lista
    EQM1['EQM4']=EQM4_lista
    
    menorEQM1_list=[sorted(EQM1_lista)[0],sorted(EQM2_lista)[0],
    sorted(EQM3_lista)[0],sorted(EQM4_lista)[0]] 
    menorEQM1=sorted(menorEQM1_list)[0]
    
    position=menorEQM1_list.index(menorEQM1)
    menorEQM_col=EQM1.columns[position]
    
    alpha=alpha_LL_lista[position]
    eps=EQM1.loc[EQM1[menorEQM_col].isin([menorEQM1])].index[0].item()
    
    print ("My program took", time.time() - start_time, "seconds to run") 
    return alpha, eps
    
alpha, eps =lassoLars_classificator(X_treino, X_teste, y_treino, y_teste)


model6=LassoLars( alpha=alpha, fit_intercept=True, verbose=False, normalize=True, 
                 max_iter=500, eps=eps, copy_X=True,fit_path=True, positive=False) 
model6_fit=model6.fit(X_treino, y_treino)
coef6=model6_fit.coef_

R26 = model6_fit.score(X_treino, y_treino) 

y_predictions6 = model6_fit.predict(X_teste)
y_predictions6= pd.DataFrame(y_predictions6, index=teste) #previsão

EQM6 = mean_squared_error(y_teste, y_predictions6)
resid6 = np.sqrt(EQM6)
print('Test MSE: %.3f' % EQM6,resid6)

accuracy_6 = r2_score(y_teste, y_predictions6)
R2_6_teste = model6_fit.score(X_teste, y_teste) 
print ('accuracy, R2_teste: %.3f' % accuracy_6, R2_6_teste) 


if forecastHorizon>1: 
    plt.figure() 
    pyplot.plot(y_treino, label='Train')
    pyplot.plot(y_teste, color='black', label='Test')
    pyplot.plot(y_predictions6 , label='Forecast')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Power Eletricity Consumption Forecast  (Lasoo Lars)')
    plt.grid()
    plt.savefig('modelBR6_'+str(forecastHorizon)+'lag.png') 
    pyplot.show()




#3) Lasso Lars CV - (7)

print(LassoLarsCV().get_params())
    
def lassoLarsCV_classificator(X_treino, X_teste, y_treino, y_teste):
    eps_LassoLarsCV_list = [1e-8, 1, 10]
    cv_lassoLarsCV_list = [3,10,30]
    
    EQM3_lista=[]
    EQM10_lista=[]
    EQM30_lista=[]
    
    for eps in eps_LassoLarsCV_list:
        start_time = time.time()
        model= LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, 
                           normalize=True, cv=cv_lassoLarsCV_list[0], max_n_alphas=1000, eps=eps, copy_X=True, 
                           positive=False) 
        model_fit=model.fit(X_treino, y_treino)
    
        y_predictions = model_fit.predict(X_teste)
        EQM3_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM3_lista)
        print ("My program took", time.time() - start_time, "seconds to run")
    
    for eps in eps_LassoLarsCV_list:
        start_time = time.time()
        model= LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, 
                           normalize=True, cv=cv_lassoLarsCV_list[1], max_n_alphas=1000, eps=eps, copy_X=True, 
                           positive=False) 
        model_fit=model.fit(X_treino, y_treino)
    
        y_predictions = model_fit.predict(X_teste)
        EQM10_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM10_lista)
        print ("My program took", time.time() - start_time, "seconds to run")
    
    for eps in eps_LassoLarsCV_list:
        start_time = time.time()
        model= LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, 
                           normalize=True, cv=cv_lassoLarsCV_list[2], max_n_alphas=1000, eps=eps, copy_X=True, 
                           positive=False) 
        model_fit=model.fit(X_treino, y_treino)
    
        y_predictions = model_fit.predict(X_teste)
        EQM30_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM30_lista)
        print ("My program took", time.time() - start_time, "seconds to run") 
    
    
    EQM1=pd.DataFrame(columns=['EQM3', 'EQM10','EQM30' ], index=eps_LassoLarsCV_list)
    EQM1['EQM3']=EQM3_lista
    EQM1['EQM10']=EQM10_lista
    EQM1['EQM30']=EQM30_lista

    menorEQM1_list=[sorted(EQM3_lista)[0],sorted(EQM10_lista)[0],sorted(EQM30_lista)[0]] 
    menorEQM1=sorted(menorEQM1_list)[0]
    
    position1=menorEQM1_list.index(menorEQM1)
    menorEQM_col=EQM1.columns[position1]
    
    cv=cv_lassoLarsCV_list[position1]
    eps=EQM1.loc[EQM1[menorEQM_col].isin([menorEQM1])].index[0].item()
    
    print ("My program took", time.time() - start_time, "seconds to run") 
    return cv, eps

cv, eps = lassoLarsCV_classificator(X_treino, X_teste, y_treino, y_teste)


model7= LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, 
                    normalize=True, cv=cv, max_n_alphas=1000, eps=eps, 
                    copy_X=True, positive=False) 
model7_fit = model7.fit(X_treino,y_treino)
print(model7_fit.coef_) 
coef7=model7_fit.coef_ 
R27 = model7_fit.score(X_treino, y_treino) 

y_predictions7 = model7_fit.predict(X_teste)
y_predictions7= pd.DataFrame(y_predictions7, index=teste) #previsão

EQM7 = mean_squared_error(y_teste, y_predictions7)
resid7 = np.sqrt(EQM7)
print('Test MSE: %.3f' % EQM7,resid7)

accuracy_7 = r2_score(y_teste, y_predictions7)
R2_7_teste = model7_fit.score(X_teste, y_teste) 
print ('accuracy, R2_teste: %.3f' % accuracy_7, R2_7_teste)



if forecastHorizon>1: 
    plt.figure() 
    pyplot.plot(y_treino, label='Train')
    pyplot.plot(y_teste, color='black', label='Test')
    pyplot.plot(y_predictions7 , label='Forecast')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Power Eletricity Consumption Forecast  (Lasso Lars CV)')
    plt.grid()
    plt.savefig('modelBR7_'+str(forecastHorizon)+'lag.png') 
    pyplot.show()



#Ridge Regression - (8)

#Olhar Default 
print(Ridge().get_params())

def ridge_classificator(X_treino, X_teste, y_treino, y_teste): 
    alpha_ridge_lista = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2,1e-1, 1, 5, 10]
    EQM_lista=[]
    for alpha in alpha_ridge_lista:
        model = Ridge(alpha=alpha, fit_intercept=True, normalize=True, copy_X=True, 
                      max_iter=None, tol=0.001, random_state=None)
        model_fit=model.fit(X_treino,y_treino)
    
        y_predictions = model_fit.predict(X_teste)
        print(alpha)
        EQM_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
    # print(EQM_lista)
    
    EQM=pd.DataFrame(columns=['alpha', 'resíduo'], index=list(range(len(alpha_ridge_lista))))
    EQM['alpha']=alpha_ridge_lista
    EQM['resíduo']=EQM_lista
    print(EQM.sort_values(by='resíduo', ascending=True)[0:5])
    EQM1=EQM.sort_values(by='resíduo', ascending=True)[0:1]['resíduo'].item()
    alpha=EQM.sort_values(by='resíduo', ascending=True)[0:1]['alpha'].item()
    print('O alpha que apresente o menor erro de previsão é igual a %.1f' % alpha) 
    
    return alpha

alpha = ridge_classificator(X_treino, X_teste, y_treino, y_teste)

model8 = Ridge(alpha=alpha, fit_intercept=True, normalize=True, copy_X=True, 
               max_iter=None, tol=0.001, random_state=None)
model8_fit=model8.fit(X_treino, y_treino)

coef8=np.transpose(model8_fit.coef_) 
R28 = model8_fit.score(X_treino, y_treino) 

y_predictions8 = model8_fit.predict(X_teste)
y_predictions8= pd.DataFrame(y_predictions8, index=teste) #previsão

EQM8 = mean_squared_error(y_teste, y_predictions8)
resid8 = np.sqrt(EQM8)
print('Test MSE: %.3f' % EQM8,resid8)

accuracy_8 = r2_score(y_teste, y_predictions8)
R2_8_teste = model8_fit.score(X_teste, y_teste) 
print ('accuracy, R2_teste: %.3f' % accuracy_8, R2_8_teste) 



if forecastHorizon>1: 
    plt.figure() 
    pyplot.plot(y_treino, label='Train')
    pyplot.plot(y_teste, color='black', label='Test')
    pyplot.plot(y_predictions8 , label='Forecast')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Power Eletricity Consumption Forecast  (Ridge)')
    plt.grid()
    plt.savefig('modelBR8_'+str(forecastHorizon)+'lag.png') 
    pyplot.show() 
    


#ElasticNet 

#1) ElasticNet sem CV - (9)

#Olhar Default 
print(ElasticNet().get_params())


def ElasticNet_classificator(X_treino, X_teste, y_treino, y_teste):
    alpha_EN_list = [1e-15, 1e-5,1e-2, 1, 5,20]
    l1_ratio_list=[0.01,0.25,0.5,0.75,0.99]
    EQM1_lista=[]
    EQM2_lista=[]
    EQM3_lista=[]
    EQM4_lista=[]
    EQM5_lista=[]
    
    for alpha in alpha_EN_list:
        start_time=time.time()
        model=ElasticNet(alpha=alpha, l1_ratio=0.01, fit_intercept=True, normalize=False, 
                         precompute=False, max_iter=1000, copy_X=True, tol=0.0001, 
                         warm_start=False, positive=False, random_state=None) 
        model_fit=model.fit(X_treino, y_treino)
    
        y_predictions = model_fit.predict(X_teste)
        EQM1_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM1_lista)
        print ("My program took", time.time() - start_time, "seconds to run")
    
    for alpha in alpha_EN_list:
        start_time=time.time() 
        model=ElasticNet(alpha=alpha, l1_ratio=0.25, fit_intercept=True, normalize=False, 
                         precompute=False, max_iter=1000, copy_X=True, tol=0.0001, 
                         warm_start=False, positive=False, random_state=None) 
        model_fit=model.fit(X_treino, y_treino)
    
    
        y_predictions = model_fit.predict(X_teste)
        EQM2_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM2_lista)
        print ("My program took", time.time() - start_time, "seconds to run")
    
    for alpha in alpha_EN_list:
        start_time=time.time()
        model=ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, normalize=False, 
                         precompute=False, max_iter=1000, copy_X=True, tol=0.0001, 
                         warm_start=False, positive=False, random_state=None) 
        model_fit=model.fit(X_treino, y_treino)
    
        y_predictions = model_fit.predict(X_teste)
        EQM3_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM3_lista)
        print ("My program took", time.time() - start_time, "seconds to run") 
    
    for alpha in alpha_EN_list:
        start_time=time.time()
        model=ElasticNet(alpha=alpha, l1_ratio=0.75, fit_intercept=True, normalize=False, 
                         precompute=False, max_iter=1000, copy_X=True, tol=0.0001, 
                         warm_start=False, positive=False, random_state=None) 
        model_fit=model.fit(X_treino, y_treino)
        
        y_predictions = model_fit.predict(X_teste)
        EQM4_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM4_lista)
        print ("My program took", time.time() - start_time, "seconds to run")
    
    for alpha in alpha_EN_list:
        start_time=time.time()
        model=ElasticNet(alpha=alpha, l1_ratio=0.99, fit_intercept=True, normalize=False, 
                         precompute=False, max_iter=1000, copy_X=True, tol=0.0001, 
                         warm_start=False, positive=False, random_state=None) 
        model_fit=model.fit(X_treino, y_treino)
        
        y_predictions = model_fit.predict(X_teste)
        EQM5_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM2_lista)
        print ("My program took", time.time() - start_time, "seconds to run")
    
    
    EQM1=pd.DataFrame(columns=['EQM1', 'EQM2','EQM3', 'EQM4','EQM5'], index=alpha_EN_list)
    EQM1['EQM1']=EQM1_lista
    EQM1['EQM2']=EQM2_lista
    EQM1['EQM3']=EQM3_lista
    EQM1['EQM4']=EQM4_lista
    EQM1['EQM5']=EQM5_lista
    
    menorEQM1_list=[sorted(EQM1_lista)[0],sorted(EQM2_lista)[0],sorted(EQM3_lista)[0],
    sorted(EQM4_lista)[0],sorted(EQM5_lista)[0]] 
    menorEQM1=sorted(menorEQM1_list)[0]
    
    position1=menorEQM1_list.index(menorEQM1)
    menorEQM_col=EQM1.columns[position1]
    
    l1_ratio=l1_ratio_list[position1]
    alpha=EQM1.loc[EQM1[menorEQM_col].isin([menorEQM1])].index[0].item()
    
    print('O epsilon e o número de coeficientes diferente de zero que apresentam'
    'o menor erro de previsão são iguais a %.1f' % eps,nZeroCoef ) 
    return l1_ratio, alpha


l1_ratio, alpha = ElasticNet_classificator(X_treino, X_teste, y_treino, y_teste)

model90=ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, normalize=False, 
                   precompute=False, max_iter=1000, copy_X=True, tol=0.0001, 
                   warm_start=False, positive=False, random_state=None) 
model90_fit=model90.fit(X_treino, y_treino)

y_predictions90 = model90_fit.predict(X_teste)

R290 = model90.score(X_treino, y_treino) 
coef90=model90.coef_

y_predictions90 = model90.predict(X_teste)
y_predictions90= pd.DataFrame(y_predictions90, index=teste) #previsão

EQM90 = mean_squared_error(y_teste, y_predictions90)
resid90 = np.sqrt(EQM90)
print('Test MSE: %.3f' % EQM90,resid90)

accuracy_90 = r2_score(y_teste, y_predictions90)
R2_90_teste = model90_fit.score(X_teste, y_teste) 
print ('accuracy, R2_teste: %.3f' % accuracy_90, R2_90_teste) 


if forecastHorizon>1: 
    plt.figure() 
    pyplot.plot(y_treino, label='Train')
    pyplot.plot(y_teste, color='black', label='Test')
    pyplot.plot(y_predictions90 , label='Forecast')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Power Eletricity Consumption Forecast  (Elastic Net)')
    plt.grid()
    plt.savefig('modelBR9_'+str(forecastHorizon)+'lag.png') 
    pyplot.show() 



#2) ElasticNetCV - (90)

#Verificar Default
print(ElasticNetCV().get_params())

def ElasticNetCV_classificator(X_treino, X_teste, y_treino, y_teste):
    l1_ratio_list=[0.25,0.5,0.75]
    eps_ENCV_list = [1e-8,1,10]
    EQM1_lista=[]
    EQM2_lista=[]
    EQM3_lista=[]
    
    for l1_ratio in l1_ratio_list:
        start_time=time.time()
        model=ElasticNetCV(alphas=None, copy_X=True, eps=eps_ENCV_list[0], fit_intercept=True,
                           l1_ratio=l1_ratio, max_iter=1000, n_alphas=100, n_jobs=None,
                           normalize=True, positive=False, precompute='auto', random_state=0,
                           selection='cyclic', tol=0.0001, verbose=0) 
    
        model_fit=model.fit(X_treino, y_treino)
    
        y_predictions = model_fit.predict(X_teste)
        EQM1_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM1_lista)
        # print ("My program took", time.time() - start_time, "seconds to run")
    
    
    for l1_ratio in l1_ratio_list:
        start_time=time.time()
        model=ElasticNetCV(alphas=None, copy_X=True, eps=eps_ENCV_list[1], fit_intercept=True,
                           l1_ratio=l1_ratio, max_iter=1000, n_alphas=100, n_jobs=None,
                           normalize=True, positive=False, precompute='auto', random_state=0,
                           selection='cyclic', tol=0.0001, verbose=0) 
        
        model_fit=model.fit(X_treino, y_treino)
        EQM2_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM2_lista)
        print ("My program took", time.time() - start_time, "seconds to run")
        
    for l1_ratio in l1_ratio_list:
        start_time=time.time()
        model=ElasticNetCV(alphas=None, copy_X=True, eps=eps_ENCV_list[2], fit_intercept=True,
                           l1_ratio=l1_ratio, max_iter=1000, n_alphas=100, n_jobs=None,
                           normalize=True, positive=False, precompute='auto', random_state=0,
                           selection='cyclic', tol=0.0001, verbose=0) 
    
        model_fit=model.fit(X_treino, y_treino)
        EQM3_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM3_lista)
        print ("My program took", time.time() - start_time, "seconds to run")
    
    
    EQM1=pd.DataFrame(columns=['EQM1', 'EQM2','EQM3'], index=l1_ratio_list)
    EQM1['EQM1']=EQM1_lista
    EQM1['EQM2']=EQM2_lista
    EQM1['EQM3']=EQM3_lista
    
    menorEQM1_list=[sorted(EQM1_lista)[0],sorted(EQM2_lista)[0],sorted(EQM3_lista)[0]] 
    menorEQM1=sorted(menorEQM1_list)[0]
    
    position1=menorEQM1_list.index(menorEQM1)
    menorEQM_col=EQM1.columns[position1]
    
    eps=eps_ENCV_list[position1]
    l1_ratio=EQM1.loc[EQM1[menorEQM_col].isin([menorEQM1])].index[0].item() 
    
    print ("My program took", time.time() - start_time, "seconds to run") 
    return l1_ratio, eps


l1_ratio, eps = ElasticNetCV_classificator(X_treino, X_teste, y_treino, y_teste)

model9 = ElasticNetCV(alphas=None, copy_X=True, eps=eps, fit_intercept=True,
                      l1_ratio=l1_ratio, max_iter=1000, n_alphas=100, n_jobs=None,
                      normalize=True, positive=False, precompute='auto', random_state=0,
selection='cyclic', tol=0.0001, verbose=0).fit(X_treino,y_treino)

model9_fit=model9.fit(X_treino,y_treino)
print(model9_fit.coef_) 

R29 = model9_fit.score(X_treino, y_treino) 
coef9=model9_fit.coef_

y_predictions9 = model9_fit.predict(X_teste)
y_predictions9= pd.DataFrame(y_predictions9, index=teste) #previsão

EQM9 = mean_squared_error(y_teste, y_predictions9)
resid9 = np.sqrt(EQM9)
print('Test MSE: %.3f' % EQM9,resid9)

accuracy_9 = r2_score(y_teste, y_predictions9)
R2_9_teste = model9_fit.score(X_teste, y_teste) 
print ('accuracy, R2_teste: %.3f' % accuracy_9, R2_9_teste)


if forecastHorizon>1: 
    plt.figure() 
    pyplot.plot(y_treino, label='Train')
    pyplot.plot(y_teste, color='black', label='Test')
    pyplot.plot(y_predictions9 , label='Forecast')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Power Eletricity Consumption Forecast  (Elastic Net CV)')
    #plt.grid()
    plt.savefig('modelBR9.png') 
    pyplot.show()


#Random Forest - (10)

# Look at parameters used by our current forest
pprint(RandomForestRegressor().get_params())
   
def RandomForest_classificator(X_treino, X_teste, y_treino, y_teste):
    n_estimators_list=[10,50,100,1000]
    EQM1_lista=[]
    
    for n_estimators in n_estimators_list:
        model=RandomForestRegressor(n_estimators=n_estimators, max_depth=None, min_samples_split=2, 
                                    min_samples_leaf=1, min_weight_fraction_leaf=0, max_leaf_nodes=None, 
                                    bootstrap=True, oob_score=False,n_jobs=1, random_state=None, 
                                    verbose=0, warm_start=False, max_features=None)
        model_fit=model.fit(X_treino, y_treino)
        y_predictions = model_fit.predict(X_teste)
    
        EQM1_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
        print(EQM1_lista)
        
        
    EQM=pd.DataFrame(columns=['n_estimators', 'resíduo'], index=list(range(len(n_estimators_list))))
    EQM['n_estimators']=n_estimators_list
    EQM['resíduo']=EQM1_lista
    
    print(EQM.sort_values(by='resíduo', ascending=True)[0:5])
    n_estimators=EQM.sort_values(by='resíduo', ascending=True)[0:1]['n_estimators'].item()
    
    print('O numero de estimatores que apresenta o menor erro de previsão é igual a %.1f' % n_estimators) 
    
    return n_estimators
    

n_estimators = RandomForest_classificator(X_treino, X_teste, y_treino, y_teste)

model10=RandomForestRegressor(n_estimators=n_estimators, max_depth=None, min_samples_split=2, 
                              min_samples_leaf=1, min_weight_fraction_leaf=0, max_leaf_nodes=None, 
                              bootstrap=True, oob_score=False,n_jobs=1, random_state=None, 
                              verbose=0, warm_start=False, max_features=None)

model10_fit=model10.fit(X_treino, y_treino)

print(model10_fit.feature_importances_)
coef10=model10_fit.feature_importances_

R210 = model10_fit.score(X_treino, y_treino) 

y_predictions10 = model10_fit.predict(X_teste)
y_predictions10= pd.DataFrame(y_predictions10, index=teste) #previsão

EQM10 = mean_squared_error(y_teste, y_predictions10)
resid10 = np.sqrt(EQM10)
print('Test MSE: %.3f' % EQM10,resid10)

accuracy_10 = r2_score(y_teste, y_predictions10)
R2_10_teste = model10_fit.score(X_teste, y_teste) 
print ('accuracy, R2_teste: %.3f' % accuracy_10, R2_10_teste) 


if forecastHorizon>1: 
    plt.figure() 
    pyplot.plot(y_treino, label='Train')
    pyplot.plot(y_teste, color='black', label='Test')
    pyplot.plot(y_predictions10 , label='Forecast')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Power Eletricity Consumption Forecast  (Random Forest)')
    plt.grid()
    plt.savefig('modelBR10_'+str(forecastHorizon)+'lag.png') 
    pyplot.show()




colunas2 = ['DM','DS','MÊS','ANO','ESTAC','FER','NEB_BR_9','NEB_BR_15','NEB_BR_21',
'NEB_BR_MED','PA_BR_9','PA_BR_15','PA_BR_21','PA_BR_MED','TEMP_BS_BR_9','TEMP_BS_BR_15','TEMP_BS_BR_21','TEMP_BS_BR_MED','TEMP_BU_BR_9','TEMP_BU_BR_15','TEMP_BU_BR_21','TEMP_BU_BR_MED','UMID_BR_9', 
'UMID_BR_15','UMID_BR_21','UMID_BR_MED','DV_BR_9','DV_BR_15','DV_BR_21','DV_BR_MED','VV_BR_9','VV_BR_15','VV_BR_21','VV_BR_MED','TAR_BR_CSO','TAR_BR_CP','TAR_BR_IP','TAR_BR_IND','TAR_BR_PP','TAR_BR_RED',
'TAR_BR_RR','TAR_BR_RRA','TAR_BR_RRI','TAR_BR_SP1','TAR_BR_SP2','TAR_BR_MED','Meta_Selic', 'Taxa_Selic','CDI','DolarC','DolarC_var','DolarV','DolarV_var','EuroC','EuroC_var','EuroV','EuroV_var','IBV_Cot',
'IBV_min','IBV_max','IBV_varabs','IBV_varperc','IBV_vol','INPC_m','INPC_ac','IPCA_m','IPCA_ac','IPAM_m','IPAM_ac','IPADI_m', 'IPADI_ac' , 'IGPM_m','IGPM_ac','IGPDI_m','IGPDI_ac','PAB_o','PAB_d',
'TVP_o','TVP_d','PICV_o','ICV_d','CCU_o','CCU_d','CS_o','CS_d','UCPIIT_FGV_o','UCPIIT_FGV_d','CPCIIT_CNI_o','CPCIIT_CNI_d','VIR_o','VIR_d','HTPIT_o','HTPIT_d','SRIT_o','SRIT_d','PPOB','PGN','PIG_o','PIG_d','PIBCa_o','PIBCa_d',
'PIBI_o','PIBI_d','PIBCo_o','PIBCo_d','PIA_o','PIA_d','ICC','INEC','ICEI','DBNDES','IEG_o','IEG_d','IETIT_o','IETIT_d','IETC_o','IETC_d','IETS_o','IETS_d','IETCV_o','IETCV_d', 'PO','TD','BM','PME',
'TEMPaj_BS_BR_9','TEMPaj_BS_BR_15','TEMPaj_BS_BR_21','TEMPaj_BS_BR_MED','TEMPaj_BU_N_MED','TEMPaj_BU_BR_9','TEMPaj_BU_BR_15','TEMPaj_BU_BR_21','TEMPaj_BU_BR_MED'] 



#OLS vs LR
if EQM3>EQM30:
    print("LR tem um erro de previsão menor")
    EQM3=EQM30
    coef3=coef30
    resid3=resid30
    accuracy_3=accuracy_30
    R2_3_teste=R2_30_teste


#LASSO vs LASSOCV
if EQM4>EQM40:
    print("Lasso com CV tem um erro de previsão menor")
    EQM4=EQM40
    coef4=coef40
    resid4=resid40
    accuracy_4=accuracy_40
    R2_4_teste=R2_40_teste

#LARS vs LARSCV
if EQM6>EQM7:
    print("LassoLars com CV tem um erro de previsão menor")
    EQM6=EQM7
    coef6=coef7
    resid6=resid7
    accuracy_6=accuracy_7
    R2_6_teste=R2_7_teste

#EN vs ENCV - OK
if EQM9>EQM90:
    print("ElasticNet com CV tem um erro de previsão menor")
    EQM9=EQM90
    coef9=coef90
    resid9=resid90
    accuracy_9=accuracy_90
    R2_9_teste=R2_90_teste

  

coef = pd.DataFrame(coef3, index=X.columns)
coef.columns = ['Linear Regression']
coef['Lasso']=coef4
coef['Lars']=coef5
coef['Lasso Lars']=coef6
coef['Ridge']=coef8
coef['Elastic Net']=coef9 
coef['Random Forest']=coef10



R2_list = [R21,0,R23,R24, R25,R26,R28,R29,R210] 

EQM_list = [EQM1,EQM2,EQM3,EQM4,EQM5,EQM6,EQM8,EQM9,EQM10]

resid_list = [resid1,resid2,resid3,resid4, resid5,resid6,resid8,resid9,resid10]

accuracy_list = [accuracy_1,accuracy_2,accuracy_3,accuracy_4,accuracy_5,accuracy_6,accuracy_8,accuracy_9,accuracy_10]

R2_test_list = [R2_1_teste,0,R2_3_teste,R2_4_teste, R2_5_teste,R2_6_teste,R2_8_teste,R2_9_teste,R2_10_teste]


index=['Modelos','R2', 'EQM', 'Resíduo','Accuracy', 'R2 Teste']    
colunas3 = ['AR','Random Walk','Linear Regression','Lasso','Lars','Lasso Lars',
'Ridge','Elastic Net','Random Forest']


previsao = pd.DataFrame([colunas3,R2_list, EQM_list, resid_list, accuracy_list,R2_test_list ],index=index, columns=list(range(1,10)))
previsao=previsao.transpose()
print(previsao)



if forecastHorizon==1:
    y_verd=y_teste.values[0]
    comparacao = pd.DataFrame(index=colunas3,)
    comparacao['Preço Verdadeiro']=y_verd 
    comparacao['Preço Estimado']=0
    comparacao['Preço Estimado'][:1]=y_predictions1+y_treino[-1:].values[0]
    comparacao['Preço Estimado'][1:2]=y_predictions2
    comparacao['Preço Estimado'][2:3]=y_predictions3
    comparacao['Preço Estimado'][3:4]=y_predictions4
    comparacao['Preço Estimado'][4:5]=y_predictions5
    comparacao['Preço Estimado'][5:6]=y_predictions6
    comparacao['Preço Estimado'][6:7]=y_predictions8
    comparacao['Preço Estimado'][7:8]=y_predictions9
    comparacao['Preço Estimado'][8:9]=y_predictions10 
    comparacao['Diferenca']= comparacao['Preço Verdadeiro']- comparacao['Preço Estimado'] 
    print(comparacao)

#Plot4
print(previsao.sort_values(by='Resíduo', ascending=True)[0:5]['Resíduo'])

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
    

if forecastHorizon>1:    
    fig2, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 7))
    axes[0].set_title(previsao.sort_values(by='Resíduo', ascending=True)[0:1]['Modelos'].item())
    axes[0].plot (y_treino, label='Train')
    axes[0].plot(y_teste, color='black', label='Test')
    axes[0].plot(previsao_aux[previsao.sort_values(by='Resíduo', ascending=True)[0:1].index.item()], label='Forecast')
    axes[0].legend(loc='best')
    axes[0].set_ylabel("KW/h")
    
    axes[1].set_title(previsao.sort_values(by='Resíduo', ascending=True)[1:2]['Modelos'].item())
    axes[1].plot (y_treino, label='Train')
    axes[1].plot(y_teste, color='black', label='Test')
    axes[1].plot(previsao_aux[previsao.sort_values(by='Resíduo', ascending=True)[1:2].index.item()], label='Forecast')   
    axes[1].legend(loc='best')
    axes[1].set_ylabel("KW/h")
    
    axes[2].set_title(previsao.sort_values(by='Resíduo', ascending=True)[2:3]['Modelos'].item())
    axes[2].plot (y_treino, label='Train')
    axes[2].plot(y_teste, color='black', label='Test')
    axes[2].plot(previsao_aux[previsao.sort_values(by='Resíduo', ascending=True)[2:3].index.item()], label='Forecast') 
    axes[2].legend(loc='best')
    axes[2].set_ylabel("KW/h")
    
    axes[3].set_title(previsao.sort_values(by='Resíduo', ascending=True)[3:4]['Modelos'].item())
    axes[3].plot (y_treino, label='Train') 
    axes[3].plot(y_teste, color='black', label='Test')
    axes[3].plot(previsao_aux[previsao.sort_values(by='Resíduo', ascending=True)[3:4].index.item()], label='Forecast')   
    axes[3].legend(loc='best')
    axes[3].set_ylabel("KW/h")
    fig2.tight_layout()
    plt.savefig('modelo'+str(forecastHorizon)+'dias.png')   


writer = pd.ExcelWriter('dados_lag'+str(forecastHorizon)+'_modo'+str(modo)+'.xlsx')
coef.to_excel(writer,'coeficientes_lag'+str(forecastHorizon))
previsao.set_index('Modelos')
previsao.to_excel(writer,'previsao_lag'+str(forecastHorizon))
if forecastHorizon==1:
    comparacao.to_excel(writer,'comparacao_lag'+str(forecastHorizon))
writer.save()

