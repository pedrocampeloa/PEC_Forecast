#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:33:42 2018

@author: pedrocampelo
"""

 
    import os
    
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    
    import pandas as pd
    import datetime as dt    
    
    from pandas import Series
    from pandas import read_csv 
    
    from statsmodels.tsa.arima_model import ARIMA
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import accuracy_score
    import statsmodels.api as sm
    from sklearn.metrics import r2_score

    
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
    

    #Definir Horizonte de Previsão de X dias:    
    forecastHorizon=30
    
    #A partir dos dois gráficos podemos decidir quantos lags usar
    #Vamos usar 3 lags para rodar os modelos
    
    y = dados['CEE_BR_TOT']
    colunas=dados.columns
    X = pd.DataFrame(index=periodo)
    
#    #modelo1
#    for i in colunas:
#        X[i+'lag1']=dados[i].shift(forecastHorizon)
#        
#    for i in colunas:
#        X[i+'lag2']=dados[i].shift(forecastHorizon+1)
#        
#    for i in colunas:
#        X[i+'lag3']=dados[i].shift(forecastHorizon+2)
#   
     
    #modelo2
    for i in colunas:
        X[i+'lag1']=dados[i].shift(round(forecastHorizon))
        
    for i in colunas:
        X[i+'lag2']=dados[i].shift(round(forecastHorizon*2))
        
    for i in colunas:
        X[i+'lag3']=dados[i].shift(round(forecastHorizon*3))
    
    #modelo3
#    for i in colunas:
#        X[i+'lag1']=dados[i].shift(round(forecastHorizon))
#        
#    for i in colunas:
#        X[i+'lag2']=dados[i].shift(round(forecastHorizon*(3/2)))
#        
#    for i in colunas:
#        X[i+'lag3']=dados[i].shift(round(forecastHorizon*(9/4)))
        
    #modelo4
#    for i in colunas:
#        X[i+'lag1']=dados[i].shift(round(forecastHorizon))
#        
#    for i in colunas:
#        X[i+'lag2']=dados[i].shift(round(forecastHorizon+7))
#        
#    for i in colunas:
#        X[i+'lag3']=dados[i].shift(round(forecastHorizon+14))  
    
        
    
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
    from pandas import DataFrame
    from pandas import concat
    
    values = DataFrame(dados['CEE_BR_TOT'].values)
    dataframe = concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t-1','t+1']
    result = dataframe.corr()
    print(result)

   
    from plotly.plotly import plot_mpl
    from statsmodels.tsa.seasonal import seasonal_decompose
   
    dec_seas = seasonal_decompose(y, model='multiplicative')
    fig = dec_seas.plot()
    plt.savefig('CEEBRseas.png')   



    #Autocorrelação PLOT   
    from pandas import Series
    from matplotlib import pyplot
    from pandas.tools.plotting import autocorrelation_plot
    from statsmodels.graphics.tsaplots import plot_acf

    
    #Brasil
    autocorrelation_plot(dados['CEE_BR_TOT'])
    plt.title('Gráfico de Autocorrelação do CEE do Brasil')
    plt.savefig('CEEBRautocorr.png')   
    plt.show()

    #Agregado
    fig, axes = plt.subplots(4,1, sharey=True, sharex=True)
    fig = autocorrelation_plot(dados['CEE_SECO_TOT'], ax=axes[0])
    fig = autocorrelation_plot(dados['CEE_SUL_TOT'], ax=axes[1])
    fig = autocorrelation_plot(dados['CEE_NE_TOT'], ax=axes[2])
    fig = autocorrelation_plot(dados['CEE_N_TOT'], ax=axes[3])
#    fig.tight_layout()
    plt.savefig('CEEautocorragreg.png')  
    plt.show()
  
    

    #Lag Plot
    from pandas import Series
    from matplotlib import pyplot
    
    import statsmodels.api as sm

    plot_acf(dados['CEE_BR_TOT'], lags=31)
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
    #AR
    from statsmodels.tsa.arima_model import ARIMA
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import accuracy_score
    import statsmodels.api as sm
    from sklearn.metrics import r2_score

 
    #outro modelo arima (consegue decidir o numero de lags)
    model1 = ARIMA(y_treino, order=(15,1,0))
    model_fit1 = model1.fit()
    print('Lag: %s' % model_fit1.k_ar)
    print('Coefficients: %s' % model_fit1.params)
    R21=r2_score(y_treino, model_fit1.predict(start=1, end=len(y_treino), dynamic=False))
    
    # make predictions
    y_predictions1 = model_fit1.predict(start=len(y_treino), end=len(y_treino)+len(y_teste)-1, dynamic=False)
    EQM1 = mean_squared_error(y_teste, y_predictions1)
    resid1 = np.sqrt(EQM1)   
    print('Test MSE: %.3f' % EQM1, resid1)
    print(model_fit1.summary())
    
    accuracy_1 = r2_score(y_teste, y_predictions1)
    R2_1_teste = accuracy_1 
    print ('accuracy, R2_teste: %.3f' % accuracy_1, R2_1_teste) 

#    if forecastHorizon>1: 
#        plt.figure()    
#        pyplot.plot(y_treino_diff, label='Treino')
#        pyplot.plot(y_teste_diff, color='black', label='Teste')
#        pyplot.plot(y_predictions1, color='red', label='Previsão')
#        #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
#        plt.legend(loc='best')
#        plt.ylabel('KW/h')
#        plt.xticks(rotation=30)
#        plt.title('Previsão Consumo de Energia Elétrica (AR)')
#        #plt.grid()
#        plt.savefig('modelBR1_'+str(forecastHorizon)+'lag.png')  
#        pyplot.show()
#


    
                               #OLS                
    
    import statsmodels.api as sm
    from sklearn.metrics import r2_score

    model2 = sm.OLS(y_treino,X_treino)                  #modelo
    model_fit2 = model2.fit() 
    print (model_fit2.summary())                        #sumário do modelo
    coef2=model_fit2.params
    
    R22=model_fit2.rsquared
    
        
    # make predictions
    y_predictions2 = model_fit2.predict(X_teste)          #previsão

    EQM2 = mean_squared_error(y_teste, y_predictions2)    #EQM
    resid2 = np.sqrt(EQM2)                                #Resíduo

    print('Test MSE, Residual: %.3f' % EQM2, resid2)
        
    accuracy_2 = r2_score(y_teste, y_predictions2)
    R2_2_teste = sm.OLS(y_teste,X_teste).fit().rsquared
    print ('accuracy, R2_teste: %.3f' % accuracy_2, R2_2_teste)
    
    if forecastHorizon>1: 
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions2, color='red', label='Previsão')
        #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
        plt.legend(loc='best')
        plt.ylabel('KW/h')
        plt.xticks(rotation=30)
        plt.title('Previsão Consumo de Energia Elétrica (OLS)')
        #plt.grid()
        plt.savefig('modelBR2_'+str(forecastHorizon)+'lag.png')   
        pyplot.show()   

    
    
    
    #2)Linear Regression
  
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    modelo=LinearRegression
        
    reg = modelo().fit(X_treino, y_treino)
    print(reg.score(X_treino, y_treino))                        #R2 fora da amostra
    print(reg.coef_)                                            #coeficientes
    coefreg=np.transpose(reg.coef_)

    R2reg=reg.score(X_treino, y_treino)    
        
    predictionsreg = reg.predict(X_teste)
    y_predictionsreg= pd.DataFrame(predictionsreg, index=teste)   #previsão
    
    EQMreg = mean_squared_error(y_teste, y_predictionsreg)      #EQM
    residreg = np.sqrt(EQMreg)                                #Residuo
    print('Test MSE, residuo: %.3f' % EQMreg,residreg)
    
    accuracy_reg = r2_score(y_teste, y_predictionsreg)
    R2_reg_teste = reg.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_reg, R2_reg_teste)

    if forecastHorizon>1:     
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictionsreg, color='red', label='Previsão')
        #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
        plt.legend(loc='best')
        plt.ylabel('KW/h')
        plt.xticks(rotation=30)
        plt.title('Previsão Consumo de Energia Elétrica (LR)')
        #plt.grid()
        plt.savefig('modelBRreg_'+str(forecastHorizon)+'lag.png')   
        pyplot.show()  




                                            #Lasso
    
    #1)Lasso normal
    
    from sklearn import linear_model
    
    def lasso_classificator(X_treino, X_teste, y_treino, y_teste):    
        alpha_lasso_lista = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2,1e-1, 1, 5, 10,20]
        EQM_lista=[]
        for alpha in alpha_lasso_lista:
            model = linear_model.Lasso(alpha=alpha, copy_X=True, fit_intercept=True, max_iter=1000,
            normalize=True, positive=False, precompute=False, random_state=None,
            selection='cyclic', tol=0.0001, warm_start=False)
            model_fit=model.fit(X_treino,y_treino)
           
            y_predictions = model_fit.predict(X_teste)
            EQM_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
#            print(EQM_lista)
            
        EQM=pd.DataFrame(columns=['alpha', 'resíduo'], index=list(range(len(alpha_lasso_lista))))
        EQM['alpha']=alpha_lasso_lista
        EQM['resíduo']=EQM_lista
        print(EQM.sort_values(by='resíduo', ascending=True)[0:5])
        alpha=EQM.sort_values(by='resíduo', ascending=True)[0:1]['alpha'].item()
        print('O alpha que apresente o menor erro de previsão é igual a %.1f' % alpha)        
        return alpha
    
   
    alpha=lasso_classificator(X_treino, X_teste, y_treino, y_teste)   
        
    model3 = linear_model.Lasso(alpha=alpha, copy_X=True, fit_intercept=True, max_iter=1000,
        normalize=True, positive=False, precompute=False, random_state=None,
        selection='cyclic', tol=0.0001, warm_start=False)
    model_fit3=model3.fit(X_treino,y_treino)

    coef3=model3.coef_
    R23 = model_fit3.score(X_treino,y_treino)
    
        # make predictions
    y_predictions3 = model_fit3.predict(X_teste)
    y_predictions3= pd.DataFrame(y_predictions3, index=teste)   #previsão


    EQM3 = mean_squared_error(y_teste, y_predictions3)
    resid3 = np.sqrt(EQM3)
    print('Test MSE, residuo: %.3f' % EQM3,resid3)
    
    accuracy_3 = r2_score(y_teste, y_predictions3)
    R2_3_teste = model_fit3.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_3, R2_3_teste)
    
    if forecastHorizon>1:       
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions3, color='red', label='Previsão')
        #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
        plt.legend(loc='best')
        plt.ylabel('KW/h')
        plt.xticks(rotation=30)
        plt.title('Previsão Consumo de Energia Elétrica (LASSO)')
        #plt.grid()
        plt.savefig('modelBR3_'+str(forecastHorizon)+'lag.png')   
        pyplot.show() 

    

#    
#    #2) Lasso CV
#    
#    from sklearn.linear_model import LassoCV
# 
#    model4 = LassoCV(cv=365, random_state=0).fit(X_treino, y_treino)
#    print(model4.coef_)
#    coef4=model4.coef_ 
#    R24 = model4.score(X_treino, y_treino) 
#
#        # make predictions
#    y_predictions4 = model4.predict(X_teste)
#    y_predictions4= pd.DataFrame(y_predictions4, index=teste)   #previsão
#
#    EQM4 = mean_squared_error(y_teste, y_predictions4)
#    resid4 = np.sqrt(EQM4)
#    print('Test MSE: %.3f' % EQM4,resid4)
#    
#    accuracy_4 = r2_score(y_teste, y_predictions4)
#    R2_4_teste = model4.score(X_teste, y_teste)  
#    print ('accuracy, R2_teste: %.3f' % accuracy_4, R2_4_teste)
#           
#    if forecastHorizon>1: 
#        plt.figure()    
#        pyplot.plot(y_treino, label='Treino')
#        pyplot.plot(y_teste, color='black', label='Teste')
#        pyplot.plot(y_predictions4, color='red', label='Previsão')
#        #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
#        plt.legend(loc='best')
#        plt.ylabel('KW/h')
#        plt.xticks(rotation=30)
#        plt.title('Previsão Consumo de Energia Elétrica (LassoCV)')
#        plt.grid()
#        plt.savefig('modelBR4.png')   
#        pyplot.show() 

    
    #XGboostin
#    import xbgoost
#    
#    from xgboost.sklearn import XGBClassifier  
#    from xgboost.sklearn import XGBRegressor
#    
#    xclas = XGBClassifier()  # and for classifier  
#    xclas.fit(X_train, y_train)  
#    xclas.predict(X_test) 
#        
                                        #Lars
                                        
    #1) Lars 
    
    from sklearn.linear_model import Lars
    
    def lars_classificator(X_treino, X_teste, y_treino, y_teste):
        eps_list = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10,20]
        EQM1_lista=[]
        EQM10_lista=[]
        EQM50_lista=[]
        EQM100_lista=[]
        EQM500_lista=[]
        
        for eps in eps_list:
                model=Lars(fit_intercept=True, verbose=False, normalize=True, 
                                            n_nonzero_coefs=1, eps=eps, 
                                            copy_X=True, fit_path=True, positive=False)      
                model_fit=model.fit(X_treino, y_treino)
                
                y_predictions = model_fit.predict(X_teste)
                EQM1_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
                
        for eps in eps_list:
                model=Lars(fit_intercept=True, verbose=False, normalize=True, 
                                            n_nonzero_coefs=10, eps=eps, 
                                            copy_X=True, fit_path=True, positive=False)      
                model_fit=model.fit(X_treino, y_treino)
                
                y_predictions = model_fit.predict(X_teste)
                EQM10_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
                
        for eps in eps_list:
                model=Lars(fit_intercept=True, verbose=False, normalize=True, 
                                            n_nonzero_coefs=50, eps=eps, 
                                            copy_X=True, fit_path=True, positive=False)      
                model_fit=model.fit(X_treino, y_treino)
                
                y_predictions = model_fit.predict(X_teste)
                EQM50_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
                
                
        for eps in eps_list:
                model=Lars(fit_intercept=True, verbose=False, normalize=True, 
                                            n_nonzero_coefs=100, eps=eps, 
                                            copy_X=True, fit_path=True, positive=False)      
                model_fit=model.fit(X_treino, y_treino)
                
                y_predictions = model_fit.predict(X_teste)
                EQM100_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
                
        for eps in eps_list:
                model=Lars(fit_intercept=True, verbose=False, normalize=True, 
                                            n_nonzero_coefs=500, eps=eps, 
                                            copy_X=True, fit_path=True, positive=False)      
                model_fit=model.fit(X_treino, y_treino)
                
                y_predictions = model_fit.predict(X_teste)
                EQM500_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))

                
        EQM=pd.DataFrame(columns=['EQM1', 'EQM10','EQM50', 'EQM100','EQM500'], index=eps_list)
        EQM['EQM1']=EQM1_lista
        EQM['EQM10']=EQM10_lista
        EQM['EQM50']=EQM50_lista
        EQM['EQM100']=EQM100_lista
        EQM['EQM500']=EQM500_lista
        
        EQM1_sorted=sorted(EQM['EQM1'])        
        EQM10_sorted=sorted(EQM['EQM10'])        
        EQM50_sorted=sorted(EQM['EQM50'])        
        EQM100_sorted=sorted(EQM['EQM100'])        
        EQM500_sorted=sorted(EQM['EQM500'])
        
        menorEQM_list=[EQM1_sorted[0],EQM10_sorted[0],EQM50_sorted[0],
                  EQM100_sorted[0],EQM500_sorted[0]]        
        menorEQM=sorted(menorEQM_list)[0]
        
        position=menorEQM_list.index(menorEQM)
        menorEQM_col=EQM.columns[position]

        nZeroCoef=float(EQM.columns[position][3:])
        eps=EQM.loc[EQM[menorEQM_col].isin([menorEQM])].index.item()
        print('O epsilon e o número de coeficientes diferente de zero que apresentam'
              'o menor erro de previsão são iguais a %.1f' % eps,nZeroCoef )               

        return eps,nZeroCoef
    
    eps,nZeroCoef = lars_classificator(X_treino, X_teste, y_treino, y_teste)
        
    model5=Lars(fit_intercept=True, verbose=False, normalize=True, 
                                            n_nonzero_coefs=int(nZeroCoef), eps=eps, 
                                            copy_X=True, fit_path=True, positive=False)      
    model5_fit=model5.fit(X_treino, y_treino)    
    R25 = model5_fit.score(X_treino, y_treino)  
    
             # make predictions
    y_predictions5 = model5_fit.predict(X_teste)
    y_predictions5= pd.DataFrame(y_predictions5, index=teste)   #previsão
    
    EQM5 = mean_squared_error(y_teste, y_predictions5)
    resid5 = np.sqrt(EQM5)
    print('Test MSE: %.3f' % EQM5,resid5)
         
    accuracy_5 = r2_score(y_teste, y_predictions5)
    R2_5_teste = model5_fit.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_5, R2_5_teste)      
            
    if forecastHorizon>1: 
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions5, color='red', label='Previsão')
        #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
        plt.legend(loc='best')
        plt.ylabel('KW/h')
        plt.xticks(rotation=30)
        plt.title('Previsão Consumo de Energia Elétrica (LARS)')
        plt.grid()
        plt.savefig('modelBR5_'+str(forecastHorizon)+'lag.png')   
        pyplot.show()

     
        
        
        
        
        
    #Lasso Lars
    #trabalhoso (defininir alpha e eps)

    
    #2) Lasso Lars (3 MENOR EQM)
    from sklearn.linear_model import LassoLars    
    
    LassoLars(alpha=alpha, fit_intercept=True, verbose=False, normalize=True, 
              precompute=’auto’, max_iter=500, eps=eps, copy_X=True,
              fit_path=True, positive=False)
        
    def lassoLars_classificator(X_treino, X_teste, y_treino, y_teste):    
        alpha_lassoLars_lista = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2,1e-1,1, 5, 10,20]
        EQM_lista=[]
        for alpha in alpha_lassoLars_lista:
            model = LassoLars(alpha=alpha, fit_intercept=True, verbose=False, normalize=True, 
              precompute=’auto’, max_iter=500, eps=eps, copy_X=True,
              fit_path=True, positive=False)
            model_fit=model.fit(X_treino,y_treino)
           
            y_predictions = model_fit.predict(X_teste)
            EQM_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
#            print(EQM_lista)
            
        EQM=pd.DataFrame(columns=['alpha', 'resíduo'], index=list(range(len(alpha_lassoLars_lista))))
        EQM['alpha']=alpha_ridge
        EQM['resíduo']=EQM_lista
        print(EQM.sort_values(by='resíduo', ascending=True)[0:5])
        alpha=EQM.sort_values(by='resíduo', ascending=True)[0:1]['alpha'].item()
        print('O alpha que apresente o menor erro de previsão é igual a %.1f' % alpha)        
        return alpha


    model6 = linear_model.LassoLars(alpha=0.01).fit(X_treino,y_treino)
    print(model6.coef_) 
    coef6=model6.coef_
    
    R26 = model6.score(X_treino, y_treino) 
  
    y_predictions6 = model6.predict(X_teste)
    y_predictions6= pd.DataFrame(y_predictions6, index=teste)   #previsão

    EQM6 = mean_squared_error(y_teste, y_predictions6)
    resid6 = np.sqrt(EQM6)
    print('Test MSE: %.3f' % EQM6,resid6)
    
    accuracy_6 = r2_score(y_teste, y_predictions6)
    R2_6_teste = model6.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_6, R2_6_teste)    
        
    if forecastHorizon>1: 
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions6, color='red', label='Previsão')
        #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
        plt.legend(loc='best')
        plt.ylabel('KW/h')
        plt.xticks(rotation=30)
        plt.title('Previsão Consumo de Energia Elétrica (LASSO LARS)')
        plt.grid()
        plt.savefig('modelBR6_'+str(forecastHorizon)+'lag.png')   
        pyplot.show() 

    
    
    
    
    
    
    #3) Lasso Lars com Cross Validation (2 menor EQM)
    #Ta demorando muito. Verificar onde é e retirar. (a partir do 50)
    
    from sklearn.linear_model import LassoLarsCV
    import time
    
    def lassoLarsCV_classificator(X_treino, X_teste, y_treino, y_teste):
        eps_LassoLarsCV_list = [1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2,1e-1, 1, 5]
        EQM3_lista=[]
        EQM10_lista=[]
        EQM50_lista=[]
        EQM100_lista=[]
        EQM500_lista=[]
    
        for eps in eps_LassoLarsCV_list:
                start_time = time.time()
                model= LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, 
                           normalize=True, cv=3, max_n_alphas=1000, eps=eps, copy_X=True, 
                           positive=False)      
                model_fit=model.fit(X_treino, y_treino)
                
                y_predictions = model_fit.predict(X_teste)
                EQM3_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
                print(sorted(EQM3_lista))
        print ("My program took", time.time() - start_time, "seconds to run")

        for eps in eps_LassoLarsCV_list:
                start_time = time.time()
                model= LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, 
                           normalize=True, cv=10, max_n_alphas=1000, eps=eps, copy_X=True, 
                           positive=False)      
                model_fit=model.fit(X_treino, y_treino)
                
                y_predictions = model_fit.predict(X_teste)
                EQM10_lista.append(np.sqrt(mean_squared_error(y_teste, y_predictions)))
                print(sorted(EQM10_lista))
        print ("My program took", time.time() - start_time, "seconds to run")

                
        for eps in eps_LassoLarsCV_list:
                start_time = time.time()
                model= LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, 
                           normalize=True, cv=50, max_n_alphas=1000, eps=eps, copy_X=True, 
                           positive=False)     
                model_fit=model.fit(X_treino, y_treino)
                
                y_predictions = model_fit.predict(X_teste)
                print(sorted(EQM50_lista))
        print ("My program took", time.time() - start_time, "seconds to run")                
                
        for eps in eps_LassoLarsCV_list:
                start_time = time.time()
                model= LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, 
                           normalize=True, cv=100, max_n_alphas=1000, eps=eps, copy_X=True, 
                           positive=False)     
                model_fit=model.fit(X_treino, y_treino)
                
                y_predictions = model_fit.predict(X_teste)
                print(sorted(EQM100_lista))
        print ("My program took", time.time() - start_time, "seconds to run")  
              
        for eps in eps_LassoLarsCV_list:
                start_time = time.time()
                model= LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, 
                           normalize=True, cv=500, max_n_alphas=1000, eps=eps, copy_X=True, 
                           positive=False)    
                model_fit=model.fit(X_treino, y_treino)
                
                y_predictions = model_fit.predict(X_teste)
                print(sorted(EQM500_lista))
        print ("My program took", time.time() - start_time, "seconds to run")
        
                
        EQM=pd.DataFrame(columns=['EQM3', 'EQM10','EQM50', 'EQM100','EQM500'], index=eps_LassoLarsCV_list)
        EQM['EQM3']=EQM1_lista
        EQM['EQM10']=EQM10_lista
        EQM['EQM50']=EQM50_lista
        EQM['EQM100']=EQM100_lista
        EQM['EQM500']=EQM500_lista
        
        EQM1_sorted=sorted(EQM['EQM3'])        
        EQM10_sorted=sorted(EQM['EQM10'])        
        EQM50_sorted=sorted(EQM['EQM50'])        
        EQM100_sorted=sorted(EQM['EQM100'])        
        EQM500_sorted=sorted(EQM['EQM500'])
        
        menorEQM_list=[EQM3_sorted[0],EQM10_sorted[0],EQM50_sorted[0],
                  EQM100_sorted[0],EQM500_sorted[0]]        
        menorEQM=sorted(menorEQM_list)[0]
        
        position=menorEQM_list.index(menorEQM)
        menorEQM_col=EQM.columns[position]

        nZeroCoef=float(EQM.columns[position][3:])
        eps=EQM.loc[EQM[menorEQM_col].isin([menorEQM])].index.item()
        print('O epsilon e o númerocv que apresentam o menor erro de previsão são iguais a %.1f' % eps,nZeroCoef)               

        return eps,cv
    
        model7= LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, 
                           normalize=True, cv=10, max_n_alphas=1000, eps=eps, 
                           copy_X=True, positive=False)    
        
        model7_fit = LassoLarsCV(cv=50).fit(X_treino,y_treino)
        print(model7_fit.coef_)    
        coef7=model7_fit.coef_
        
        R27 = model7_fit.score(X_treino, y_treino) 
        
        y_predictions7 = model7_fit.predict(X_teste)
        y_predictions7= pd.DataFrame(y_predictions7, index=teste)   #previsão
    
        EQM7 = mean_squared_error(y_teste, y_predictions7)
        resid7 = np.sqrt(EQM7)
        print('Test MSE: %.3f' % EQM7,resid7)
        
        accuracy_7 = r2_score(y_teste, y_predictions7)
        R2_7_teste = model7.score(X_teste, y_teste)  
        print ('accuracy, R2_teste: %.3f' % accuracy_7, R2_7_teste)
               
        if forecastHorizon>1: 
            plt.figure()    
            pyplot.plot(y_treino, label='Treino')
            pyplot.plot(y_teste, color='black', label='Teste')
            pyplot.plot(y_predictions7, color='red', label='Previsão')
            #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
            plt.legend(loc='best')
            plt.ylabel('KW/h')
            plt.xticks(rotation=30)
            plt.title('Previsão Consumo de Energia Elétrica (LASSO LARS CV)')
            plt.grid()
            plt.savefig('modelBR7_'+str(forecastHorizon)+'lag.png')   
            pyplot.show() 
    
        
        
     
                                #Ridge Regression 
                                
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
                                     
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
#            print(EQM_lista)
            
        EQM=pd.DataFrame(columns=['alpha', 'resíduo'], index=list(range(len(alpha_ridge_lista))))
        EQM['alpha']=alpha_ridge_lista
        EQM['resíduo']=EQM_lista
        print(EQM.sort_values(by='resíduo', ascending=True)[0:5])
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
    y_predictions8= pd.DataFrame(y_predictions8, index=teste)   #previsão

    EQM8 = mean_squared_error(y_teste, y_predictions8)
    resid8 = np.sqrt(EQM8)
    print('Test MSE: %.3f' % EQM8,resid8)
        
    accuracy_8 = r2_score(y_teste, y_predictions8)
    R2_8_teste = model8_fit.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_8, R2_8_teste)    
        
    if forecastHorizon>1: 
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions8, color='red', label='Previsão')
        #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
        plt.legend(loc='best')
        plt.ylabel('KW/h')
        plt.xticks(rotation=30)
        plt.title('Previsão Consumo de Energia Elétrica (Ridge)')
        plt.grid()
        plt.savefig('modelBR8_'+str(forecastHorizon)+'lag.png')   
        pyplot.show() 

    

    
                                #ElasticNet (4 MENOR EQM)
                                
    #1) ElasticNet sem CV dava o mesmo resultado que o com CV
    #Vai dar trabalho
    
    l1_ratio = [0,0.25,0.5,0.75,1]
    eps_list = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10,20]

    
    model=ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, normalize=False, 
                     precompute=False, max_iter=1000, copy_X=True, tol=0.0001, 
                     warm_start=False, positive=False, random_state=None)
   
    from sklearn.linear_model import ElasticNet
   
    model90 = ElasticNet().fit(X_treino,y_treino)
    print(model90.coef_) 

    R290 = model90.score(X_treino, y_treino) 
    coef90=model90.coef_
    
    y_predictions90 = model90.predict(X_teste)
    y_predictions90= pd.DataFrame(y_predictions90, index=teste)   #previsão

    EQM90 = mean_squared_error(y_teste, y_predictions90)
    resid90 = np.sqrt(EQM90)
    print('Test MSE: %.3f' % EQM90,resid90)
    
    accuracy_90 = r2_score(y_teste, y_predictions90)
    R2_90_teste = model90.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_90, R2_90_teste)
        
    if forecastHorizon>1: 
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions90, color='red', label='Previsão')
        #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
        plt.legend(loc='best')
        plt.ylabel('KW/h')
        plt.xticks(rotation=30)
        plt.title('Previsão Consumo de Energia Elétrica (ElasticNet)')
        plt.grid()
        plt.savefig('modelBR9_'+str(forecastHorizon)+'lag.png')  
        pyplot.show() 
    
#    
#    #2) ElasticNetCV
#    
#    from sklearn.linear_model import ElasticNetCV       
#
#    model9 = ElasticNetCV(alphas=None, copy_X=True, cv=100, eps=0.001, fit_intercept=True,
#                          l1_ratio=0.5, max_iter=1000, n_alphas=100, n_jobs=None,
#                          normalize=False, positive=False, precompute='auto', random_state=0,
#                          selection='cyclic', tol=0.0001, verbose=0).fit(X_treino,y_treino)
#    print(model9.coef_) 
#
#    R29 = model9.score(X_treino, y_treino) 
#    coef9=model9.coef_
#    
#    y_predictions9 = model9.predict(X_teste)
#    y_predictions9= pd.DataFrame(y_predictions9, index=teste)   #previsão
#
#    EQM9 = mean_squared_error(y_teste, y_predictions9)
#    resid9 = np.sqrt(EQM9)
#    print('Test MSE: %.3f' % EQM9,resid9)
#    
#    accuracy_9 = r2_score(y_teste, y_predictions9)
#    R2_9_teste = model9.score(X_teste, y_teste)  
#    print ('accuracy, R2_teste: %.3f' % accuracy_9, R2_9_teste)
#        
#    plt.figure()    
#    pyplot.plot(y_treino, label='Treino')
#    pyplot.plot(y_teste, color='black', label='Teste')
#    pyplot.plot(y_predictions9, color='red', label='Previsão')
#    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
#    plt.legend(loc='best')
#    plt.ylabel('KW/h')
#    plt.xticks(rotation=30)
#    plt.title('Previsão Consumo de Energia Elétrica (ElasticNetCV)')
#    #plt.grid()
#    plt.savefig('modelBR9.png')   
#    pyplot.show() 

    
   #Random Forest  (MELHOR EQM)
   #Vai dar trabalho
   
    from sklearn.ensemble import RandomForestRegressor

    {'bootstrap': True,
 'criterion': 'mse',
 'max_depth': None,
 'max_features': 'auto',
 'max_leaf_nodes': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 10,
 'n_jobs': 1,
 'oob_score': False,
 'random_state': 42,
 'verbose': 0,
 'warm_start': False}    
    
    model=RandomForestRegressor(n_estimators=1000, max_depth=None, min_samples_split=2, 
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                min_impurity_split=None, bootstrap=True, oob_score=False, 
                                n_jobs=1, random_state=None, verbose=0, warm_start=False)
        
    
#    model10 = RandomForestRegressor(n_estimators = 1000, random_state = 0)
    
    model10_fit=model.fit(X_treino, y_treino)
    
    print(model10_fit.feature_importances_)
    coef10=model10_fit.feature_importances_
    
    R210 = model10_fit.score(X_treino, y_treino) 

    y_predictions10 = model10_fit.predict(X_teste)
    y_predictions10= pd.DataFrame(y_predictions10, index=teste)   #previsão

    EQM10 = mean_squared_error(y_teste, y_predictions10)
    resid10 = np.sqrt(EQM10)
    print('Test MSE: %.3f' % EQM10,resid10)
        
    accuracy_10 = r2_score(y_teste, y_predictions10)
    R2_10_teste = model10_fit.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_10, R2_10_teste)    
        
    if forecastHorizon>1:     
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions10, color='red', label='Previsão')
        #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
        plt.legend(loc='best')
        plt.ylabel('KW/h')
        plt.xticks(rotation=30)
        plt.title('Previsão Consumo de Energia Elétrica (Random Forest)')
        plt.grid()
        plt.savefig('modelBR10_'+str(forecastHorizon)+'lag.png')   
        pyplot.show() 
#    
    

  
    
    
    
    colunas2 =  ['DM','DS','MÊS','ANO','ESTAC','FER','NEB_BR_9','NEB_BR_15','NEB_BR_21',
                'NEB_BR_MED','PA_BR_9','PA_BR_15','PA_BR_21','PA_BR_MED','TEMP_BS_BR_9','TEMP_BS_BR_15','TEMP_BS_BR_21','TEMP_BS_BR_MED','TEMP_BU_BR_9','TEMP_BU_BR_15','TEMP_BU_BR_21','TEMP_BU_BR_MED','UMID_BR_9', 
                'UMID_BR_15','UMID_BR_21','UMID_BR_MED','DV_BR_9','DV_BR_15','DV_BR_21','DV_BR_MED','VV_BR_9','VV_BR_15','VV_BR_21','VV_BR_MED','TAR_BR_CSO','TAR_BR_CP','TAR_BR_IP','TAR_BR_IND','TAR_BR_PP','TAR_BR_RED',
                'TAR_BR_RR','TAR_BR_RRA','TAR_BR_RRI','TAR_BR_SP1','TAR_BR_SP2','TAR_BR_MED','Meta_Selic', 'Taxa_Selic','CDI','DolarC','DolarC_var','DolarV','DolarV_var','EuroC','EuroC_var','EuroV','EuroV_var','IBV_Cot',
                'IBV_min','IBV_max','IBV_varabs','IBV_varperc','IBV_vol','INPC_m','INPC_ac','IPCA_m','IPCA_ac','IPAM_m','IPAM_ac','IPADI_m', 'IPADI_ac' , 'IGPM_m','IGPM_ac','IGPDI_m','IGPDI_ac','PAB_o','PAB_d',
                'TVP_o','TVP_d','PICV_o','ICV_d','CCU_o','CCU_d','CS_o','CS_d','UCPIIT_FGV_o','UCPIIT_FGV_d','CPCIIT_CNI_o','CPCIIT_CNI_d','VIR_o','VIR_d','HTPIT_o','HTPIT_d','SRIT_o','SRIT_d','PPOB','PGN','PIG_o','PIG_d','PIBCa_o','PIBCa_d',
                'PIBI_o','PIBI_d','PIBCo_o','PIBCo_d','PIA_o','PIA_d','ICC','INEC','ICEI','DBNDES','IEG_o','IEG_d','IETIT_o','IETIT_d','IETC_o','IETC_d','IETS_o','IETS_d','IETCV_o','IETCV_d', 'PO','TD','BM','PME',
                'TEMPaj_BS_BR_9','TEMPaj_BS_BR_15','TEMPaj_BS_BR_21','TEMPaj_BS_BR_MED','TEMPaj_BU_N_MED','TEMPaj_BU_BR_9','TEMPaj_BU_BR_15','TEMPaj_BU_BR_21','TEMPaj_BU_BR_MED'] 


    coef = pd.DataFrame(coef2, index=X.columns)
    coef.columns = ['OLS']
    coef['LinearRegression']=coefreg
    coef['Lasso']=coef3
#    coef['LassoCV']=coef4
    coef['Lars']=coef5
    coef['LassoLars']=coef6
    coef['LassoLarsCV']=coef7
    coef['Ridge']=coef8
    coef['ElasticNet']=coef90  
#    coef['ElasticNetCV']=coef9
    coef['RandomForest']=coef10
    
  
#    R210=EQM10=resid10=accurcy_10=R2_10_teste=0

    
    R2_list      = [R21,R22,R2reg, R23,R25,R26,R27,R28,R290,R210] 
  
    EQM_list     = [EQM1,EQM2,EQMreg,EQM3,EQM5,EQM6,EQM7,EQM8,EQM90,EQM10]
   
    resid_list    = [resid1,resid2,residreg, resid3,resid5,resid6,resid7,
                    resid8,resid90,resid10]
   
    accuracy_list = [accuracy_1,accuracy_2,accuracy_reg,accuracy_3,accuracy_5,
                    accuracy_6,accuracy_7,accuracy_8,accuracy_90,accuracy_10]
    
    R2_test_list = [R2_1_teste,R2_2_teste,R2_reg_teste, R2_3_teste,R2_5_teste,
                    R2_6_teste,R2_7_teste,R2_8_teste,R2_90_teste,R2_10_teste]
        
    
    index=['R2', 'EQM', 'Resíduo','Accuracy','R2 teste']
    
    colunas3 = ['AR','OLS','LinearRegression','Lasso','Lars','LassoLars',
                'LassoLarsCV','Ridge','ElasticNet','RandomForest']
    

    
    previsao = pd.DataFrame([R2_list, EQM_list, resid_list, accuracy_list,
                             R2_test_list],index=index, columns=colunas3)
    previsao=previsao.transpose()
    print(previsao)

    if forecastHorizon==1:
        y_verd=y_teste.values[0]
        comparacao = pd.DataFrame(index=colunas3,)
        comparacao['Preço Verdadeiro']=y_verd  
        comparacao['Preço Estimado']=0
        comparacao['Preço Estimado'][:1]=y_predictions1+y_treino[-1:].values[0]
        comparacao['Preço Estimado'][1:2]=y_predictions2        
        comparacao['Preço Estimado'][2:3]=y_predictionsreg
        comparacao['Preço Estimado'][3:4]=y_predictions3
#       comparacao['Preço Estimado'][3:4]=y_predictions4
        comparacao['Preço Estimado'][4:5]=y_predictions5
        comparacao['Preço Estimado'][5:6]=y_predictions6
        comparacao['Preço Estimado'][6:7]=y_predictions7
        comparacao['Preço Estimado'][7:8]=y_predictions8
#       comparacao['Preço Estimado'][9:9]=y_predictions9
        comparacao['Preço Estimado'][8:9]=y_predictions90
        comparacao['Preço Estimado'][9:10]=y_predictions10 
        comparacao['Diferenca']= comparacao['Preço Verdadeiro']- comparacao['Preço Estimado']    
        print(comparacao)

    #Caso deseje expotar os dados para o excel:

#   writer = pd.ExcelWriter('dados_lag'+str(forecastHorizon)+'.xlsx')
#   coef.to_excel(writer,'coeficientes_lag'+str(forecastHorizon))
#   previsao.to_excel(writer,'previsao_lag'+str(forecastHorizon))
#   if forecastHorizon==1:
#       comparacao.to_excel(writer,'comparacao_lag'+str(forecastHorizon))
#   writer.save()


    #Plot4
    print(previsao.sort_values(by='Resíduo', ascending=True)[0:5]['Resíduo']) 

    
    if forecastHorizon>1:    
        fig2, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 7))
        axes[0].set_title("Previsão CEE (Random Forest) - Erro = 48017 KW/h")
        axes[0].plot (y_treino, label='Treino')
        axes[0].plot(y_teste, color='black', label='Teste')
        axes[0].plot(y_predictions6, label='Previsão')
        axes[0].legend(loc='best')
        axes[0].set_ylabel("KW/h")
        
        axes[1].set_title("Previsão CEE (Lasso Lars) - Erro = 71596 KW/h")
        axes[1].plot (y_treino, label='Treino')
        axes[1].plot(y_teste, color='black', label='Teste')
        axes[1].plot(y_predictions8, label='Previsão')
        axes[1].legend(loc='best')
        axes[1].set_ylabel("KW/h")
        
        axes[2].set_title("Previsão CEE (Ridge) - Erro = 77504 KW/h")
        axes[2].plot (y_treino, label='Treino')
        axes[2].plot(y_teste, color='black', label='Teste')
        axes[2].plot(y_predictions10, label='Previsão')
        axes[2].legend(loc='best')
        axes[2].set_ylabel("KW/h")
        
        axes[3].set_title("Previsão CEE (Lasso) - Erro = 140756 KW/h")
        axes[3].plot (y_treino, label='Treino')
        axes[3].plot(y_teste, color='black', label='Teste')
        axes[3].plot(y_predictions3, label='Previsão')
        axes[3].legend(loc='best')
        axes[3].set_ylabel("KW/h")
        fig2.tight_layout()
        plt.savefig('modelo'+str(forecastHorizon)+'dias.png')   
#
    


INCLUIR XBOOSTING
INCLUIR RANDOM WALK
DEFINIR OS LAGS OTIMOS
VERIFICAR OS LAMBDAS (LASSO, RANDOM FOREST)




