r u#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 17:17:47 2018

@author: pedrocampelo
"""


 
    import os

    cwd = os.getcwd()
    cwd
    
    os.chdir('/Users/pedrocampelo/Downloads')

    import pandas as pd
    import datetime as dt
    
    periodo = pd.date_range('2/1/2017', periods=546)
    treino = pd.date_range('2/1/2017', periods=365)
    teste = pd.date_range('2/1/2018', periods=181)


    from pandas import Series
    from pandas import read_csv

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




    label = read_csv('label.csv', sep=';',header=0, dtype={'label':str})
    label2 = pd.DataFrame(label, colunas)

    #agregando os dados de consumo de EE com as demais variáveis
    
    import numpy as np  
    
    agregado = np.hstack((consumo, variaveis))
    dados = pd.DataFrame(agregado, columns = colunas , index=periodo)
    
    y=dados['CEE_BR_TOT']

        
    del consumo, variaveis, agregado

    
    #Dividindo em teste e treino
    train_size = int(len(dados) * (365/546))
    dados_treino, dados_teste = dados[0:train_size], dados[train_size:len(dados)]

    
    print('Observations: %d' % (len(dados)))
    print('Training Observations: %d' % (len(dados_treino)))
    print('Testing Observations: %d' % (len(dados_teste)))
    
     
        #Tirando a média, max, min, var..
     print(dados['CEE_BR_TOT'].mean(), dados['CEE_SECO_TOT'].mean(),dados['CEE_SUL_TOT'].mean(),
           dados['CEE_NE_TOT'].mean(), dados['CEE_N_TOT'].mean())

     print(dados['CEE_BR_TOT'].min(), dados['CEE_SECO_TOT'].min(),dados['CEE_SUL_TOT'].min(),
           dados['CEE_NE_TOT'].min(), dados['CEE_N_TOT'].min())        

     print(dados['CEE_BR_TOT'].max(), dados['CEE_SECO_TOT'].max(),dados['CEE_SUL_TOT'].max(),
           dados['CEE_NE_TOT'].max(), dados['CEE_N_TOT'].max())
     
     print(dados['CEE_BR_TOT'].median(), dados['CEE_SECO_TOT'].median(),dados['CEE_SUL_TOT'].median(),
           dados['CEE_NE_TOT'].median(), dados['CEE_N_TOT'].median())

     print(dados['CEE_BR_TOT'].std(), dados['CEE_SECO_TOT'].std(),dados['CEE_SUL_TOT'].std(),
           dados['CEE_NE_TOT'].std(), dados['CEE_N_TOT'].std())


    
    #Plots
    
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    
    
    #Decomposicao TS
    
    from statsmodels.tsa.seasonal import seasonal_decompose
   
    dec_seas = seasonal_decompose(y, model='multiplicative')
    fig = dec_seas.plot()
    plt.savefig('CEEseas.png')    

    
   
    #Agregado
    plt.figure()
    dados['CEE_BR_TOT'].plot(label='Brasil')
    dados['CEE_SECO_TOT'].plot(label='Sud/CO')
    dados['CEE_SUL_TOT'].plot(label='Sul')
    dados['CEE_NE_TOT'].plot(label='Nordeste')
    dados['CEE_N_TOT'].plot(label='Norte')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.title('Consumo de Energia Elétrica (Agregado)')
    plt.grid()
    plt.savefig('CEEagregado.png')    
    plt.show()

    
    
    dados['CEE_BR_TOT'].describe()
    dados['CEE_SECO_TOT'].describe()
    dados['CEE_SUL_TOT'].describe()
    dados['CEE_NE_TOT'].describe()
    dados['CEE_N_TOT'].describe()
    
    
   #MÉDIAS 
    plt.figure()
    dados['CEE_BR_MED'].plot(label='Brasil')
    dados['CEE_SECO_MED'].plot(label='Sud/CO')
    dados['CEE_SUL_MED'].plot(label='Sul')
    dados['CEE_NE_MED'].plot(label='Nordeste')
    dados['CEE_N_MED'].plot(label='Norte')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.title('Consumo de Energia Elétrica (Médio)')
    plt.grid()
    plt.savefig('CEEmedia.png')   
    plt.show()


    
    dados['CEE_BR_TOT'].describe()
    dados['CEE_SECO_TOT'].describe()
    dados['CEE_SUL_TOT'].describe()
    dados['CEE_NE_TOT'].describe()
    dados['CEE_N_TOT'].describe()
    
    
    #Horários
        #Brasil
    plt.figure()
    dados['CEE_BR_24'].plot(color='pink', label='Hora 00')
    dados['CEE_BR_09'].plot(color='red', label='Hora 09')
    dados['CEE_BR_12'].plot(color='orange', label='Hora 12')
    dados['CEE_BR_15'].plot(color='green', label='Hora 15')
    dados['CEE_BR_18'].plot(color='red', label='Hora 18')
    dados['CEE_BR_21'].plot(color='red', label='Hora 21')
    dados['CEE_BR_MED'].plot(color='purple', label='Hora Med')
    plt.ylim(0,100000)
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.title('Consumo de Energia Elétrica do Brasil por Hora')
    #plt.grid()
    plt.savefig('CEEBRhora.png')   
    plt.show()
    
    #Agregado
    
        
    
    fig2, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 7))
    axes[0].set_title("Sudeste/Centro-Oeste")
    axes[0].plot(periodo, dados['CEE_SECO_09'], label="09h")
    axes[0].plot(periodo, dados['CEE_SECO_15'], label="15h")
    axes[0].plot(periodo, dados['CEE_SECO_21'], label="18h")
    axes[0].plot(periodo, dados['CEE_SECO_MED'], label="Média")
    axes[0].legend(loc='best')
    axes[0].set_ylabel("KW/h")
    
    axes[1].set_title("Sul")
    axes[1].plot(periodo, dados['CEE_SUL_09'], label="9h")
    axes[1].plot(periodo, dados['CEE_SUL_15'], label="15h")    
    axes[1].plot(periodo, dados['CEE_SUL_21'], label="21h")
    axes[1].plot(periodo, dados['CEE_SUL_MED'], label="Média")
    axes[1].legend(loc='best')    
    axes[1].set_ylabel("KW/h")
    
    axes[2].set_title("Nordeste")
    axes[2].plot(periodo, dados['CEE_NE_09'], label="9h")
    axes[2].plot(periodo, dados['CEE_NE_15'], label="15h")
    axes[2].plot(periodo, dados['CEE_NE_21'], label="21h")
    axes[2].plot(periodo, dados['CEE_NE_MED'], label="Média")
    axes[2].legend(loc='best')
    axes[2].set_ylabel("KW/h")
    
    axes[3].set_title("Norte")
    axes[3].plot(periodo, dados['CEE_N_09'], label="9h")
    axes[3].plot(periodo, dados['CEE_N_15'], label="15h")
    axes[3].plot(periodo, dados['CEE_N_21'], label="21h")
    axes[3].plot(periodo, dados['CEE_N_MED'], label="Média")
    axes[3].legend(loc='best')
    axes[3].set_ylabel("KW/h")

    fig2.tight_layout()
    plt.savefig('CEEhorraagreg.png')   

    plt.show()

    
 """
    Por região   
    
    
        #SE/CO
    plt.figure()
    dados['CEE_SECO_24'].plot( label='Hora 00')
    dados['CEE_SECO_06'].plot(label='Hora 06')
    dados['CEE_SECO_09'].plot(label='Hora 09')
    dados['CEE_SECO_12'].plot(label='Hora 12')
    dados['CEE_SECO_15'].plot( label='Hora 15')
    dados['CEE_SECO_18'].plot( label='Hora 18')
    dados['CEE_SECO_MED'].plot(label='Hora Med')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.title('Consumo de Energia Elétrica do SE/CO por Hora')
    #plt.grid()
    plt.savefig('CEESECOhora.png')   
    plt.show()
   
    
        #SUL
    plt.figure()
    dados['CEE_SUL_24'].plot(label='Hora 00')
    dados['CEE_SUL_06'].plot(label='Hora 06')
    dados['CEE_SUL_09'].plot(label='Hora 09')
    dados['CEE_SUL_12'].plot(label='Hora 12')
    dados['CEE_SUL_15'].plot( label='Hora 15')
    dados['CEE_SUL_18'].plot(label='Hora 18')
    dados['CEE_SUL_MED'].plot(label='Hora Med')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.title('Consumo de Energia Elétrica do Sul por Hora')
    #plt.grid()
    plt.savefig('CEESULhora.png')   
    plt.show()
   
    
        #NE
    plt.figure()
    dados['CEE_NE_24'].plot(label='Hora 00')
    dados['CEE_NE_06'].plot(label='Hora 06')
    dados['CEE_NE_09'].plot(label='Hora 09')
    dados['CEE_NE_12'].plot(label='Hora 12')
    dados['CEE_NE_15'].plot(label='Hora 15')
    dados['CEE_NE_18'].plot(label='Hora 18')
    dados['CEE_NE_MED'].plot(label='Hora Med')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.title('Consumo de Energia Elétrica do Nordeste por Hora')
    #plt.grid()
    plt.savefig('CEENEhora.png')   
    plt.show()

    
    
        #N - Consumo de EE caiu muito em 21/03 horas 17,18,19
    plt.figure()
    dados['CEE_N_24'].plot(label='Hora 00')
    dados['CEE_N_06'].plot(label='Hora 06')
    dados['CEE_N_09'].plot(label='Hora 09')
    dados['CEE_N_12'].plot(label='Hora 12')
    dados['CEE_N_15'].plot(label='Hora 15')
    dados['CEE_N_18'].plot(label='Hora 18')
    dados['CEE_N_MED'].plot(label='Hora Med')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.title('Consumo de Energia Elétrica do Norte por Hora')
    #plt.grid()
    plt.savefig('CEENRhora.png')   
    plt.show()

 """   
   
    #Por setores
    
        #Brasil
    plt.figure()
    dados['CEE_BR_CSO'].plot(label='Comercial, Serviços e Outras')
    dados['CEE_BR_CP'].plot(label='Consumo Próprio')
    dados['CEE_BR_IP'].plot(label='Iluminação Pública')
    dados['CEE_BR_IND'].plot(label='Industrial')
    dados['CEE_BR_PP'].plot(label='Poder Público')
    dados['CEE_BR_RED'].plot(label='Residencial')
    #dados['CEE_BR_RRA'].plot(label='Rural Agricultor')
    #dados['CEE_BR_RRI'].plot(label='Rural Irrigante')
    dados['CEE_BR_SP1'].plot(label='Serv. Púb. (água, esgoto)')
    #dados['CEE_BR_SP2'].plot(label='Serv. Púb. (tração elétrica)')
    #dados['CEE_BR_MED'].plot(color='black', label='Med')
    #dados['CEE_BR_TOT'].plot(color='black', label='Total')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.title('Consumo de Energia Elétrica do Brasil por Setor')
    #plt.grid()
    plt.ylim(0, 1380000)
    plt.savefig('CEEBRsetor.png')   
    plt.show()

    #Agregado    
    
    fig2, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 7))
    axes[0].set_title("Sudeste/Centro-Oeste")
    axes[0].plot(periodo, dados['CEE_SECO_RED'], label="Residencial")
    axes[0].plot(periodo, dados['CEE_SECO_CSO'], label="Comercial, Serviços e Outras")
    axes[0].plot(periodo, dados['CEE_SECO_IND'], label="Industrial")
    axes[0].plot(periodo, dados['CEE_SECO_PP'], label="Poder Público")
    axes[0].legend(loc='best')
    axes[0].set_ylabel("KW/h")
    
    axes[1].set_title("Sul")
    axes[1].plot(periodo, dados['CEE_SUL_RED'], label="Residencial")
    axes[1].plot(periodo, dados['CEE_SUL_CSO'], label="Comercial, Serviços e Outras")
    axes[1].plot(periodo, dados['CEE_SUL_IND'], label="Industrial")
    axes[1].plot(periodo, dados['CEE_SUL_PP'], label="Poder Público")
    axes[1].legend(loc='best')    
    axes[1].set_ylabel("KW/h")
    
    axes[2].set_title("Nordeste")
    axes[2].plot(periodo, dados['CEE_NE_RED'], label="Residencial")
    axes[2].plot(periodo, dados['CEE_NE_CSO'], label="Comercial, Serviços e Outras")
    axes[2].plot(periodo, dados['CEE_NE_IND'], label="Industrial")
    axes[2].plot(periodo, dados['CEE_NE_PP'], label="Rural Agricultor")
    axes[2].legend(loc='best')    
    axes[2].set_ylabel("KW/h")
    
    axes[3].set_title("Norte")
    axes[3].plot(periodo, dados['CEE_N_RED'], label="Residencial")
    axes[3].plot(periodo, dados['CEE_N_CSO'], label="Comercial, Serviços e Outras")
    axes[3].plot(periodo, dados['CEE_N_IND'], label="Industrial")
    axes[3].plot(periodo, dados['CEE_N_PP'], label="Poder Público")
    axes[3].legend(loc='best')    
    axes[3].set_ylabel("KW/h")

    fig2.tight_layout()
    plt.savefig('CEEsetoragreg.png')   

    plt.show()

    
 """   
        #SE/CO
    plt.figure()
    dados['CEE_SECO_CSO'].plot(label='Comercial, Serviços e Outras')
    dados['CEE_SECO_CP'].plot(label='Consumo Próprio')
    dados['CEE_SECO_IP'].plot(label='Iluminação Pública')
    dados['CEE_SECO_IND'].plot(label='Industrial')
    dados['CEE_SECO_PP'].plot(label='Poder Público')
    dados['CEE_SECO_RED'].plot(label='Residencial')
    dados['CEE_SECO_RRA'].plot(label='Rural Agricultor')
    dados['CEE_SECO_RRI'].plot(label='Rural Irrigante')
    dados['CEE_SECO_SP1'].plot( label='Serv. Púb. (água, esgoto)')
    dados['CEE_SECO_SP2'].plot(label='Serv. Púb. (tração elétrica)')
    #dados['CEE_SECO_MED'].plot(color='black', label='Med')
    #dados['CEE_SECO_TOT'].plot(color='black', label='Total')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.title('Consumo de Energia Elétrica do SE/CO por Setor')
    #plt.grid()
    plt.savefig('CEESECOSetor.png')   
    plt.show()


    
    
        #SUL
    plt.figure()
    dados['CEE_SUL_CSO'].plot(label='Comercial, Serviços e Outras')
    dados['CEE_SUL_CP'].plot(label='Consumo Próprio')
    dados['CEE_SUL_IP'].plot( label='Iluminação Pública')
    dados['CEE_SUL_IND'].plot( label='Industrial')
    dados['CEE_SUL_PP'].plot(label='Poder Público')
    dados['CEE_SUL_RED'].plot(label='Residencial')
    dados['CEE_SUL_RRA'].plot( label='Rural Agricultor')
    dados['CEE_SUL_RRI'].plot(label='Rural Irrigante')
    dados['CEE_SUL_SP1'].plot(label='Serv. Púb. (água, esgoto)')
    dados['CEE_SUL_SP2'].plot(label='Serv. Púb. (tração elétrica)')
    #dados['CEE_SUL_MED'].plot(color='black', label='Med')
    #dados['CEE_SUL_TOT'].plot(color='black', label='Total')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.title('Consumo de Energia Elétrica do Sul por Setor')
    #plt.grid()
    plt.savefig('CEESULSetor.png')   
    plt.show()

    
    
        #NE
    plt.figure()
    dados['CEE_NE_CSO'].plot(label='Comercial, Serviços e Outras')
    dados['CEE_NE_CP'].plot(label='Consumo Próprio')
    dados['CEE_NE_IP'].plot(label='Iluminação Pública')
    dados['CEE_NE_IND'].plot(label='Industrial')
    dados['CEE_NE_PP'].plot( label='Poder Público')
    dados['CEE_NE_RED'].plot(label='Residencial')
    dados['CEE_NE_RRA'].plot(label='Rural Agricultor')
    dados['CEE_NE_RRI'].plot(label='Rural Irrigante')
    dados['CEE_NE_SP1'].plot( label='Serv. Púb. (água, esgoto)')
    dados['CEE_NE_SP2'].plot(label='Serv. Púb. (tração elétrica)')
    #dados['CEE_NE_MED'].plot(color='black', label='Med')
    #dados['CEE_NE_TOT'].plot(color='black', label='Total')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.title('Consumo de Energia Elétrica do Nordeste por Setor')
    #plt.grid()
    plt.savefig('CEENESetor.png')   
    plt.show()  

    
    
        #N - Consumo de EE caiu muito em 21/03 horas 17,18,19
    plt.figure()
    dados['CEE_N_CSO'].plot(label='Comercial, Serviços e Outras')
    dados['CEE_N_CP'].plot(label='Consumo Próprio')
    dados['CEE_N_IP'].plot(label='Iluminação Pública')
    dados['CEE_N_IND'].plot(label='Industrial')
    dados['CEE_N_PP'].plot(label='Poder Público')
    dados['CEE_N_RED'].plot(label='Residencial')
    dados['CEE_N_RRA'].plot(label='Rural Agricultor')
    dados['CEE_N_RRI'].plot(label='Rural Irrigante')
    dados['CEE_N_SP1'].plot( label='Serv. Púb. (água, esgoto)')
    dados['CEE_N_SP2'].plot(label='Serv. Púb. (tração elétrica)')
    #dados['CEE_N_MED'].plot(color='black', label='Med')
    #dados['CEE_N_TOT'].plot(color='black', label='Total')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.title('Consumo de Energia Elétrica do Norte por Setor')
    #plt.grid()
    plt.savefig('CEENSetor.png')   
    plt.show()    

    """
    
    
    #tarifas média
    plt.figure()
    dados['TAR_BR_MED'].plot(label='Brasil')
    dados['TAR_SECO_MED'].plot( label='Sud/CO')
    dados['TAR_SUL_MED'].plot(label='Sul')
    dados['TAR_NE_MED'].plot( label='Nordeste')
    dados['TAR_N_MED'].plot(label='Norte')
    plt.legend(loc='best')
    plt.ylabel('R$')
    plt.title('Tarifas de Energia Elétrica')
    #plt.grid()
    plt.savefig('tarifasmed.png')   
    plt.show()
   
    dados['TAR_BR_MED'].describe()
    dados['TAR_SECO_MED'].describe()
    dados['TAR_SUL_MED'].describe()
    dados['TAR_NE_MED'].describe()
    dados['TAR_N_MED'].describe()
    
       
    #Por setores
    
    #Brasil
    
    plt.figure()
    dados['TAR_BR_CSO'].plot(label='Comercial, Serviços e Outras')
    dados['TAR_BR_CP'].plot(label='Consumo Próprio')
    #dados['TAR_BR_IP'].plot(label='Iluminação Pública')
    dados['TAR_BR_IND'].plot(label='Industrial')
    dados['TAR_BR_PP'].plot(label='Poder Público')
    dados['TAR_BR_RED'].plot(label='Residencial')
    #dados['TAR_BR_RRA'].plot(label='Rural Agricultor')
    #dados['TAR_BR_RRI'].plot(label='Rural Irrigante')
    dados['TAR_BR_SP1'].plot(label='Serv. Púb. (água, esgoto)')
    #dados['TAR_BR_SP2'].plot(label='Serv. Púb. (tração elétrica)')
    dados['TAR_BR_MED'].plot(color='black',label='Média')
    plt.ylim(300, 720)
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.title('Tarifa de Energia Elétrica do Brasil por Setor')
    #plt.grid()
    plt.savefig('tarBRsetor.png')   
    plt.show()
    

    #TARagregado
       
    fig2, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 7))
    axes[0].set_title("Sudeste/Centro-Oeste")
    axes[0].plot(periodo, dados['TAR_SECO_RED'], label="Residencial")
    axes[0].plot(periodo, dados['TAR_SECO_CSO'], label="Comercial, Serviços e Outras")
    axes[0].plot(periodo, dados['TAR_SECO_IND'], label="Industrial")
    axes[0].plot(periodo, dados['TAR_SECO_PP'], label="Poder Público")
    axes[0].legend(loc='best')
    axes[0].set_ylabel("KW/h")
    
    axes[1].set_title("Sul")
    axes[1].plot(periodo, dados['TAR_SUL_RED'], label="Residencial")
    axes[1].plot(periodo, dados['TAR_SUL_CSO'], label="Comercial, Serviços e Outras")
    axes[1].plot(periodo, dados['TAR_SUL_IND'], label="Industrial")
    axes[1].plot(periodo, dados['TAR_SUL_PP'], label="Poder Público")
    axes[1].legend(loc='best')    
    axes[1].set_ylabel("KW/h")
    
    axes[2].set_title("Nordeste")
    axes[2].plot(periodo, dados['TAR_NE_RED'], label="Residencial")
    axes[2].plot(periodo, dados['TAR_NE_CSO'], label="Comercial, Serviços e Outras")
    axes[2].plot(periodo, dados['TAR_NE_IND'], label="Industrial")
    axes[2].plot(periodo, dados['TAR_NE_PP'], label="Rural Agricultor")
    axes[2].legend(loc='best')    
    axes[2].set_ylabel("KW/h")
    
    axes[3].set_title("Norte")
    axes[3].plot(periodo, dados['TAR_N_RED'], label="Residencial")
    axes[3].plot(periodo, dados['TAR_N_CSO'], label="Comercial, Serviços e Outras")
    axes[3].plot(periodo, dados['TAR_N_IND'], label="Industrial")
    axes[3].plot(periodo, dados['TAR_N_PP'], label="Poder Público")
    axes[3].legend(loc='best')    
    axes[3].set_ylabel("KW/h")

    fig2.tight_layout()
    plt.savefig('TARsetoragreg.png')   

    plt.show()
    
    
    
    
    
    
    
  
    #Tempratura
    plt.figure()
    dados['TEMP_BS_BR_MED'].plot(label='Brasil')
    dados['TEMP_BS_SECO_MED'].plot(label='Sud/CO')
    dados['TEMP_BS_SUL_MED'].plot( label='Sul')
    dados['TEMP_BS_NE_MED'].plot(label='Nordeste')
    dados['TEMP_BS_N_MED'].plot( label='Norte')
    plt.legend(loc='best')
    plt.ylabel('oC')
    plt.title('Temperatura  por Região')
    plt.grid()
    plt.savefig('tempmed.png')   
    plt.show()

    
    dados['TEMP_BS_BR_MED'].describe()
    dados['TEMP_BS_SECO_MED'].describe()
    dados['TEMP_BS_SUL_MED'].describe()
    dados['TEMP_BS_NE_MED'].describe()
    dados['TEMP_BS_N_MED'].describe()
    
    
    
    
    #Temperatura por Hora 
    #Brasil
    plt.figure()
    dados['TEMP_BS_BR_9'].plot(label='9h')
    dados['TEMP_BS_BR_15'].plot(label='15h')
    dados['TEMP_BS_BR_21'].plot( label='21h')
    dados['TEMP_BS_BR_MED'].plot(label='Média')
    plt.legend(loc='best')
    plt.ylabel('oC')
    plt.title('Temperatura do Brasil por Hora')
    plt.grid()
    plt.savefig('tempBRHora.png')   
    plt.show()
    
    
    
    
    fig2, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 7))
    axes[0].set_title("Sudeste/Centro-Oeste")
    axes[0].plot(periodo, dados['TEMP_BS_SECO_9'], label="9h")
    axes[0].plot(periodo, dados['TEMP_BS_SECO_15'], label="15h")
    axes[0].plot(periodo, dados['TEMP_BS_SECO_21'], label="21h")
    axes[0].plot(periodo, dados['TEMP_BS_SECO_MED'], label="Média")
    axes[0].legend(loc='best')
    axes[0].set_ylabel("oC")
    
    axes[1].set_title("Sul")
    axes[1].plot(periodo, dados['TEMP_BS_SUL_9'], label="9h")
    axes[1].plot(periodo, dados['TEMP_BS_SUL_15'], label="15h")
    axes[1].plot(periodo, dados['TEMP_BS_SUL_21'], label="21h")
    axes[1].plot(periodo, dados['TEMP_BS_SUL_MED'], label="Média")
    axes[1].legend(loc='best')    
    axes[1].set_ylabel("oC")
    
    axes[2].set_title("Nordeste")
    axes[2].plot(periodo, dados['TEMP_BS_NE_9'], label="9h")
    axes[2].plot(periodo, dados['TEMP_BS_NE_15'], label="15h")
    axes[2].plot(periodo, dados['TEMP_BS_NE_21'], label="21h")
    axes[2].plot(periodo, dados['TEMP_BS_NE_MED'], label="Média")
    axes[2].legend(loc='best')
    axes[2].set_ylabel("oC")
    
    axes[3].set_title("Norte")
    axes[3].plot(periodo, dados['TEMP_BS_N_9'], label="9h")
    axes[3].plot(periodo, dados['TEMP_BS_N_15'], label="15h")
    axes[3].plot(periodo, dados['TEMP_BS_N_21'], label="21h")
    axes[3].plot(periodo, dados['TEMP_BS_N_MED'], label="Média")
    axes[3].legend(loc='best')
    axes[3].set_ylabel("oC")

    fig2.tight_layout()
    plt.savefig('tempagreg.png')   

    plt.show()

    
 """  
    #REsto do Brasil por imagens separadas
    #SECO
    

    plt.figure()
    dados['TEMP_BS_BR_9'].plot(label='9h')
    dados['TEMP_BS_BR_15'].plot(label='15h')
    dados['TEMP_BS_BR_21'].plot(label='21h')
    dados['TEMP_BS_BR_MED'].plot( label='Média')
    plt.legend(loc='best')
    plt.ylabel('oC')
    plt.title('Temperatura do SE/CO por Hora')
    plt.grid()
    plt.savefig('tempSECOHora.png')   
    plt.show()
    
    #Sul
    
    plt.figure()
    dados['TEMP_BS_SUL_9'].plot(label='9h')
    dados['TEMP_BS_SUL_15'].plot( label='15h')
    dados['TEMP_BS_SUL_21'].plot( label='21h')
    dados['TEMP_BS_SUL_MED'].plot(label='Média')
    plt.legend(loc='best')
    plt.ylabel('oC')
    plt.title('Temperatura do Sul por Hora')
    plt.grid()
    plt.savefig('tempSULHora.png')   
    plt.show()
    
    #NE
    
    plt.figure()
    dados['TEMP_BS_NE_9'].plot(label='9h')
    dados['TEMP_BS_NE_15'].plot(label='15h')
    dados['TEMP_BS_NE_21'].plot(label='21h')
    dados['TEMP_BS_NE_MED'].plot(label='Média')
    plt.legend(loc='best')
    plt.ylabel('oC')
    plt.title('Temperatura do Nordeste por Hora')
    plt.grid()
    plt.savefig('tempNEHora.png')   
    plt.show()
    
     #N
    plt.figure()
    dados['TEMP_BS_N_9'].plot( label='9h')
    dados['TEMP_BS_N_15'].plot(label='15h')
    dados['TEMP_BS_N_21'].plot( label='21h')
    dados['TEMP_BS_N_MED'].plot(label='Média')
    plt.legend(loc='best')
    plt.ylabel('oC')
    plt.title('Temperatura do Norte por Hora')
    plt.grid()
    plt.savefig('tempNHora.png')   
    plt.show()
     
    
"""   
    
    
    
    
        #Tempratura Ajustada
    plt.figure()
    dados['TEMPaj_BS_BR_MED'].plot( label='Brasil')
    dados['TEMPaj_BS_SECO_MED'].plot(label='Sud/CO')
    dados['TEMPaj_BS_SUL_MED'].plot(label='Sul')
    dados['TEMPaj_BS_NE_MED'].plot( label='Nord')
    dados['TEMPaj_BS_N_MED'].plot(label='Norte')
    plt.legend(loc='best')
    plt.ylabel('oC')
    plt.title('Temperatura Ajustada por Região')
    #plt.grid()
    plt.savefig('tempajustmed.png')   
    plt.show()

    
    dados['TEMP_BS_BR_MED'].describe()
    dados['TEMP_BS_SECO_MED'].describe()
    dados['TEMP_BS_SUL_MED'].describe()
    dados['TEMP_BS_NE_MED'].describe()
    dados['TEMP_BS_N_MED'].describe()
    
        
    
    
    #Brasil
    
    plt.figure()
    dados['TEMPaj_BS_BR_9'].plot(label='9h')
    dados['TEMPaj_BS_BR_15'].plot( label='15h')
    dados['TEMPaj_BS_BR_21'].plot( label='21h')
    dados['TEMPaj_BS_BR_MED'].plot(label='Média')
    plt.legend(loc='best')
    plt.ylabel('oC')
    plt.title('Temperatura do Brasil por Hora')
    plt.grid()
    plt.savefig('tempajBRHora.png')   
    plt.show()
    
    
    
    
    #Regios agrupadas
    fig2, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 7))
    axes[0].set_title("Sudeste/Centro-Oeste")
    axes[0].plot(periodo, dados['TEMPaj_BS_SECO_9'], label="9h")
    axes[0].plot(periodo, dados['TEMPaj_BS_SECO_15'], label="15h")
    axes[0].plot(periodo, dados['TEMPaj_BS_SECO_21'], label="21h")
    axes[0].plot(periodo, dados['TEMPaj_BS_SECO_MED'], label="Média")
    axes[0].legend(loc='best')
    axes[0].set_ylabel("oC")
    
    axes[1].set_title("Sul")
    axes[1].plot(periodo, dados['TEMPaj_BS_SUL_9'], label="9h")
    axes[1].plot(periodo, dados['TEMPaj_BS_SUL_15'], label="15h")
    axes[1].plot(periodo, dados['TEMPaj_BS_SUL_21'], label="21h")
    axes[1].plot(periodo, dados['TEMPaj_BS_SUL_MED'], label="Média")
    axes[1].legend(loc='best')    
    axes[1].set_ylabel("oC")
    
    axes[2].set_title("Nordeste")
    axes[2].plot(periodo, dados['TEMPaj_BS_NE_9'], label="9h")
    axes[2].plot(periodo, dados['TEMPaj_BS_NE_15'], label="15h")
    axes[2].plot(periodo, dados['TEMPaj_BS_NE_21'], label="21h")
    axes[2].plot(periodo, dados['TEMPaj_BS_NE_MED'], label="Média")
    axes[2].legend(loc='best')
    axes[2].set_ylabel("oC")
    
    axes[3].set_title("Norte")
    axes[3].plot(periodo, dados['TEMPaj_BS_N_9'], label="9h")
    axes[3].plot(periodo, dados['TEMPaj_BS_N_15'], label="15h")
    axes[3].plot(periodo, dados['TEMPaj_BS_N_21'], label="21h")
    axes[3].plot(periodo, dados['TEMPaj_BS_N_MED'], label="Média")
    axes[3].legend(loc='best')
    axes[3].set_ylabel("oC")

    fig2.tight_layout()
    plt.savefig('tempaajgreg.png')   

    plt.show()
    
    
 """   
    Graficos individuais por região
    #SECO
    
    plt.figure()
    dados['TEMPaj_BS_SECO_9'].plot(label='9h')
    dados['TEMPaj_BS_SECO_15'].plot(label='15h')
    dados['TEMPaj_BS_SECO_21'].plot(label='21h')
    dados['TEMPaj_BS_SECO_MED'].plot(label='Média')
    plt.legend(loc='best')
    plt.ylabel('oC')
    plt.title('Temperatura do SE/CO por Hora')
    plt.grid()
    plt.savefig('tempajSECOHora.png')   
    plt.show()
    
    
    
    #Sul
    
    plt.figure()
    dados['TEMPaj_BS_SUL_9'].plot(label='9h')
    dados['TEMPaj_BS_SUL_15'].plot(label='15h')
    dados['TEMPaj_BS_SUL_21'].plot(label='21h')
    dados['TEMPaj_BS_SUL_MED'].plot(label='Média')
    plt.legend(loc='best')
    plt.ylabel('oC')
    plt.title('Temperatura do Sul por Hora')
    plt.grid()
    plt.savefig('tempajSULHora.png')   
    plt.show()
    
    #NE
    
    plt.figure()
    dados['TEMPaj_BS_NE_9'].plot(label='9h')
    dados['TEMPaj_BS_NE_15'].plot(label='15h')
    dados['TEMPaj_BS_NE_21'].plot(label='21h')
    dados['TEMPaj_BS_NE_MED'].plot(label='Média')
    plt.legend(loc='best')
    plt.ylabel('oC')
    plt.title('Temperatura do Nordeste por Hora')
    plt.grid()
    plt.savefig('tempajNEHora.png')   
    plt.show()
    
     #N
    plt.figure()
    dados['TEMPaj_BS_N_9'].plot(label='9h')
    dados['TEMPaj_BS_N_15'].plot(label='15h')
    dados['TEMPaj_BS_N_21'].plot(label='21h')
    dados['TEMPaj_BS_N_MED'].plot(label='Média')
    plt.legend(loc='best')
    plt.ylabel('oC')
    plt.title('Temperatura do Norte por Hora')
    plt.grid()
    plt.savefig('tempajNHora.png')   
    plt.show()
     
    
"""


    #LAG plot
        from pandas import Series
        from matplotlib import pyplot
        from pandas.tools.plotting import lag_plot
  
    lag_plot(dados['CEE_BR_TOT'])
    plt.tight_layout()
    plt.savefig('CEEBRlag.png')   

    

     #lagplot agregado   
     

    plt.subplot(221)
    lag_plot(dados['CEE_SECO_TOT'], alpha=0.5)
    plt.title('Sudeste/Centro-Oeste')
    plt.xlabel('')
    #plt.ylabel('$y_{(t+1)}$')
    plt.xticks([])
    plt.yticks([])
  
    plt.subplot(222)
    lag_plot(dados['CEE_SUL_TOT'], alpha=0.5)
    plt.title('Sul')
    plt.xlabel('')
    #plt.ylabel('')
    plt.xticks([])
    plt.yticks([])
        
    plt.subplot(223)
    lag_plot(dados['CEE_NE_TOT'], alpha=0.5)
    plt.title('Nordeste')
    #plt.xlabel('$$')
    #plt.ylabel('$y_{(t+1)}$')
    plt.xticks([])
    plt.yticks([])
       
    plt.subplot(224)
    lag_plot(dados['CEE_N_TOT'],alpha=0.5)
    plt.title('Norte')
    #plt.xlabel('$y_{(t)}$')
    #plt.ylabel('')
    plt.xticks([])
    plt.yticks([])    

    plt.tight_layout()
    plt.savefig('CEElagagreg.png')   
    plt.show()





    
    
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
    fig.tight_layout()
    plt.savefig('CEEautocorragreg.png')  
    plot.show()
  
    
    


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

