#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:26:14 2018

@author: pedrocampelo
"""

   import os
   
   os.chdir('/Users/pedrocampelo/Downloads')

   cwd = os.getcwd()
   cwd
    
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
    
        
    del consumo, variaveis, agregado

    
    #Dividindo em teste e treino
    train_size = int(len(dados) * (365/546))
    dados_treino, dados_teste = dados[0:train_size], dados[train_size:len(dados)]

    
    print('Observations: %d' % (len(dados)))
    print('Training Observations: %d' % (len(dados_treino)))
    print('Testing Observations: %d' % (len(dados_teste)))
    
    
   #definindo as matrizes X e y
    
   import patsy as ps
   
   y_seco = dados['CEE_SECO_TOT']







                    #Região Sudeste/CO
    
     y_seco_treino,X_seco_treino = ps.dmatrices('CEE_SECO_TOT ~ DM+DS	+MÊS+ANO+ESTAC+FER + NEB_SECO_9 +\
                                    NEB_SECO_15 + NEB_SECO_21 + NEB_SECO_MED+PA_SECO_9+PA_SECO_15+PA_SECO_21+PA_SECO_MED+TEMP_BS_SECO_9	+ TEMP_BS_SECO_15 + \
                                    TEMP_BS_SECO_21+TEMP_BS_SECO_MED	+TEMP_BU_SECO_9	+TEMP_BU_SECO_15+TEMP_BU_SECO_21+TEMP_BU_SECO_MED	+UMID_SECO_9 + \
                                    UMID_SECO_15 + UMID_SECO_21 + UMID_SECO_MED	+ DV_SECO_9 + DV_SECO_15 + DV_SECO_21	+ DV_SECO_MED + VV_SECO_9	+ VV_SECO_15 + \
                                    VV_SECO_21	+VV_SECO_MED+TAR_SECO_CSO+TAR_SECO_CP+TAR_SECO_IP+TAR_SECO_IND+TAR_SECO_PP	+TAR_SECO_RED+TAR_SECO_RR + \
                                    TAR_SECO_RRA	+TAR_SECO_RRI	+TAR_SECO_SP1	+TAR_SECO_SP2	+TAR_SECO_MED	+Meta_Selic+ Taxa_Selic+CDI+DolarC+DolarC_var + \
                                    DolarV + DolarV_var+EuroC + EuroC_var+EuroV + EuroV_var	+ IBV_Cot	+IBV_min	+ IBV_max	+IBV_varabs	+ \
                                    IBV_varperc	+ IBV_vol	+INPC_m+INPC_ac+IPCA_m+IPCA_ac + IPAM_m + IPAM_ac + IPADI_m + IPADI_ac	+ \
                                    IGPM_m+IGPM_ac + IGPDI_m	+IGPDI_ac	+PAB_o	+PAB_d	+ TVP_o	+ TVP_d	+PICV_o	+ICV_d	+CCU_o + \
                                    CCU_d + CS_o + CS_d+UCPIIT_FGV_o+UCPIIT_FGV_d+CPCIIT_CNI_o+CPCIIT_CNI_d+VIR_o	+VIR_d	+HTPIT_o	+ \
                                    HTPIT_d	+ SRIT_o	+SRIT_d	+PPOB	+PGN	+PIG_o	+PIG_d	+PIBCa_o+PIBCa_d + PIBI_o+PIBI_d+PIBCo_o + \
                                    PIBCo_d + PIA_o	+PIA_d+ICC+INEC+ICEI+DBNDES	+IEG_o	+IEG_d	+IETIT_o	+IETIT_d	+IETC_o	+ \
                                    IETC_d + IETS_o	+IETS_d	+IETCV_o+IETCV_d + PO+TD+BM + PME + \
                                    TEMPaj_BS_SECO_9	+ TEMPaj_BS_SECO_15 + TEMPaj_BS_SECO_21+TEMPaj_BS_SECO_MED	+TEMPaj_BU_SECO_9	+TEMPaj_BU_SECO_15+TEMPaj_BU_SECO_21+TEMPaj_BU_SECO_MED ', 
                                    data=dados_treino, return_type='dataframe')

                                    #X_seco_treino = X_seco_treino.drop(['Intercept'], axis=1)

    y_seco_teste,X_seco_teste = ps.dmatrices('CEE_SECO_TOT ~ DM+DS	+MÊS+ANO+ESTAC+FER + NEB_SECO_9 +\
                                    NEB_SECO_15 + NEB_SECO_21 + NEB_SECO_MED+PA_SECO_9+PA_SECO_15+PA_SECO_21+PA_SECO_MED+TEMP_BS_SECO_9	+ TEMP_BS_SECO_15 + \
                                    TEMP_BS_SECO_21+TEMP_BS_SECO_MED	+TEMP_BU_SECO_9	+TEMP_BU_SECO_15+TEMP_BU_SECO_21+TEMP_BU_SECO_MED	+UMID_SECO_9 + \
                                    UMID_SECO_15 + UMID_SECO_21 + UMID_SECO_MED	+ DV_SECO_9 + DV_SECO_15 + DV_SECO_21	+ DV_SECO_MED + VV_SECO_9	+ VV_SECO_15 + \
                                    VV_SECO_21	+VV_SECO_MED+TAR_SECO_CSO+TAR_SECO_CP+TAR_SECO_IP+TAR_SECO_IND+TAR_SECO_PP	+TAR_SECO_RED+TAR_SECO_RR + \
                                    TAR_SECO_RRA	+TAR_SECO_RRI	+TAR_SECO_SP1	+TAR_SECO_SP2	+TAR_SECO_MED	+Meta_Selic+ Taxa_Selic+CDI+DolarC+DolarC_var + \
                                    DolarV + DolarV_var+EuroC + EuroC_var+EuroV + EuroV_var	+ IBV_Cot	+IBV_min	+ IBV_max	+IBV_varabs	+ \
                                    IBV_varperc	+ IBV_vol	+INPC_m+INPC_ac+IPCA_m+IPCA_ac + IPAM_m + IPAM_ac + IPADI_m + IPADI_ac	+ \
                                    IGPM_m+IGPM_ac + IGPDI_m	+IGPDI_ac	+PAB_o	+PAB_d	+ TVP_o	+ TVP_d	+PICV_o	+ICV_d	+CCU_o + \
                                    CCU_d + CS_o + CS_d+UCPIIT_FGV_o+UCPIIT_FGV_d+CPCIIT_CNI_o+CPCIIT_CNI_d+VIR_o	+VIR_d	+HTPIT_o	+ \
                                    HTPIT_d	+ SRIT_o	+SRIT_d	+PPOB	+PGN	+PIG_o	+PIG_d	+PIBCa_o+PIBCa_d + PIBI_o+PIBI_d+PIBCo_o + \
                                    PIBCo_d + PIA_o	+PIA_d+ICC+INEC+ICEI+DBNDES	+IEG_o	+IEG_d	+IETIT_o	+IETIT_d	+IETC_o	+ \
                                    IETC_d + IETS_o	+IETS_d	+IETCV_o+IETCV_d + PO+TD+BM + PME+\
                                    TEMPaj_BS_SECO_9	+ TEMPaj_BS_SECO_15 + TEMPaj_BS_SECO_21+TEMPaj_BS_SECO_MED	+TEMPaj_BU_SECO_9	+TEMPaj_BU_SECO_15+TEMPaj_BU_SECO_21+TEMPaj_BU_SECO_MED ', 
                                    data=dados_teste, return_type='dataframe')  
                





    
    #Correlação
    
    from pandas import DataFrame
    from pandas import concat
    
    values_seco= DataFrame(dados['CEE_SECO_TOT'].values)
    dataframe = concat([values_seco.shift(1), values_seco], axis=1)
    dataframe.columns = ['t-1','t+1']
    result = dataframe.corr()
    print(result)
    
    
    
    
    from plotly.plotly import plot_mpl
    from statsmodels.tsa.seasonal import seasonal_decompose
   
    dec_seas_seco = seasonal_decompose(y_seco, model='multiplicative')
    fig_seco = dec_seas_seco.plot()


    #Checando estacionariedade da série de consumo de EE    
    
    from statsmodels.tsa.stattools import adfuller, kpss


    def adf_test(y):
        # perform Augmented Dickey Fuller test
        print('Results of Augmented Dickey-Fuller test:')
        dftest = adfuller(y, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value ({})'.format(key)] = value
        print(dfoutput)
       

     data_seco = y_seco_treino.iloc[:,0].values
     data2_seco = y_seco_teste.iloc[:,0].values
     y_seco= dados['CEE_SECO_TOT']
     
     adf_test(data_seco)
     adf_test(data2_seco)
     adf_test(y_seco)
    
    
    #Não podemos rejeitas a hipótese nula a 10%, então a série é não estacionária (série toda e separada).

       
    
    #1st difference
    y_seco_treino_diff = np.diff(data)
    ts_diagnostics(y_seco_treino_diff, lags=30, title='International Airline Passengers diff', filename='adf_diff')
    adf_test(y_seco_treino_diff)     
    
    y_seco_teste_diff=np.diff(data2)
    adf_test(y_seco_teste_diff)     

    #1a diferença é estacionária
    
     
                                            #AR


    from statsmodels.tsa.ar_model import AR
    from statsmodels.tsa.arima_model import ARIMA
    from sklearn.metrics import mean_squared_error

    from sklearn.metrics import accuracy_score




    #Modelo 1 (Sem tirar a primeira diferenca)
    
    model_seco = AR(y_seco_treino)                                #modelo
    model_seco_fit = model_seco.fit()                             #lags
    print('Lag: %s' % model_seco_fit.k_ar)
    print('Coefficients: %s' % model_seco_fit.params)        #coeficientes
    
    R2_seco_AR=0 
    accuracy_seco_AR_seco=0
    R2_seco_AR_tese=0


    # make predictions
    y_seco_predictions = model_seco_fit.predict(start=len(y_seco_treino), end=len(y_seco_treino)+len(y_seco_teste)-1, dynamic=False)

    EQM_seco = mean_squared_error(y_seco_teste, y_seco_predictions)
    resid_seco=np.sqrt(EQM_seco)
    print('Test MSE: %.3f' % EQM_seco, resid_seco)
    

 """   
           #outro modelo arima (consegue decidir o numero de lags - 20 no caso)
    model_seco1 = ARIMA(y_seco_treino, order=(17,0,0))
    model_seco_fit1 = model_seco1.fit()
    print('Lag: %s' % model_seco_fit1.k_ar)
    print('Coefficients: %s' % model_seco_fit1.params)
    # make predictions
    y_seco_predictions1 = model_seco_fit1.predict(start=len(y_seco_treino), end=len(y_seco_treino)+len(y_seco_teste)-1, dynamic=False)
    EQM_seco1 = mean_squared_error(y_seco_teste, y_seco_predictions1)
    resid_seco1 = np.sqrt(EQM_seco1)
    print('Test MSE: %.3f' % EQM_seco1, resid_seco1)
    print(model_seco_fit1.summary())
"""


    
    # plot results
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    
    plt.figure()    
    pyplot.plot(y_seco_treino, label='Treino')
    pyplot.plot(y_seco_teste, color='black', label='Teste')
    pyplot.plot(y_seco_predictions, color='red', label='Previsão')
    #dados['CEE_SECO_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (AR)')
    plt.grid()
    pyplot.show()

    
 """   
    #Modelo 2 (Tirando a primeira diferenca) - Não consegui, deve faltar detalhe
    
    #1)
    y_diff=np.diff(y)
    periodo_diff = pd.date_range('2/1/2017', periods=545)
    y_diff = pd.DataFrame(y_diff,index=periodo_diff)
    y_seco_treino_diff, y_seco_teste_diff = y_diff[0:train_size], y_diff[train_size:len(dados)]
    
    #2)
    treino_diff = pd.date_range('2/1/2017', periods=364)
    teste_diff = pd.date_range('1/31/2018', periods=180)
    y_seco_treino_diff = pd.DataFrame(y_seco_treino_diff,index=treino_diff)
    y_seco_teste_diff = pd.DataFrame(y_seco_teste_diff,index=teste_diff)
    
    
    model_secodiff = AR(y_seco_treino_diff)                          #modelo
    model_secodiff_fit = model_secodiff.fit()                             #lags
    print('Lag: %s' % model_secodiff_fit.k_ar)
    print('Coefficients: %s' % model_secodiff_fit.params)        #coeficientes

    # make predictions
    y_seco_predictions_diff = model_secodiff_fit.predict(start=len(y_seco_treino_diff), end=len(y_seco_treino_diff)+len(y_seco_teste_diff)-1, dynamic=False)

    EQM_seco_diff = mean_squared_error(y_seco_teste_diff, y_seco_predictions_diff)
    resid_seco_diff =np.sqrt(EQM_seco_diff)

    print('Test MSE: %.3f' % EQM_seco_diff, resid_seco_diff)
  
    # plot results
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    
    plt.figure()    
    pyplot.plot(y_seco_treino_diff, label='Treino')
    pyplot.plot(y_seco_teste_diff, color='black', label='Teste')
    pyplot.plot(y_seco_predictions_diff, color='red', label='Previsão')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (AR - 1a Diferença')
    plt.grid()
    pyplot.show()

"""


                           #OLS                
    
    import statsmodels.api as sm
    from sklearn.metrics import r2_score



    model_seco2 = sm.OLS(y_seco_treino,X_seco_treino)                  #modelo
    model_seco_fit2 = model_seco2.fit() 
    print (model_seco_fit2.summary())                        #sumário do modelo
    coef2_seco=model_seco_fit2.params
    
    R2_seco2=model_seco_fit2.rsquared
    
        
    # make predictions
    y_seco_predictions2 = model_seco_fit2.predict(X_seco_teste)          #previsão

    EQM_seco2 = mean_squared_error(y_seco_teste, y_seco_predictions2)    #EQM_seco
    resid_seco2 = np.sqrt(EQM_seco2)                                #Resíduo

    print('Test MSE, resid_secoual: %.3f' % EQM_seco2, resid_seco2)
    
    
    accuracy_seco_2 = r2_score(y_seco_teste, y_seco_predictions2)
    R2_seco_2_teste = sm.OLS(y_seco_teste,X_seco_teste).fit().rsquared
    print ('accuracy_seco, R2_seco_teste: %.3f' % accuracy_seco_2, R2_seco_2_teste)
    
    plt.figure()    
    pyplot.plot(y_seco_treino, label='Treino')
    pyplot.plot(y_seco_teste, color='black', label='Teste')
    pyplot.plot(y_seco_predictions2, color='red', label='Previsão')
    #dados['CEE_SECO_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (OLS)')
    plt.grid()
    pyplot.show()   
    
    
    
    #2)Linear Regression
  
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    reg = LinearRegression().fit(X_seco_treino, y_seco_treino)
    print(reg.score(X_seco_treino, y_seco_treino))                        #R2_seco fora da amostra
    print(reg.coef_)                                            #coeficientes
    coefreg_seco=np.transpose(reg.coef_)

    R2_secoreg=reg.score(X_seco_treino, y_seco_treino)    

    predictionsreg_seco = reg.predict(X_seco_teste)
    y_seco_predictionsreg= pd.DataFrame(predictionsreg_seco, index=teste)   #previsão

    EQM_secoreg = mean_squared_error(y_seco_teste, y_seco_predictionsreg)      #EQM_seco
    resid_secoreg = np.sqrt(EQM_secoreg)                                #resid_secouo
    print('Test MSE, resid_secouo: %.3f' % EQM_secoreg,resid_secoreg)
    
    accuracy_seco_reg = r2_score(y_seco_teste, y_seco_predictionsreg)
    R2_seco_reg_teste = reg.score(X_seco_teste, y_seco_teste)  
    print ('accuracy_seco, R2_seco_teste: %.3f' % accuracy_seco_reg, R2_seco_reg_teste)
    
    
    
    plt.figure()    
    pyplot.plot(y_seco_treino, label='Treino')
    pyplot.plot(y_seco_teste, color='black', label='Teste')
    pyplot.plot(y_seco_predictionsreg, color='red', label='Previsão')
    #dados['CEE_SECO_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (LinearRegression)')
    plt.grid()
    pyplot.show()   



                                            #Lasso
    
    #1)Lasso normal
    
    from sklearn import linear_model
    
    model_seco3 = linear_model.Lasso( alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
    normalize=False, positive=False, precompute=False, random_state=None,
    selection='cyclic', tol=0.0001, warm_start=False)
    model_seco_fit3=model_seco3.fit(X_seco_treino,y_seco_treino)
    coef3_seco=model_seco3.coef_
    
    print(model_seco_fit3.coef_)
    print(model_seco_fit3.intercept_) 
    print(model_seco_fit3.score(X_seco_treino,y_seco_treino))

    R2_seco3 = model_seco_fit3.score(X_seco_treino,y_seco_treino)
    
        # make predictions
    y_seco_predictions3 = model_seco_fit3.predict(X_seco_teste)
    y_seco_predictions3= pd.DataFrame(y_seco_predictions3, index=teste)   #previsão


    EQM_seco3 = mean_squared_error(y_seco_teste, y_seco_predictions3)
    resid_seco3 = np.sqrt(EQM_seco3)
    print('Test MSE, resid_secouo: %.3f' % EQM_seco3,resid_seco3)
    
    accuracy_seco_3 = r2_score(y_seco_teste, y_seco_predictions3)
    R2_seco_3_teste = model_seco_fit3.score(X_seco_teste, y_seco_teste)  
    print ('accuracy_seco, R2_seco_teste: %.3f' % accuracy_seco_3, R2_seco_3_teste)
    


      
    plt.figure()    
    pyplot.plot(y_seco_treino, label='Treino')
    pyplot.plot(y_seco_teste, color='black', label='Teste')
    pyplot.plot(y_seco_predictions3, color='red', label='Previsão')
    #dados['CEE_SECO_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lasso)')
    plt.grid()
    pyplot.show() 
    

    
    #2) Lasso CV (DESASTRE)
    
    from sklearn.linear_model import LassoCV

    
    model_seco4 = LassoCV(cv=365, random_state=0).fit(X_seco_treino, y_seco_treino)
    print(model_seco4.coef_)
    coef4_seco=model_seco4.coef_
    
    R2_seco4 = model_seco4.score(X_seco_treino, y_seco_treino) 


        # make predictions
    y_seco_predictions4 = model_seco4.predict(X_seco_teste)
    y_seco_predictions4= pd.DataFrame(y_seco_predictions4, index=teste)   #previsão


    EQM_seco4 = mean_squared_error(y_seco_teste, y_seco_predictions4)
    resid_seco4 = np.sqrt(EQM_seco4)
    print('Test MSE: %.3f' % EQM_seco4,resid_seco4)
    
    accuracy_seco_4 = r2_score(y_seco_teste, y_seco_predictions4)
    R2_seco_4_teste = model_seco4.score(X_seco_teste, y_seco_teste)  
    print ('accuracy_seco, R2_seco_teste: %.3f' % accuracy_seco_4, R2_seco_4_teste)
    
        
    plt.figure()    
    pyplot.plot(y_seco_treino, label='Treino')
    pyplot.plot(y_seco_teste, color='black', label='Teste')
    pyplot.plot(y_seco_predictions4, color='red', label='Previsão')
    #dados['CEE_SECO_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lasso CV)')
    plt.grid()
    pyplot.show() 
    
    
    
                                        #Lars
                                        
    #1) Lars normal (DESASTRE)
     
    model_seco5 = linear_model.Lars(n_nonzero_coefs=6)
    model_seco5_fit=model_seco5.fit(X_seco_treino, y_seco_treino)
    print(model_seco5_fit.coef_) 
    coef5_seco=model_seco5_fit.coef_
    
    R2_seco5 = model_seco5_fit.score(X_seco_treino, y_seco_treino) 

     
         # make predictions
    y_seco_predictions5 = model_seco5_fit.predict(X_seco_teste)
    y_seco_predictions5= pd.DataFrame(y_seco_predictions5, index=teste)   #previsão


    EQM_seco5 = mean_squared_error(y_seco_teste, y_seco_predictions5)
    resid_seco5 = np.sqrt(EQM_seco5)
    print('Test MSE: %.3f' % EQM_seco5,resid_seco5)
     
    accuracy_seco_5 = r2_score(y_seco_teste, y_seco_predictions5)
    R2_seco_5_teste = model_seco5_fit.score(X_seco_teste, y_seco_teste)  
    print ('accuracy_seco, R2_seco_teste: %.3f' % accuracy_seco_5, R2_seco_5_teste)
       
        
    plt.figure()    
    pyplot.plot(y_seco_treino, label='Treino')
    pyplot.plot(y_seco_teste, color='black', label='Teste')
    pyplot.plot(y_seco_predictions5, color='red', label='Previsão')
    #dados['CEE_SECO_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lars)')
    plt.grid()
    pyplot.show() 
     
    
    #2) Lasso Lars (3 MENOR EQM_seco)
    model_seco6 = linear_model.LassoLars(alpha=0.01).fit(X_seco_treino,y_seco_treino)
    print(model_seco6.coef_) 
    coef6_seco=model_seco6.coef_
    
    R2_seco6 = model_seco6.score(X_seco_treino, y_seco_treino) 

    
    y_seco_predictions6 = model_seco6.predict(X_seco_teste)
    y_seco_predictions6= pd.DataFrame(y_seco_predictions6, index=teste)   #previsão


    EQM_seco6 = mean_squared_error(y_seco_teste, y_seco_predictions6)
    resid_seco6 = np.sqrt(EQM_seco6)
    print('Test MSE: %.3f' % EQM_seco6,resid_seco6)
    
    accuracy_seco_6 = r2_score(y_seco_teste, y_seco_predictions6)
    R2_seco_6_teste = model_seco6.score(X_seco_teste, y_seco_teste)  
    print ('accuracy_seco, R2_seco_teste: %.3f' % accuracy_seco_6, R2_seco_6_teste)
    
        
    plt.figure()    
    pyplot.plot(y_seco_treino, label='Treino')
    pyplot.plot(y_seco_teste, color='black', label='Teste')
    pyplot.plot(y_seco_predictions6, color='red', label='Previsão')
    #dados['CEE_SECO_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (lasso Lars)')
    plt.grid()
    pyplot.show() 
    
    
    
    #3) Lasso Lars com Cross Validation (2 menor EQM_seco)
    
    model_seco7 = linear_model.LassoLarsCV(cv=50).fit(X_seco_treino,y_seco_treino)
    print(model_seco7.coef_)
    
    coef7_seco=model_seco7.coef_

    
    R2_seco7 = model_seco7.score(X_seco_treino, y_seco_treino) 

    
    y_seco_predictions7 = model_seco7.predict(X_seco_teste)
    y_seco_predictions7= pd.DataFrame(y_seco_predictions7, index=teste)   #previsão

    EQM_seco7 = mean_squared_error(y_seco_teste, y_seco_predictions7)
    resid_seco7 = np.sqrt(EQM_seco7)
    print('Test MSE: %.3f' % EQM_seco7,resid_seco7)
    
    accuracy_seco_7 = r2_score(y_seco_teste, y_seco_predictions7)
    R2_seco_7_teste = model_seco7.score(X_seco_teste, y_seco_teste)  
    print ('accuracy_seco, R2_seco_teste: %.3f' % accuracy_seco_7, R2_seco_7_teste)
    
        
    plt.figure()    
    pyplot.plot(y_seco_treino, label='Treino')
    pyplot.plot(y_seco_teste, color='black', label='Teste')
    pyplot.plot(y_seco_predictions7, color='red', label='Previsão')
    #dados['CEE_SECO_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lasso Lars CV)')
    plt.grid()
    pyplot.show() 
    
    
     
                                #Ridge Regression (MENOR EQM_seco)
    
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
    
    model_seco8 = Ridge(alpha=0.1,normalize=True)
    model_seco8_fit=model_seco8.fit(X_seco_treino, y_seco_treino)
    
    coef8_seco=np.transpose(model_seco8_fit.coef_)

        
    R2_seco8 = model_seco8_fit.score(X_seco_treino, y_seco_treino) 

    
    y_seco_predictions8 = model_seco8_fit.predict(X_seco_teste)
    y_seco_predictions8= pd.DataFrame(y_seco_predictions8, index=teste)   #previsão

    EQM_seco8 = mean_squared_error(y_seco_teste, y_seco_predictions8)
    resid_seco8 = np.sqrt(EQM_seco8)
    print('Test MSE: %.3f' % EQM_seco8,resid_seco8)
    
    
    accuracy_seco_8 = r2_score(y_seco_teste, y_seco_predictions8)
    R2_seco_8_teste = model_seco8_fit.score(X_seco_teste, y_seco_teste)  
    print ('accuracy_seco, R2_seco_teste: %.3f' % accuracy_seco_8, R2_seco_8_teste)
    
        
    plt.figure()    
    pyplot.plot(y_seco_treino, label='Treino')
    pyplot.plot(y_seco_teste, color='black', label='Teste')
    pyplot.plot(y_seco_predictions8, color='red', label='Previsão')
    #dados['CEE_SECO_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Ridge)')
    plt.grid()
    pyplot.show() 
    

    
                                #ElasticNet (4 MENOR EQM_seco)
                                
    #1) ElasticNet 
    
      from sklearn.linear_model import ElasticNet

    
    model_seco90 = ElasticNet().fit(X_seco_treino,y_seco_treino)
    print(model_seco90.coef_) 

    R2_seco90 = model_seco90.score(X_seco_treino, y_seco_treino) 
    coef90_seco=model_seco90.coef_
    
    y_seco_predictions90 = model_seco90.predict(X_seco_teste)
    y_seco_predictions90= pd.DataFrame(y_seco_predictions90, index=teste)   #previsão

    EQM_seco90 = mean_squared_error(y_seco_teste, y_seco_predictions90)
    resid_seco90 = np.sqrt(EQM_seco90)
    print('Test MSE: %.3f' % EQM_seco90,resid_seco90)
    
    accuracy_seco_90 = r2_score(y_seco_teste, y_seco_predictions90)
    R2_seco_90_teste = model_seco90.score(X_seco_teste, y_seco_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_seco_90, R2_seco_90_teste)
        
    plt.figure()    
    pyplot.plot(y_seco_treino, label='Treino')
    pyplot.plot(y_seco_teste, color='black', label='Teste')
    pyplot.plot(y_seco_predictions90, color='red', label='Previsão')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (ElasticNet)')
    plt.grid()
    pyplot.show() 
    
    
    
    
    
    
    
    
    #2) ElasticNetCV
    
        from sklearn.linear_model import ElasticNetCV

    model_seco9 = ElasticNetCV(alphas=None, copy_X=True, cv=20, eps=0.001, fit_intercept=True,
       l1_ratio=0.5, max_iter=1000, n_alphas=100, n_jobs=None,
       normalize=False, positive=False, precompute='auto', random_state=0,
       selection='cyclic', tol=0.0001, verbose=0).fit(X_seco_treino,y_seco_treino)
    print(model_seco9.coef_) 

    R2_seco9 = model_seco9.score(X_seco_treino, y_seco_treino) 
    coef9_seco=model_seco9.coef_
    
    y_seco_predictions9 = model_seco9.predict(X_seco_teste)
    y_seco_predictions9= pd.DataFrame(y_seco_predictions9, index=teste)   #previsão

    EQM_seco9 = mean_squared_error(y_seco_teste, y_seco_predictions9)
    resid_seco9 = np.sqrt(EQM_seco9)
    print('Test MSE: %.3f' % EQM_seco9,resid_seco9)
    
    accuracy_seco_9 = r2_score(y_seco_teste, y_seco_predictions9)
    R2_seco_9_teste = model_seco9.score(X_seco_teste, y_seco_teste)  
    print ('accuracy_seco, R2_seco_teste: %.3f' % accuracy_seco_9, R2_seco_9_teste)
        
    plt.figure()    
    pyplot.plot(y_seco_treino, label='Treino')
    pyplot.plot(y_seco_teste, color='black', label='Teste')
    pyplot.plot(y_seco_predictions9, color='red', label='Previsão')
    #dados['CEE_SECO_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (ElasticNetCV')
    plt.grid()
    pyplot.show() 
    
                                    #Random Forest  (MELHOR EQM_seco)

    
    from sklearn.ensemble import RandomForestRegressor
    
    model_seco10 = RandomForestRegressor(n_estimators = 1000, random_state = 0).fit(X_seco_treino, y_seco_treino)
    
    print(model_seco10.feature_importances_)
    coef10_seco=model_seco10.feature_importances_
    
    
    R2_seco10 = model_seco10.score(X_seco_treino, y_seco_treino) 
    
    y_seco_predictions10 = model_seco10.predict(X_seco_teste)
    y_seco_predictions10= pd.DataFrame(y_seco_predictions10, index=teste)   #previsão

    EQM_seco10 = mean_squared_error(y_seco_teste, y_seco_predictions10)
    resid_seco10 = np.sqrt(EQM_seco10)
    print('Test MSE: %.3f' % EQM_seco10,resid_seco10)
    
    
    accuracy_seco_10 = r2_score(y_seco_teste, y_seco_predictions10)
    R2_seco_10_teste = model_seco10.score(X_seco_teste, y_seco_teste)  
    print ('accuracy_seco, R2_seco_teste: %.3f' % accuracy_seco_10, R2_seco_10_teste)
    
        
    plt.figure()    
    pyplot.plot(y_seco_treino, label='Treino')
    pyplot.plot(y_seco_teste, color='black', label='Teste')
    pyplot.plot(y_seco_predictions10, color='red', label='Previsão')
    #dados['CEE_SECO_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Random Forest)')
    plt.grid()
    pyplot.show() 
    
    

    
     colunas_seco2 =  ['DM','DS','MÊS','ANO','ESTAC','FER','NEB_SECO_9','NEB_SECO_15','NEB_SECO_21',
                'NEB_SECO_MED','PA_SECO_9','PA_SECO_15','PA_SECO_21','PA_SECO_MED','TEMP_BS_SECO_9','TEMP_BS_SECO_15','TEMP_BS_SECO_21','TEMP_BS_SECO_MED','TEMP_BU_SECO_9','TEMP_BU_SECO_15','TEMP_BU_SECO_21','TEMP_BU_SECO_MED','UMID_SECO_9', 
                'UMID_SECO_15','UMID_SECO_21','UMID_SECO_MED','DV_SECO_9','DV_SECO_15','DV_SECO_21','DV_SECO_MED','VV_SECO_9','VV_SECO_15','VV_SECO_21','VV_SECO_MED','TAR_SECO_CSO','TAR_SECO_CP','TAR_SECO_IP','TAR_SECO_IND','TAR_SECO_PP','TAR_SECO_RED',
                'TAR_SECO_RR','TAR_SECO_RRA','TAR_SECO_RRI','TAR_SECO_SP1','TAR_SECO_SP2','TAR_SECO_MED','Meta_Selic', 'Taxa_Selic','CDI','DolarC','DolarC_var','DolarV','DolarV_var','EuroC','EuroC_var','EuroV','EuroV_var','IBV_Cot',
                'IBV_min','IBV_max','IBV_varabs','IBV_varperc','IBV_vol','INPC_m','INPC_ac','IPCA_m','IPCA_ac','IPAM_m','IPAM_ac','IPADI_m', 'IPADI_ac' , 'IGPM_m','IGPM_ac','IGPDI_m','IGPDI_ac','PAB_o','PAB_d',
                'TVP_o','TVP_d','PICV_o','ICV_d','CCU_o','CCU_d','CS_o','CS_d','UCPIIT_FGV_o','UCPIIT_FGV_d','CPCIIT_CNI_o','CPCIIT_CNI_d','VIR_o','VIR_d','HTPIT_o','HTPIT_d','SRIT_o','SRIT_d','PPOB','PGN','PIG_o','PIG_d','PIBCa_o','PIBCa_d',
                'PIBI_o','PIBI_d','PIBCo_o','PIBCo_d','PIA_o','PIA_d','ICC','INEC','ICEI','DBNDES','IEG_o','IEG_d','IETIT_o','IETIT_d','IETC_o','IETC_d','IETS_o','IETS_d','IETCV_o','IETCV_d', 'PO','TD','BM','PME',
                'TEMPaj_BS_SECO_9','TEMPaj_BS_SECO_15','TEMPaj_BS_SECO_21','TEMPaj_BS_SECO_MED','TEMPaj_BU_N_MED','TEMPaj_BU_SECO_9','TEMPaj_BU_SECO_15','TEMPaj_BU_SECO_21','TEMPaj_BU_SECO_MED'] 


    coef_seco = pd.DataFrame(coef2_seco, index=colunas_seco2)
    coef_seco.columns = ['OLS']
    coef_seco['LinearRegression']=coefreg_seco
    coef_seco['Lasso']=coef3_seco
    coef_seco['LassoCV']=coef4_seco
    coef_seco['Lars']=coef5_seco
    coef_seco['LassoLars']=coef6_seco
    coef_seco['LassoLarsCV']=coef7_seco
    coef_seco['Ridge']=coef8_seco
    coef_seco['ElasticNet']=coef90_seco    
    coef_seco['ElasticNetCV']=coef9_seco
    coef_seco['RandomForest']=coef10_seco
    
    
    
    
    R2_seco_list       = [R2_seco_AR,R2_seco2,R2_secoreg, R2_seco3,R2_seco4,R2_seco5,R2_seco6,R2_seco7,R2_seco8,R2_seco90,R2_seco9,R2_seco10] 
    EQM_seco_list      = [EQM_seco,EQM_seco2,EQM_secoreg, EQM_seco3,EQM_seco4,EQM_seco5,EQM_seco6,EQM_seco7,EQM_seco8,EQM_seco90,EQM_seco9,EQM_seco10]
    resid_seco_list    = [resid_seco,resid_seco2,resid_secoreg, resid_seco3,resid_seco4,resid_seco5,resid_seco6,resid_seco7,
                          resid_seco8,resid_seco90,resid_seco9,resid_seco10]
    accuracy_seco_list = [accuracy_seco_AR_seco,accuracy_seco_2,accuracy_seco_reg,accuracy_seco_3,accuracy_seco_4,accuracy_seco_5,
                          accuracy_seco_6,accuracy_seco_7,accuracy_seco_8,accuracy_seco_90,accuracy_seco_9,accuracy_seco_10]
    R2_seco_test_list  = [R2_seco_AR_tese,R2_seco_2_teste,R2_seco_reg_teste, R2_seco_3_teste,R2_seco_4_teste,R2_seco_5_teste,
                          R2_seco_6_teste,R2_seco_7_teste,R2_seco_8_teste,R2_seco_90_teste,R2_seco_9_teste,R2_seco_10_teste]   
    
    
    
    index=['R2_seco', 'EQM_seco', 'Resíduo_seco','accuracy_seco','R2_seco teste']
    
    colunas3 = ['AR','OLS','LinearRegression','Lasso','LassoCV','Lars','LassoLars',
                'LassoLarsCV','Ridge','ElasticNet','ElasticNetCV','RandomForest']
    

    
    previsao_seco = pd.DataFrame([R2_seco_list, EQM_seco_list, resid_seco_list, accuracy_seco_list,
                             R2_seco_test_list],index=index, columns=colunas3)                  
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                        #Região Sul
    
     y_sul =dados['CEE_SUL_TOT']
        
     import patsy as ps

    
     y_sul_treino,X_sul_treino = ps.dmatrices('CEE_SUL_TOT ~ DM+DS	+MÊS+ANO+ESTAC+FER + NEB_SUL_9 +\
                                    NEB_SUL_15 + NEB_SUL_21 + NEB_SUL_MED+PA_SUL_9+PA_SUL_15+PA_SUL_21+PA_SUL_MED+TEMP_BS_SUL_9	+ TEMP_BS_SUL_15 + \
                                    TEMP_BS_SUL_21+TEMP_BS_SUL_MED	+TEMP_BU_SUL_9	+TEMP_BU_SUL_15+TEMP_BU_SUL_21+TEMP_BU_SUL_MED	+UMID_SUL_9 + \
                                    UMID_SUL_15 + UMID_SUL_21 + UMID_SUL_MED	+ DV_SUL_9 + DV_SUL_15 + DV_SUL_21	+ DV_SUL_MED + VV_SUL_9	+ VV_SUL_15 + \
                                    VV_SUL_21	+VV_SUL_MED+TAR_SUL_CSO+TAR_SUL_CP+TAR_SUL_IP+TAR_SUL_IND+TAR_SUL_PP	+TAR_SUL_RED+TAR_SUL_RR + \
                                    TAR_SUL_RRA	+TAR_SUL_RRI	+TAR_SUL_SP1	+TAR_SUL_SP2	+TAR_SUL_MED	+Meta_Selic+ Taxa_Selic+CDI+DolarC+DolarC_var + \
                                    DolarV + DolarV_var+EuroC + EuroC_var+EuroV + EuroV_var	+ IBV_Cot	+IBV_min	+ IBV_max	+IBV_varabs	+ \
                                    IBV_varperc	+ IBV_vol	+INPC_m+INPC_ac+IPCA_m+IPCA_ac + IPAM_m + IPAM_ac + IPADI_m + IPADI_ac	+ \
                                    IGPM_m+IGPM_ac + IGPDI_m	+IGPDI_ac	+PAB_o	+PAB_d	+ TVP_o	+ TVP_d	+PICV_o	+ICV_d	+CCU_o + \
                                    CCU_d + CS_o + CS_d+UCPIIT_FGV_o+UCPIIT_FGV_d+CPCIIT_CNI_o+CPCIIT_CNI_d+VIR_o	+VIR_d	+HTPIT_o	+ \
                                    HTPIT_d	+ SRIT_o	+SRIT_d	+PPOB	+PGN	+PIG_o	+PIG_d	+PIBCa_o+PIBCa_d + PIBI_o+PIBI_d+PIBCo_o + \
                                    PIBCo_d + PIA_o	+PIA_d+ICC+INEC+ICEI+DBNDES	+IEG_o	+IEG_d	+IETIT_o	+IETIT_d	+IETC_o	+ \
                                    IETC_d + IETS_o	+IETS_d	+IETCV_o+IETCV_d + PO+TD+BM + PME + \
                                    TEMPaj_BS_SUL_9	+ TEMPaj_BS_SUL_15 + TEMPaj_BS_SUL_21+TEMPaj_BS_SUL_MED	+TEMPaj_BU_SUL_9	+TEMPaj_BU_SUL_15+TEMPaj_BU_SUL_21+TEMPaj_BU_SUL_MED ', 
                                    data=dados_treino, return_type='dataframe')

                                    #X_SUL_treino = X_SUL_treino.drop(['Intercept'], axis=1)

    y_sul_teste,X_sul_teste = ps.dmatrices('CEE_SUL_TOT ~ DM+DS	+MÊS+ANO+ESTAC+FER + NEB_SUL_9 +\
                                    NEB_SUL_15 + NEB_SUL_21 + NEB_SUL_MED+PA_SUL_9+PA_SUL_15+PA_SUL_21+PA_SUL_MED+TEMP_BS_SUL_9	+ TEMP_BS_SUL_15 + \
                                    TEMP_BS_SUL_21+TEMP_BS_SUL_MED	+TEMP_BU_SUL_9	+TEMP_BU_SUL_15+TEMP_BU_SUL_21+TEMP_BU_SUL_MED	+UMID_SUL_9 + \
                                    UMID_SUL_15 + UMID_SUL_21 + UMID_SUL_MED	+ DV_SUL_9 + DV_SUL_15 + DV_SUL_21	+ DV_SUL_MED + VV_SUL_9	+ VV_SUL_15 + \
                                    VV_SUL_21	+VV_SUL_MED+TAR_SUL_CSO+TAR_SUL_CP+TAR_SUL_IP+TAR_SUL_IND+TAR_SUL_PP	+TAR_SUL_RED+TAR_SUL_RR + \
                                    TAR_SUL_RRA	+TAR_SUL_RRI	+TAR_SUL_SP1	+TAR_SUL_SP2	+TAR_SUL_MED	+Meta_Selic+ Taxa_Selic+CDI+DolarC+DolarC_var + \
                                    DolarV + DolarV_var+EuroC + EuroC_var+EuroV + EuroV_var	+ IBV_Cot	+IBV_min	+ IBV_max	+IBV_varabs	+ \
                                    IBV_varperc	+ IBV_vol	+INPC_m+INPC_ac+IPCA_m+IPCA_ac + IPAM_m + IPAM_ac + IPADI_m + IPADI_ac	+ \
                                    IGPM_m+IGPM_ac + IGPDI_m	+IGPDI_ac	+PAB_o	+PAB_d	+ TVP_o	+ TVP_d	+PICV_o	+ICV_d	+CCU_o + \
                                    CCU_d + CS_o + CS_d+UCPIIT_FGV_o+UCPIIT_FGV_d+CPCIIT_CNI_o+CPCIIT_CNI_d+VIR_o	+VIR_d	+HTPIT_o	+ \
                                    HTPIT_d	+ SRIT_o	+SRIT_d	+PPOB	+PGN	+PIG_o	+PIG_d	+PIBCa_o+PIBCa_d + PIBI_o+PIBI_d+PIBCo_o + \
                                    PIBCo_d + PIA_o	+PIA_d+ICC+INEC+ICEI+DBNDES	+IEG_o	+IEG_d	+IETIT_o	+IETIT_d	+IETC_o	+ \
                                    IETC_d + IETS_o	+IETS_d	+IETCV_o+IETCV_d + PO+TD+BM + PME+\
                                    TEMPaj_BS_SUL_9	+ TEMPaj_BS_SUL_15 + TEMPaj_BS_SUL_21+TEMPaj_BS_SUL_MED	+TEMPaj_BU_SUL_9	+TEMPaj_BU_SUL_15+TEMPaj_BU_SUL_21+TEMPaj_BU_SUL_MED ', 
                                    data=dados_teste, return_type='dataframe')  
                





    
    #Correlação
    
    from pandas import DataFrame
    from pandas import concat
    
    values_sul= DataFrame(dados['CEE_SUL_TOT'].values)
    dataframe = concat([values_sul.shift(1), values_sul], axis=1)
    dataframe.columns = ['t-1','t+1']
    result = dataframe.corr()
    print(result)
    
    
    
    
    from plotly.plotly import plot_mpl
    from statsmodels.tsa.seasonal import seasonal_decompose
   
    dec_seas_sul = seasonal_decompose(y_sul, model='multiplicative')
    fig_sul = dec_seas_sul.plot()


    #Checando estacionariedade da série de consumo de EE    
    
    from statsmodels.tsa.stattools import adfuller, kpss


    def adf_test(y):
        # perform Augmented Dickey Fuller test
        print('Results of Augmented Dickey-Fuller test:')
        dftest = adfuller(y, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value ({})'.format(key)] = value
        print(dfoutput)
       

     data_sul = y_sul_treino.iloc[:,0].values
     data2_sul = y_sul_teste.iloc[:,0].values
     y_sul= dados['CEE_SUL_TOT']
     
     adf_test(data_sul)
     adf_test(data2_sul)
     adf_test(y_sul)
    
    
    #Não podemos rejeitas a hipótese nula a 10%, então a série é não estacionária (série toda e separada).

       
    
    #1st difference
    y_sul_treino_diff = np.diff(data)
    ts_diagnostics(y_sul_treino_diff, lags=30, title='International Airline Passengers diff', filename='adf_diff')
    adf_test(y_sul_treino_diff)     
    
    y_sul_teste_diff=np.diff(data2)
    adf_test(y_sul_teste_diff)     

    #1a diferença é estacionária
    
     
                                            #AR


    from statsmodels.tsa.ar_model import AR
    from statsmodels.tsa.arima_model import ARIMA
    from sklearn.metrics import mean_squared_error

    from sklearn.metrics import accuracy_score




    #Modelo 1 (Sem tirar a primeira diferenca)
    
    model_sul = AR(y_sul_treino)                                #modelo
    model_sul_fit = model_sul.fit()                             #lags
    print('Lag: %s' % model_sul_fit.k_ar)
    print('Coefficients: %s' % model_sul_fit.params)        #coeficientes
    
    R2_sul_AR=0 
    accuracy_sul_AR_sul=0
    R2_sul_AR_tese=0


    # make predictions
    y_sul_predictions = model_sul_fit.predict(start=len(y_sul_treino), end=len(y_sul_treino)+len(y_sul_teste)-1, dynamic=False)

    EQM_sul = mean_squared_error(y_sul_teste, y_sul_predictions)
    resid_sul=np.sqrt(EQM_sul)
    print('Test MSE: %.3f' % EQM_sul, resid_sul)
    

 """   
           #outro modelo arima (consegue decidir o numero de lags - 20 no caso)
    model_sul1 = ARIMA(y_sul_treino, order=(17,0,0))
    model_sul_fit1 = model_sul1.fit()
    print('Lag: %s' % model_sul_fit1.k_ar)
    print('Coefficients: %s' % model_sul_fit1.params)
    # make predictions
    y_sul_predictions1 = model_sul_fit1.predict(start=len(y_sul_treino), end=len(y_sul_treino)+len(y_sul_teste)-1, dynamic=False)
    EQM_sul1 = mean_squared_error(y_sul_teste, y_sul_predictions1)
    resid_sul1 = np.sqrt(EQM_sul1)
    print('Test MSE: %.3f' % EQM_sul1, resid_sul1)
    print(model_sul_fit1.summary())
"""


    
    # plot results
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    
    plt.figure()    
    pyplot.plot(y_sul_treino, label='Treino')
    pyplot.plot(y_sul_teste, color='black', label='Teste')
    pyplot.plot(y_sul_predictions, color='red', label='Previsão')
    #dados['CEE_sul_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (AR)')
    plt.grid()
    pyplot.show()

    
 """   
    #Modelo 2 (Tirando a primeira diferenca) - Não consegui, deve faltar detalhe
    
    #1)
    y_diff=np.diff(y)
    periodo_diff = pd.date_range('2/1/2017', periods=545)
    y_diff = pd.DataFrame(y_diff,index=periodo_diff)
    y_sul_treino_diff, y_sul_teste_diff = y_diff[0:train_size], y_diff[train_size:len(dados)]
    
    #2)
    treino_diff = pd.date_range('2/1/2017', periods=364)
    teste_diff = pd.date_range('1/31/2018', periods=180)
    y_sul_treino_diff = pd.DataFrame(y_sul_treino_diff,index=treino_diff)
    y_sul_teste_diff = pd.DataFrame(y_sul_teste_diff,index=teste_diff)
    
    
    model_suldiff = AR(y_sul_treino_diff)                          #modelo
    model_suldiff_fit = model_suldiff.fit()                             #lags
    print('Lag: %s' % model_suldiff_fit.k_ar)
    print('Coefficients: %s' % model_suldiff_fit.params)        #coeficientes

    # make predictions
    y_sul_predictions_diff = model_suldiff_fit.predict(start=len(y_sul_treino_diff), end=len(y_sul_treino_diff)+len(y_sul_teste_diff)-1, dynamic=False)

    EQM_sul_diff = mean_squared_error(y_sul_teste_diff, y_sul_predictions_diff)
    resid_sul_diff =np.sqrt(EQM_sul_diff)

    print('Test MSE: %.3f' % EQM_sul_diff, resid_sul_diff)
  
    # plot results
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    
    plt.figure()    
    pyplot.plot(y_sul_treino_diff, label='Treino')
    pyplot.plot(y_sul_teste_diff, color='black', label='Teste')
    pyplot.plot(y_sul_predictions_diff, color='red', label='Previsão')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (AR - 1a Diferença')
    plt.grid()
    pyplot.show()

"""


                           #OLS                
    
    import statsmodels.api as sm
    from sklearn.metrics import r2_score



    model_sul2 = sm.OLS(y_sul_treino,X_sul_treino)                  #modelo
    model_sul_fit2 = model_sul2.fit() 
    print (model_sul_fit2.summary())                        #sumário do modelo
    coef2_sul=model_sul_fit2.params
    
    R2_sul2=model_sul_fit2.rsquared
    
        
    # make predictions
    y_sul_predictions2 = model_sul_fit2.predict(X_sul_teste)          #previsão

    EQM_sul2 = mean_squared_error(y_sul_teste, y_sul_predictions2)    #EQM_sul
    resid_sul2 = np.sqrt(EQM_sul2)                                #Resíduo

    print('Test MSE, resid_sulual: %.3f' % EQM_sul2, resid_sul2)
    
    
    accuracy_sul_2 = r2_score(y_sul_teste, y_sul_predictions2)
    R2_sul_2_teste = sm.OLS(y_sul_teste,X_sul_teste).fit().rsquared
    print ('accuracy_sul, R2_sul_teste: %.3f' % accuracy_sul_2, R2_sul_2_teste)
    
    plt.figure()    
    pyplot.plot(y_sul_treino, label='Treino')
    pyplot.plot(y_sul_teste, color='black', label='Teste')
    pyplot.plot(y_sul_predictions2, color='red', label='Previsão')
    #dados['CEE_sul_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (OLS)')
    plt.grid()
    pyplot.show()   
    
    
    
    #2)Linear Regression
  
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    reg = LinearRegression().fit(X_sul_treino, y_sul_treino)
    print(reg.score(X_sul_treino, y_sul_treino))                        #R2_sul fora da amostra
    print(reg.coef_)                                            #coeficientes
    coefreg_sul=np.transpose(reg.coef_)

    R2_sulreg=reg.score(X_sul_treino, y_sul_treino)    

    predictionsreg_sul = reg.predict(X_sul_teste)
    y_sul_predictionsreg= pd.DataFrame(predictionsreg_sul, index=teste)   #previsão

    EQM_sulreg = mean_squared_error(y_sul_teste, y_sul_predictionsreg)      #EQM_sul
    resid_sulreg = np.sqrt(EQM_sulreg)                                #resid_suluo
    print('Test MSE, resid_sul: %.3f' % EQM_sulreg,resid_sulreg)
    
    accuracy_sul_reg = r2_score(y_sul_teste, y_sul_predictionsreg)
    R2_sul_reg_teste = reg.score(X_sul_teste, y_sul_teste)  
    print ('accuracy_sul, R2_sul_teste: %.3f' % accuracy_sul_reg, R2_sul_reg_teste)
    
    
    
    plt.figure()    
    pyplot.plot(y_sul_treino, label='Treino')
    pyplot.plot(y_sul_teste, color='black', label='Teste')
    pyplot.plot(y_sul_predictionsreg, color='red', label='Previsão')
    #dados['CEE_sul_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (LinearRegression)')
    plt.grid()
    pyplot.show()   



                                            #Lasso
    
    #1)Lasso normal
    
    from sklearn import linear_model
    
    model_sul3 = linear_model.Lasso( alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
    normalize=False, positive=False, precompute=False, random_state=None,
    selection='cyclic', tol=0.0001, warm_start=False)
    model_sul_fit3=model_sul3.fit(X_sul_treino,y_sul_treino)
    coef3_sul=model_sul3.coef_
    
    print(model_sul_fit3.coef_)
    print(model_sul_fit3.intercept_) 
    print(model_sul_fit3.score(X_sul_treino,y_sul_treino))

    R2_sul3 = model_sul_fit3.score(X_sul_treino,y_sul_treino)
    
        # make predictions
    y_sul_predictions3 = model_sul_fit3.predict(X_sul_teste)
    y_sul_predictions3= pd.DataFrame(y_sul_predictions3, index=teste)   #previsão


    EQM_sul3 = mean_squared_error(y_sul_teste, y_sul_predictions3)
    resid_sul3 = np.sqrt(EQM_sul3)
    print('Test MSE, resid_suluo: %.3f' % EQM_sul3,resid_sul3)
    
    accuracy_sul_3 = r2_score(y_sul_teste, y_sul_predictions3)
    R2_sul_3_teste = model_sul_fit3.score(X_sul_teste, y_sul_teste)  
    print ('accuracy_sul, R2_sul_teste: %.3f' % accuracy_sul_3, R2_sul_3_teste)
    


      
    plt.figure()    
    pyplot.plot(y_sul_treino, label='Treino')
    pyplot.plot(y_sul_teste, color='black', label='Teste')
    pyplot.plot(y_sul_predictions3, color='red', label='Previsão')
    #dados['CEE_sul_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lasso)')
    plt.grid()
    pyplot.show() 
    

    
    #2) Lasso CV (DESASTRE)
    
    from sklearn.linear_model import LassoCV

    
    model_sul4 = LassoCV(cv=365, random_state=0).fit(X_sul_treino, y_sul_treino)
    print(model_sul4.coef_)
    coef4_sul=model_sul4.coef_
    
    R2_sul4 = model_sul4.score(X_sul_treino, y_sul_treino) 


        # make predictions
    y_sul_predictions4 = model_sul4.predict(X_sul_teste)
    y_sul_predictions4= pd.DataFrame(y_sul_predictions4, index=teste)   #previsão


    EQM_sul4 = mean_squared_error(y_sul_teste, y_sul_predictions4)
    resid_sul4 = np.sqrt(EQM_sul4)
    print('Test MSE: %.3f' % EQM_sul4,resid_sul4)
    
    accuracy_sul_4 = r2_score(y_sul_teste, y_sul_predictions4)
    R2_sul_4_teste = model_sul4.score(X_sul_teste, y_sul_teste)  
    print ('accuracy_sul, R2_sul_teste: %.3f' % accuracy_sul_4, R2_sul_4_teste)
    
        
    plt.figure()    
    pyplot.plot(y_sul_treino, label='Treino')
    pyplot.plot(y_sul_teste, color='black', label='Teste')
    pyplot.plot(y_sul_predictions4, color='red', label='Previsão')
    #dados['CEE_sul_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lasso CV)')
    plt.grid()
    pyplot.show() 
    
    
    
                                        #Lars
                                        
    #1) Lars normal (DESASTRE)
     
    model_sul5 = linear_model.Lars(n_nonzero_coefs=6)
    model_sul5_fit=model_sul5.fit(X_sul_treino, y_sul_treino)
    print(model_sul5_fit.coef_) 
    coef5_sul=model_sul5_fit.coef_
    
    R2_sul5 = model_sul5_fit.score(X_sul_treino, y_sul_treino) 

     
         # make predictions
    y_sul_predictions5 = model_sul5_fit.predict(X_sul_teste)
    y_sul_predictions5= pd.DataFrame(y_sul_predictions5, index=teste)   #previsão


    EQM_sul5 = mean_squared_error(y_sul_teste, y_sul_predictions5)
    resid_sul5 = np.sqrt(EQM_sul5)
    print('Test MSE: %.3f' % EQM_sul5,resid_sul5)
     
    accuracy_sul_5 = r2_score(y_sul_teste, y_sul_predictions5)
    R2_sul_5_teste = model_sul5_fit.score(X_sul_teste, y_sul_teste)  
    print ('accuracy_sul, R2_sul_teste: %.3f' % accuracy_sul_5, R2_sul_5_teste)
       
        
    plt.figure()    
    pyplot.plot(y_sul_treino, label='Treino')
    pyplot.plot(y_sul_teste, color='black', label='Teste')
    pyplot.plot(y_sul_predictions5, color='red', label='Previsão')
    #dados['CEE_sul_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lars)')
    plt.grid()
    pyplot.show() 
     
    
    #2) Lasso Lars (3 MENOR EQM_sul)
    model_sul6 = linear_model.LassoLars(alpha=0.01).fit(X_sul_treino,y_sul_treino)
    print(model_sul6.coef_) 
    coef6_sul=model_sul6.coef_
    
    R2_sul6 = model_sul6.score(X_sul_treino, y_sul_treino) 

    
    y_sul_predictions6 = model_sul6.predict(X_sul_teste)
    y_sul_predictions6= pd.DataFrame(y_sul_predictions6, index=teste)   #previsão


    EQM_sul6 = mean_squared_error(y_sul_teste, y_sul_predictions6)
    resid_sul6 = np.sqrt(EQM_sul6)
    print('Test MSE: %.3f' % EQM_sul6,resid_sul6)
    
    accuracy_sul_6 = r2_score(y_sul_teste, y_sul_predictions6)
    R2_sul_6_teste = model_sul6.score(X_sul_teste, y_sul_teste)  
    print ('accuracy_sul, R2_sul_teste: %.3f' % accuracy_sul_6, R2_sul_6_teste)
    
        
    plt.figure()    
    pyplot.plot(y_sul_treino, label='Treino')
    pyplot.plot(y_sul_teste, color='black', label='Teste')
    pyplot.plot(y_sul_predictions6, color='red', label='Previsão')
    #dados['CEE_sul_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (lasso Lars)')
    plt.grid()
    pyplot.show() 
    
    
    
    #3) Lasso Lars com Cross Validation (2 menor EQM_sul)
    
    model_sul7 = linear_model.LassoLarsCV(cv=50).fit(X_sul_treino,y_sul_treino)
    print(model_sul7.coef_)
    
    coef7_sul=model_sul7.coef_

    
    R2_sul7 = model_sul7.score(X_sul_treino, y_sul_treino) 

    
    y_sul_predictions7 = model_sul7.predict(X_sul_teste)
    y_sul_predictions7= pd.DataFrame(y_sul_predictions7, index=teste)   #previsão

    EQM_sul7 = mean_squared_error(y_sul_teste, y_sul_predictions7)
    resid_sul7 = np.sqrt(EQM_sul7)
    print('Test MSE: %.3f' % EQM_sul7,resid_sul7)
    
    accuracy_sul_7 = r2_score(y_sul_teste, y_sul_predictions7)
    R2_sul_7_teste = model_sul7.score(X_sul_teste, y_sul_teste)  
    print ('accuracy_sul, R2_sul_teste: %.3f' % accuracy_sul_7, R2_sul_7_teste)
    
        
    plt.figure()    
    pyplot.plot(y_sul_treino, label='Treino')
    pyplot.plot(y_sul_teste, color='black', label='Teste')
    pyplot.plot(y_sul_predictions7, color='red', label='Previsão')
    #dados['CEE_sul_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lasso Lars CV)')
    plt.grid()
    pyplot.show() 
    
    
     
                                #Ridge Regression (MENOR EQM_sul)
    
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
    
    model_sul8 = Ridge(alpha=0.1,normalize=True)
    model_sul8_fit=model_sul8.fit(X_sul_treino, y_sul_treino)
    
    coef8_sul=np.transpose(model_sul8_fit.coef_)

        
    R2_sul8 = model_sul8_fit.score(X_sul_treino, y_sul_treino) 

    
    y_sul_predictions8 = model_sul8_fit.predict(X_sul_teste)
    y_sul_predictions8= pd.DataFrame(y_sul_predictions8, index=teste)   #previsão

    EQM_sul8 = mean_squared_error(y_sul_teste, y_sul_predictions8)
    resid_sul8 = np.sqrt(EQM_sul8)
    print('Test MSE: %.3f' % EQM_sul8,resid_sul8)
    
    
    accuracy_sul_8 = r2_score(y_sul_teste, y_sul_predictions8)
    R2_sul_8_teste = model_sul8_fit.score(X_sul_teste, y_sul_teste)  
    print ('accuracy_sul, R2_sul_teste: %.3f' % accuracy_sul_8, R2_sul_8_teste)
    
        
    plt.figure()    
    pyplot.plot(y_sul_treino, label='Treino')
    pyplot.plot(y_sul_teste, color='black', label='Teste')
    pyplot.plot(y_sul_predictions8, color='red', label='Previsão')
    #dados['CEE_sul_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Ridge)')
    plt.grid()
    pyplot.show() 
    

    
                                #ElasticNet (4 MENOR EQM_sul)
                                
    #1) ElasticNet 
    
      from sklearn.linear_model import ElasticNet

    
    model_sul90 = ElasticNet().fit(X_sul_treino,y_sul_treino)
    print(model_sul90.coef_) 

    R2_sul90 = model_sul90.score(X_sul_treino, y_sul_treino) 
    coef90_sul=model_sul90.coef_
    
    y_sul_predictions90 = model_sul90.predict(X_sul_teste)
    y_sul_predictions90= pd.DataFrame(y_sul_predictions90, index=teste)   #previsão

    EQM_sul90 = mean_squared_error(y_sul_teste, y_sul_predictions90)
    resid_sul90 = np.sqrt(EQM_sul90)
    print('Test MSE: %.3f' % EQM_sul90,resid_sul90)
    
    accuracy_sul_90 = r2_score(y_sul_teste, y_sul_predictions90)
    R2_sul_90_teste = model_sul90.score(X_sul_teste, y_sul_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_sul_90, R2_sul_90_teste)
        
    plt.figure()    
    pyplot.plot(y_sul_treino, label='Treino')
    pyplot.plot(y_sul_teste, color='black', label='Teste')
    pyplot.plot(y_sul_predictions90, color='red', label='Previsão')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (ElasticNet)')
    plt.grid()
    pyplot.show() 
    
    
    
    
    
    
    
    
    #2) ElasticNetCV
    
    from sklearn.linear_model import ElasticNetCV

    model_sul9 = ElasticNetCV(alphas=None, copy_X=True, cv=20, eps=0.001, fit_intercept=True,
       l1_ratio=0.5, max_iter=1000, n_alphas=100, n_jobs=None,
       normalize=False, positive=False, precompute='auto', random_state=0,
       selection='cyclic', tol=0.0001, verbose=0).fit(X_sul_treino,y_sul_treino)
    print(model_sul9.coef_) 

    R2_sul9 = model_sul9.score(X_sul_treino, y_sul_treino) 
    coef9_sul=model_sul9.coef_
    
    y_sul_predictions9 = model_sul9.predict(X_sul_teste)
    y_sul_predictions9= pd.DataFrame(y_sul_predictions9, index=teste)   #previsão

    EQM_sul9 = mean_squared_error(y_sul_teste, y_sul_predictions9)
    resid_sul9 = np.sqrt(EQM_sul9)
    print('Test MSE: %.3f' % EQM_sul9,resid_sul9)
    
    accuracy_sul_9 = r2_score(y_sul_teste, y_sul_predictions9)
    R2_sul_9_teste = model_sul9.score(X_sul_teste, y_sul_teste)  
    print ('accuracy_sul, R2_sul_teste: %.3f' % accuracy_sul_9, R2_sul_9_teste)
        
    plt.figure()    
    pyplot.plot(y_sul_treino, label='Treino')
    pyplot.plot(y_sul_teste, color='black', label='Teste')
    pyplot.plot(y_sul_predictions9, color='red', label='Previsão')
    #dados['CEE_sul_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (ElasticNetCV')
    plt.grid()
    pyplot.show() 
    
                                    #Random Forest  (MELHOR EQM_sul)

    
    from sklearn.ensemble import RandomForestRegressor
    
    model_sul10 = RandomForestRegressor(n_estimators = 1000, random_state = 0).fit(X_sul_treino, y_sul_treino)
    
    print(model_sul10.feature_importances_)
    coef10_sul=model_sul10.feature_importances_
    
    
    R2_sul10 = model_sul10.score(X_sul_treino, y_sul_treino) 
    
    y_sul_predictions10 = model_sul10.predict(X_sul_teste)
    y_sul_predictions10= pd.DataFrame(y_sul_predictions10, index=teste)   #previsão

    EQM_sul10 = mean_squared_error(y_sul_teste, y_sul_predictions10)
    resid_sul10 = np.sqrt(EQM_sul10)
    print('Test MSE: %.3f' % EQM_sul10,resid_sul10)
    
    
    accuracy_sul_10 = r2_score(y_sul_teste, y_sul_predictions10)
    R2_sul_10_teste = model_sul10.score(X_sul_teste, y_sul_teste)  
    print ('accuracy_sul, R2_sul_teste: %.3f' % accuracy_sul_10, R2_sul_10_teste)
    
        
    plt.figure()    
    pyplot.plot(y_sul_treino, label='Treino')
    pyplot.plot(y_sul_teste, color='black', label='Teste')
    pyplot.plot(y_sul_predictions10, color='red', label='Previsão')
    #dados['CEE_sul_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Random Forest)')
    plt.grid()
    pyplot.show() 
    
    

    
     colunas_sul2 =  ['DM','DS','MÊS','ANO','ESTAC','FER','NEB_SUL_9','NEB_SUL_15','NEB_SUL_21',
                'NEB_SUL_MED','PA_SUL_9','PA_SUL_15','PA_SUL_21','PA_SUL_MED','TEMP_BS_SUL_9','TEMP_BS_SUL_15','TEMP_BS_SUL_21','TEMP_BS_SUL_MED','TEMP_BU_SUL_9','TEMP_BU_SUL_15','TEMP_BU_SUL_21','TEMP_BU_SUL_MED','UMID_SUL_9', 
                'UMID_SUL_15','UMID_SUL_21','UMID_SUL_MED','DV_SUL_9','DV_SUL_15','DV_SUL_21','DV_SUL_MED','VV_SUL_9','VV_SUL_15','VV_SUL_21','VV_SUL_MED','TAR_SUL_CSO','TAR_SUL_CP','TAR_SUL_IP','TAR_SUL_IND','TAR_SUL_PP','TAR_SUL_RED',
                'TAR_SUL_RR','TAR_SUL_RRA','TAR_SUL_RRI','TAR_SUL_SP1','TAR_SUL_SP2','TAR_SUL_MED','Meta_Selic', 'Taxa_Selic','CDI','DolarC','DolarC_var','DolarV','DolarV_var','EuroC','EuroC_var','EuroV','EuroV_var','IBV_Cot',
                'IBV_min','IBV_max','IBV_varabs','IBV_varperc','IBV_vol','INPC_m','INPC_ac','IPCA_m','IPCA_ac','IPAM_m','IPAM_ac','IPADI_m', 'IPADI_ac' , 'IGPM_m','IGPM_ac','IGPDI_m','IGPDI_ac','PAB_o','PAB_d',
                'TVP_o','TVP_d','PICV_o','ICV_d','CCU_o','CCU_d','CS_o','CS_d','UCPIIT_FGV_o','UCPIIT_FGV_d','CPCIIT_CNI_o','CPCIIT_CNI_d','VIR_o','VIR_d','HTPIT_o','HTPIT_d','SRIT_o','SRIT_d','PPOB','PGN','PIG_o','PIG_d','PIBCa_o','PIBCa_d',
                'PIBI_o','PIBI_d','PIBCo_o','PIBCo_d','PIA_o','PIA_d','ICC','INEC','ICEI','DBNDES','IEG_o','IEG_d','IETIT_o','IETIT_d','IETC_o','IETC_d','IETS_o','IETS_d','IETCV_o','IETCV_d', 'PO','TD','BM','PME',
                'TEMPaj_BS_SUL_9','TEMPaj_BS_SUL_15','TEMPaj_BS_SUL_21','TEMPaj_BS_SUL_MED','TEMPaj_BU_N_MED','TEMPaj_BU_SUL_9','TEMPaj_BU_SUL_15','TEMPaj_BU_SUL_21','TEMPaj_BU_SUL_MED'] 


    coef_sul = pd.DataFrame(coef2_sul, index=colunas_sul2)
    coef_sul.columns = ['OLS']
    coef_sul['LinearRegression']=coefreg_sul
    coef_sul['Lasso']=coef3_sul
    coef_sul['LassoCV']=coef4_sul
    coef_sul['Lars']=coef5_sul
    coef_sul['LassoLars']=coef6_sul
    coef_sul['LassoLarsCV']=coef7_sul
    coef_sul['Ridge']=coef8_sul
    coef_sul['ElasticNet']=coef90_sul    
    coef_sul['ElasticNetCV']=coef9_sul
    coef_sul['RandomForest']=coef10_sul
    
    
    
    
    R2_sul_list       = [R2_sul_AR,R2_sul2,R2_sulreg, R2_sul3,R2_sul4,R2_sul5,R2_sul6,R2_sul7,R2_sul8,R2_sul90,R2_sul9,R2_sul10] 
    EQM_sul_list      = [EQM_sul,EQM_sul2,EQM_sulreg, EQM_sul3,EQM_sul4,EQM_sul5,EQM_sul6,EQM_sul7,EQM_sul8,EQM_sul90,EQM_sul9,EQM_sul10]
    resid_sul_list    = [resid_sul,resid_sul2,resid_sulreg, resid_sul3,resid_sul4,resid_sul5,resid_sul6,resid_sul7,
                          resid_sul8,resid_sul90,resid_sul9,resid_sul10]
    accuracy_sul_list = [accuracy_sul_AR_sul,accuracy_sul_2,accuracy_sul_reg,accuracy_sul_3,accuracy_sul_4,accuracy_sul_5,
                          accuracy_sul_6,accuracy_sul_7,accuracy_sul_8,accuracy_sul_90,accuracy_sul_9,accuracy_sul_10]
    R2_sul_test_list  = [R2_sul_AR_tese,R2_sul_2_teste,R2_sul_reg_teste, R2_sul_3_teste,R2_sul_4_teste,R2_sul_5_teste,
                          R2_sul_6_teste,R2_sul_7_teste,R2_sul_8_teste,R2_sul_90_teste,R2_sul_9_teste,R2_sul_10_teste]   
    
    
    
    index=['R2_sul', 'EQM_sul', 'Resíduo_sul','accuracy_sul','R2_sul teste']
    
    colunas3 = ['AR','OLS','LinearRegression','Lasso','LassoCV','Lars','LassoLars',
                'LassoLarsCV','Ridge','ElasticNet','ElasticNetCV','RandomForest']
    

    
    previsao_sul = pd.DataFrame([R2_sul_list, EQM_sul_list, resid_sul_list, accuracy_sul_list,
                             R2_sul_test_list],index=index, columns=colunas3)                  
      

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      #definindo as matrizes X e y
    
   import patsy as ps
   
   y_ne = dados['CEE_NE_TOT']







                    #Região Nordeste
    
     y_ne_treino,X_ne_treino = ps.dmatrices('CEE_NE_TOT ~ DM+DS	+MÊS+ANO+ESTAC+FER + NEB_NE_9 +\
                                    NEB_NE_15 + NEB_NE_21 + NEB_NE_MED+PA_NE_9+PA_NE_15+PA_NE_21+PA_NE_MED+TEMP_BS_NE_9	+ TEMP_BS_NE_15 + \
                                    TEMP_BS_NE_21+TEMP_BS_NE_MED	+TEMP_BU_NE_9	+TEMP_BU_NE_15+TEMP_BU_NE_21+TEMP_BU_NE_MED	+UMID_NE_9 + \
                                    UMID_NE_15 + UMID_NE_21 + UMID_NE_MED	+ DV_NE_9 + DV_NE_15 + DV_NE_21	+ DV_NE_MED + VV_NE_9	+ VV_NE_15 + \
                                    VV_NE_21	+VV_NE_MED+TAR_NE_CSO+TAR_NE_CP+TAR_NE_IP+TAR_NE_IND+TAR_NE_PP	+TAR_NE_RED+TAR_NE_RR + \
                                    TAR_NE_RRA	+TAR_NE_RRI	+TAR_NE_SP1	+TAR_NE_SP2	+TAR_NE_MED	+Meta_Selic+ Taxa_Selic+CDI+DolarC+DolarC_var + \
                                    DolarV + DolarV_var+EuroC + EuroC_var+EuroV + EuroV_var	+ IBV_Cot	+IBV_min	+ IBV_max	+IBV_varabs	+ \
                                    IBV_varperc	+ IBV_vol	+INPC_m+INPC_ac+IPCA_m+IPCA_ac + IPAM_m + IPAM_ac + IPADI_m + IPADI_ac	+ \
                                    IGPM_m+IGPM_ac + IGPDI_m	+IGPDI_ac	+PAB_o	+PAB_d	+ TVP_o	+ TVP_d	+PICV_o	+ICV_d	+CCU_o + \
                                    CCU_d + CS_o + CS_d+UCPIIT_FGV_o+UCPIIT_FGV_d+CPCIIT_CNI_o+CPCIIT_CNI_d+VIR_o	+VIR_d	+HTPIT_o	+ \
                                    HTPIT_d	+ SRIT_o	+SRIT_d	+PPOB	+PGN	+PIG_o	+PIG_d	+PIBCa_o+PIBCa_d + PIBI_o+PIBI_d+PIBCo_o + \
                                    PIBCo_d + PIA_o	+PIA_d+ICC+INEC+ICEI+DBNDES	+IEG_o	+IEG_d	+IETIT_o	+IETIT_d	+IETC_o	+ \
                                    IETC_d + IETS_o	+IETS_d	+IETCV_o+IETCV_d + PO+TD+BM + PME + \
                                    TEMPaj_BS_NE_9	+ TEMPaj_BS_NE_15 + TEMPaj_BS_NE_21+TEMPaj_BS_NE_MED	+TEMPaj_BU_NE_9	+TEMPaj_BU_NE_15+TEMPaj_BU_NE_21+TEMPaj_BU_NE_MED ', 
                                    data=dados_treino, return_type='dataframe')

                                    #X_NE_treino = X_NE_treino.drop(['Intercept'], axis=1)

    y_ne_teste,X_ne_teste = ps.dmatrices('CEE_NE_TOT ~ DM+DS	+MÊS+ANO+ESTAC+FER + NEB_NE_9 +\
                                    NEB_NE_15 + NEB_NE_21 + NEB_NE_MED+PA_NE_9+PA_NE_15+PA_NE_21+PA_NE_MED+TEMP_BS_NE_9	+ TEMP_BS_NE_15 + \
                                    TEMP_BS_NE_21+TEMP_BS_NE_MED	+TEMP_BU_NE_9	+TEMP_BU_NE_15+TEMP_BU_NE_21+TEMP_BU_NE_MED	+UMID_NE_9 + \
                                    UMID_NE_15 + UMID_NE_21 + UMID_NE_MED	+ DV_NE_9 + DV_NE_15 + DV_NE_21	+ DV_NE_MED + VV_NE_9	+ VV_NE_15 + \
                                    VV_NE_21	+VV_NE_MED+TAR_NE_CSO+TAR_NE_CP+TAR_NE_IP+TAR_NE_IND+TAR_NE_PP	+TAR_NE_RED+TAR_NE_RR + \
                                    TAR_NE_RRA	+TAR_NE_RRI	+TAR_NE_SP1	+TAR_NE_SP2	+TAR_NE_MED	+Meta_Selic+ Taxa_Selic+CDI+DolarC+DolarC_var + \
                                    DolarV + DolarV_var+EuroC + EuroC_var+EuroV + EuroV_var	+ IBV_Cot	+IBV_min	+ IBV_max	+IBV_varabs	+ \
                                    IBV_varperc	+ IBV_vol	+INPC_m+INPC_ac+IPCA_m+IPCA_ac + IPAM_m + IPAM_ac + IPADI_m + IPADI_ac	+ \
                                    IGPM_m+IGPM_ac + IGPDI_m	+IGPDI_ac	+PAB_o	+PAB_d	+ TVP_o	+ TVP_d	+PICV_o	+ICV_d	+CCU_o + \
                                    CCU_d + CS_o + CS_d+UCPIIT_FGV_o+UCPIIT_FGV_d+CPCIIT_CNI_o+CPCIIT_CNI_d+VIR_o	+VIR_d	+HTPIT_o	+ \
                                    HTPIT_d	+ SRIT_o	+SRIT_d	+PPOB	+PGN	+PIG_o	+PIG_d	+PIBCa_o+PIBCa_d + PIBI_o+PIBI_d+PIBCo_o + \
                                    PIBCo_d + PIA_o	+PIA_d+ICC+INEC+ICEI+DBNDES	+IEG_o	+IEG_d	+IETIT_o	+IETIT_d	+IETC_o	+ \
                                    IETC_d + IETS_o	+IETS_d	+IETCV_o+IETCV_d + PO+TD+BM + PME+\
                                    TEMPaj_BS_NE_9	+ TEMPaj_BS_NE_15 + TEMPaj_BS_NE_21+TEMPaj_BS_NE_MED	+TEMPaj_BU_NE_9	+TEMPaj_BU_NE_15+TEMPaj_BU_NE_21+TEMPaj_BU_NE_MED ', 
                                    data=dados_teste, return_type='dataframe')  
                





    
    #Correlação
    
    from pandas import DataFrame
    from pandas import concat
    
    values_ne= DataFrame(dados['CEE_NE_TOT'].values)
    dataframe = concat([values_ne.shift(1), values_ne], axis=1)
    dataframe.columns = ['t-1','t+1']
    result = dataframe.corr()
    print(result)
    
    
    
    
    from plotly.plotly import plot_mpl
    from statsmodels.tsa.seasonal import seasonal_decompose
   
    dec_seas_ne = seasonal_decompose(y_ne, model='multiplicative')
    fig_ne = dec_seas_ne.plot()


    #Checando estacionariedade da série de consumo de EE    
    
    from statsmodels.tsa.stattools import adfuller, kpss


    def adf_test(y):
        # perform Augmented Dickey Fuller test
        print('Results of Augmented Dickey-Fuller test:')
        dftest = adfuller(y, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value ({})'.format(key)] = value
        print(dfoutput)
       

     data_ne = y_ne_treino.iloc[:,0].values
     data2_ne = y_ne_teste.iloc[:,0].values
     y_ne= dados['CEE_ne_TOT']
     
     adf_test(data_ne)
     adf_test(data2_ne)
     adf_test(y_ne)
    
    
    #Não podemos rejeitas a hipótese nula a 10%, então a série é não estacionária (série toda e separada).

       
    
    #1st difference
    y_ne_treino_diff = np.diff(data)
    ts_diagnostics(y_ne_treino_diff, lags=30, title='International Airline Passengers diff', filename='adf_diff')
    adf_test(y_ne_treino_diff)     
    
    y_ne_teste_diff=np.diff(data2)
    adf_test(y_ne_teste_diff)     

    #1a diferença é estacionária
    
     
                                            #AR


    from statsmodels.tsa.ar_model import AR
    from statsmodels.tsa.arima_model import ARIMA
    from sklearn.metrics import mean_squared_error

    from sklearn.metrics import accuracy_score




    #Modelo 1 (Sem tirar a primeira diferenca)
    
    model_ne = AR(y_ne_treino)                                #modelo
    model_ne_fit = model_ne.fit()                             #lags
    print('Lag: %s' % model_ne_fit.k_ar)
    print('Coefficients: %s' % model_ne_fit.params)        #coeficientes
    
    R2_ne_AR=0 
    accuracy_ne_AR_ne=0
    R2_ne_AR_tese=0


    # make predictions
    y_ne_predictions = model_ne_fit.predict(start=len(y_ne_treino), end=len(y_ne_treino)+len(y_ne_teste)-1, dynamic=False)

    EQM_ne = mean_squared_error(y_ne_teste, y_ne_predictions)
    resid_ne=np.sqrt(EQM_ne)
    print('Test MSE: %.3f' % EQM_ne, resid_ne)
    

 """   
           #outro modelo arima (consegue decidir o numero de lags - 20 no caso)
    model_ne1 = ARIMA(y_ne_treino, order=(17,0,0))
    model_ne_fit1 = model_ne1.fit()
    print('Lag: %s' % model_ne_fit1.k_ar)
    print('Coefficients: %s' % model_ne_fit1.params)
    # make predictions
    y_ne_predictions1 = model_ne_fit1.predict(start=len(y_ne_treino), end=len(y_ne_treino)+len(y_ne_teste)-1, dynamic=False)
    EQM_ne1 = mean_squared_error(y_ne_teste, y_ne_predictions1)
    resid_ne1 = np.sqrt(EQM_ne1)
    print('Test MSE: %.3f' % EQM_ne1, resid_ne1)
    print(model_ne_fit1.summary())
"""


    
    # plot results
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    
    plt.figure()    
    pyplot.plot(y_ne_treino, label='Treino')
    pyplot.plot(y_ne_teste, color='black', label='Teste')
    pyplot.plot(y_ne_predictions, color='red', label='Previsão')
    #dados['CEE_NE_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (AR)')
    plt.grid()
    pyplot.show()

    
 """   
    #Modelo 2 (Tirando a primeira diferenca) - Não consegui, deve faltar detalhe
    
    #1)
    y_diff=np.diff(y)
    periodo_diff = pd.date_range('2/1/2017', periods=545)
    y_diff = pd.DataFrame(y_diff,index=periodo_diff)
    y_ne_treino_diff, y_ne_teste_diff = y_diff[0:train_size], y_diff[train_size:len(dados)]
    
    #2)
    treino_diff = pd.date_range('2/1/2017', periods=364)
    teste_diff = pd.date_range('1/31/2018', periods=180)
    y_ne_treino_diff = pd.DataFrame(y_ne_treino_diff,index=treino_diff)
    y_ne_teste_diff = pd.DataFrame(y_ne_teste_diff,index=teste_diff)
    
    
    model_nediff = AR(y_ne_treino_diff)                          #modelo
    model_nediff_fit = model_nediff.fit()                             #lags
    print('Lag: %s' % model_nediff_fit.k_ar)
    print('Coefficients: %s' % model_nediff_fit.params)        #coeficientes

    # make predictions
    y_ne_predictions_diff = model_nediff_fit.predict(start=len(y_ne_treino_diff), end=len(y_ne_treino_diff)+len(y_ne_teste_diff)-1, dynamic=False)

    EQM_ne_diff = mean_squared_error(y_ne_teste_diff, y_ne_predictions_diff)
    resid_ne_diff =np.sqrt(EQM_ne_diff)

    print('Test MSE: %.3f' % EQM_ne_diff, resid_ne_diff)
  
    # plot results
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    
    plt.figure()    
    pyplot.plot(y_ne_treino_diff, label='Treino')
    pyplot.plot(y_ne_teste_diff, color='black', label='Teste')
    pyplot.plot(y_ne_predictions_diff, color='red', label='Previsão')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (AR - 1a Diferença')
    plt.grid()
    pyplot.show()

"""


                           #OLS                
    
    import statsmodels.api as sm
    from sklearn.metrics import r2_score



    model_ne2 = sm.OLS(y_ne_treino,X_ne_treino)                  #modelo
    model_ne_fit2 = model_ne2.fit() 
    print (model_ne_fit2.summary())                        #sumário do modelo
    coef2_ne=model_ne_fit2.params
    
    R2_ne2=model_ne_fit2.rsquared
    
        
    # make predictions
    y_ne_predictions2 = model_ne_fit2.predict(X_ne_teste)          #previsão

    EQM_ne2 = mean_squared_error(y_ne_teste, y_ne_predictions2)    #EQM_ne
    resid_ne2 = np.sqrt(EQM_ne2)                                #Resíduo

    print('Test MSE, resid_neual: %.3f' % EQM_ne2, resid_ne2)
    
    
    accuracy_ne_2 = r2_score(y_ne_teste, y_ne_predictions2)
    R2_ne_2_teste = sm.OLS(y_ne_teste,X_ne_teste).fit().rsquared
    print ('accuracy_ne, R2_ne_teste: %.3f' % accuracy_ne_2, R2_ne_2_teste)
    
    plt.figure()    
    pyplot.plot(y_ne_treino, label='Treino')
    pyplot.plot(y_ne_teste, color='black', label='Teste')
    pyplot.plot(y_ne_predictions2, color='red', label='Previsão')
    #dados['CEE_NE_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (OLS)')
    plt.grid()
    pyplot.show()   
    
    
    
    #2)Linear Regression
  
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    reg = LinearRegression().fit(X_ne_treino, y_ne_treino)
    print(reg.score(X_ne_treino, y_ne_treino))                        #R2_ne fora da amostra
    print(reg.coef_)                                            #coeficientes
    coefreg_ne=np.transpose(reg.coef_)

    R2_nereg=reg.score(X_ne_treino, y_ne_treino)    

    predictionsreg_ne = reg.predict(X_ne_teste)
    y_ne_predictionsreg= pd.DataFrame(predictionsreg_ne, index=teste)   #previsão

    EQM_nereg = mean_squared_error(y_ne_teste, y_ne_predictionsreg)      #EQM_ne
    resid_nereg = np.sqrt(EQM_nereg)                                #resid_neuo
    print('Test MSE, resid_neuo: %.3f' % EQM_nereg,resid_nereg)
    
    accuracy_ne_reg = r2_score(y_ne_teste, y_ne_predictionsreg)
    R2_ne_reg_teste = reg.score(X_ne_teste, y_ne_teste)  
    print ('accuracy_ne, R2_ne_teste: %.3f' % accuracy_ne_reg, R2_ne_reg_teste)
    
    
    
    plt.figure()    
    pyplot.plot(y_ne_treino, label='Treino')
    pyplot.plot(y_ne_teste, color='black', label='Teste')
    pyplot.plot(y_ne_predictionsreg, color='red', label='Previsão')
    #dados['CEE_ne_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (LinearRegression)')
    plt.grid()
    pyplot.show()   



                                            #Lasso
    
    #1)Lasso normal
    
    from sklearn import linear_model
    
    model_ne3 = linear_model.Lasso( alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
    normalize=False, positive=False, precompute=False, random_state=None,
    selection='cyclic', tol=0.0001, warm_start=False)
    model_ne_fit3=model_ne3.fit(X_ne_treino,y_ne_treino)
    coef3_ne=model_ne3.coef_
    
    print(model_ne_fit3.coef_)
    print(model_ne_fit3.intercept_) 
    print(model_ne_fit3.score(X_ne_treino,y_ne_treino))

    R2_ne3 = model_ne_fit3.score(X_ne_treino,y_ne_treino)
    
        # make predictions
    y_ne_predictions3 = model_ne_fit3.predict(X_ne_teste)
    y_ne_predictions3= pd.DataFrame(y_ne_predictions3, index=teste)   #previsão


    EQM_ne3 = mean_squared_error(y_ne_teste, y_ne_predictions3)
    resid_ne3 = np.sqrt(EQM_ne3)
    print('Test MSE, resid_neuo: %.3f' % EQM_ne3,resid_ne3)
    
    accuracy_ne_3 = r2_score(y_ne_teste, y_ne_predictions3)
    R2_ne_3_teste = model_ne_fit3.score(X_ne_teste, y_ne_teste)  
    print ('accuracy_ne, R2_ne_teste: %.3f' % accuracy_ne_3, R2_ne_3_teste)
    


      
    plt.figure()    
    pyplot.plot(y_ne_treino, label='Treino')
    pyplot.plot(y_ne_teste, color='black', label='Teste')
    pyplot.plot(y_ne_predictions3, color='red', label='Previsão')
    #dados['CEE_ne_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lasso)')
    plt.grid()
    pyplot.show() 
    

    
    #2) Lasso CV (DESASTRE)
    
    from sklearn.linear_model import LassoCV

    
    model_ne4 = LassoCV(cv=365, random_state=0).fit(X_ne_treino, y_ne_treino)
    print(model_ne4.coef_)
    coef4_ne=model_ne4.coef_
    
    R2_ne4 = model_ne4.score(X_ne_treino, y_ne_treino) 


        # make predictions
    y_ne_predictions4 = model_ne4.predict(X_ne_teste)
    y_ne_predictions4= pd.DataFrame(y_ne_predictions4, index=teste)   #previsão


    EQM_ne4 = mean_squared_error(y_ne_teste, y_ne_predictions4)
    resid_ne4 = np.sqrt(EQM_ne4)
    print('Test MSE: %.3f' % EQM_ne4,resid_ne4)
    
    accuracy_ne_4 = r2_score(y_ne_teste, y_ne_predictions4)
    R2_ne_4_teste = model_ne4.score(X_ne_teste, y_ne_teste)  
    print ('accuracy_ne, R2_ne_teste: %.3f' % accuracy_ne_4, R2_ne_4_teste)
    
        
    plt.figure()    
    pyplot.plot(y_ne_treino, label='Treino')
    pyplot.plot(y_ne_teste, color='black', label='Teste')
    pyplot.plot(y_ne_predictions4, color='red', label='Previsão')
    #dados['CEE_ne_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lasso CV)')
    plt.grid()
    pyplot.show() 
    
    
    
                                        #Lars
                                        
    #1) Lars normal (DESASTRE)
     
    model_ne5 = linear_model.Lars(n_nonzero_coefs=6)
    model_ne5_fit=model_ne5.fit(X_ne_treino, y_ne_treino)
    print(model_ne5_fit.coef_) 
    coef5_ne=model_ne5_fit.coef_
    
    R2_ne5 = model_ne5_fit.score(X_ne_treino, y_ne_treino) 

     
         # make predictions
    y_ne_predictions5 = model_ne5_fit.predict(X_ne_teste)
    y_ne_predictions5= pd.DataFrame(y_ne_predictions5, index=teste)   #previsão


    EQM_ne5 = mean_squared_error(y_ne_teste, y_ne_predictions5)
    resid_ne5 = np.sqrt(EQM_ne5)
    print('Test MSE: %.3f' % EQM_ne5,resid_ne5)
     
    accuracy_ne_5 = r2_score(y_ne_teste, y_ne_predictions5)
    R2_ne_5_teste = model_ne5_fit.score(X_ne_teste, y_ne_teste)  
    print ('accuracy_ne, R2_ne_teste: %.3f' % accuracy_ne_5, R2_ne_5_teste)
       
        
    plt.figure()    
    pyplot.plot(y_ne_treino, label='Treino')
    pyplot.plot(y_ne_teste, color='black', label='Teste')
    pyplot.plot(y_ne_predictions5, color='red', label='Previsão')
    #dados['CEE_NE_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lars)')
    plt.grid()
    pyplot.show() 
     
    
    #2) Lasso Lars (3 MENOR EQM_ne)
    model_ne6 = linear_model.LassoLars(alpha=0.01).fit(X_ne_treino,y_ne_treino)
    print(model_ne6.coef_) 
    coef6_ne=model_ne6.coef_
    
    R2_ne6 = model_ne6.score(X_ne_treino, y_ne_treino) 

    
    y_ne_predictions6 = model_ne6.predict(X_ne_teste)
    y_ne_predictions6= pd.DataFrame(y_ne_predictions6, index=teste)   #previsão


    EQM_ne6 = mean_squared_error(y_ne_teste, y_ne_predictions6)
    resid_ne6 = np.sqrt(EQM_ne6)
    print('Test MSE: %.3f' % EQM_ne6,resid_ne6)
    
    accuracy_ne_6 = r2_score(y_ne_teste, y_ne_predictions6)
    R2_ne_6_teste = model_ne6.score(X_ne_teste, y_ne_teste)  
    print ('accuracy_ne, R2_ne_teste: %.3f' % accuracy_ne_6, R2_ne_6_teste)
    
        
    plt.figure()    
    pyplot.plot(y_ne_treino, label='Treino')
    pyplot.plot(y_ne_teste, color='black', label='Teste')
    pyplot.plot(y_ne_predictions6, color='red', label='Previsão')
    #dados['CEE_NE_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (lasso Lars)')
    plt.grid()
    pyplot.show() 
    
    
    
    #3) Lasso Lars com Cross Validation (2 menor EQM_ne)
    
    model_ne7 = linear_model.LassoLarsCV(cv=50).fit(X_ne_treino,y_ne_treino)
    print(model_ne7.coef_)
    
    coef7_ne=model_ne7.coef_

    
    R2_ne7 = model_ne7.score(X_ne_treino, y_ne_treino) 

    
    y_ne_predictions7 = model_ne7.predict(X_ne_teste)
    y_ne_predictions7= pd.DataFrame(y_ne_predictions7, index=teste)   #previsão

    EQM_ne7 = mean_squared_error(y_ne_teste, y_ne_predictions7)
    resid_ne7 = np.sqrt(EQM_ne7)
    print('Test MSE: %.3f' % EQM_ne7,resid_ne7)
    
    accuracy_ne_7 = r2_score(y_ne_teste, y_ne_predictions7)
    R2_ne_7_teste = model_ne7.score(X_ne_teste, y_ne_teste)  
    print ('accuracy_ne, R2_ne_teste: %.3f' % accuracy_ne_7, R2_ne_7_teste)
    
        
    plt.figure()    
    pyplot.plot(y_ne_treino, label='Treino')
    pyplot.plot(y_ne_teste, color='black', label='Teste')
    pyplot.plot(y_ne_predictions7, color='red', label='Previsão')
    #dados['CEE_ne_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lasso Lars CV)')
    plt.grid()
    pyplot.show() 
    
    
     
                                #Ridge Regression (MENOR EQM_ne)
    
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
    
    model_ne8 = Ridge(alpha=0.1,normalize=True)
    model_ne8_fit=model_ne8.fit(X_ne_treino, y_ne_treino)
    
    coef8_ne=np.transpose(model_ne8_fit.coef_)

        
    R2_ne8 = model_ne8_fit.score(X_ne_treino, y_ne_treino) 

    
    y_ne_predictions8 = model_ne8_fit.predict(X_ne_teste)
    y_ne_predictions8= pd.DataFrame(y_ne_predictions8, index=teste)   #previsão

    EQM_ne8 = mean_squared_error(y_ne_teste, y_ne_predictions8)
    resid_ne8 = np.sqrt(EQM_ne8)
    print('Test MSE: %.3f' % EQM_ne8,resid_ne8)
    
    
    accuracy_ne_8 = r2_score(y_ne_teste, y_ne_predictions8)
    R2_ne_8_teste = model_ne8_fit.score(X_ne_teste, y_ne_teste)  
    print ('accuracy_ne, R2_ne_teste: %.3f' % accuracy_ne_8, R2_ne_8_teste)
    
        
    plt.figure()    
    pyplot.plot(y_ne_treino, label='Treino')
    pyplot.plot(y_ne_teste, color='black', label='Teste')
    pyplot.plot(y_ne_predictions8, color='red', label='Previsão')
    #dados['CEE_ne_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Ridge)')
    plt.grid()
    pyplot.show() 
    

    
                                #ElasticNet (4 MENOR EQM_ne)
                                
    #1) ElasticNet 
    
      from sklearn.linear_model import ElasticNet

    
    model_ne90 = ElasticNet().fit(X_ne_treino,y_ne_treino)
    print(model_ne90.coef_) 

    R2_ne90 = model_ne90.score(X_ne_treino, y_ne_treino) 
    coef90_ne=model_ne90.coef_
    
    y_ne_predictions90 = model_ne90.predict(X_ne_teste)
    y_ne_predictions90= pd.DataFrame(y_ne_predictions90, index=teste)   #previsão

    EQM_ne90 = mean_squared_error(y_ne_teste, y_ne_predictions90)
    resid_ne90 = np.sqrt(EQM_ne90)
    print('Test MSE: %.3f' % EQM_ne90,resid_ne90)
    
    accuracy_ne_90 = r2_score(y_ne_teste, y_ne_predictions90)
    R2_ne_90_teste = model_ne90.score(X_ne_teste, y_ne_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_ne_90, R2_ne_90_teste)
        
    plt.figure()    
    pyplot.plot(y_ne_treino, label='Treino')
    pyplot.plot(y_ne_teste, color='black', label='Teste')
    pyplot.plot(y_ne_predictions90, color='red', label='Previsão')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (ElasticNet)')
    plt.grid()
    pyplot.show() 
    
    
    
    
    
    
    
    
    #2) ElasticNetCV
    
        from sklearn.linear_model import ElasticNetCV

    model_ne9 = ElasticNetCV(alphas=None, copy_X=True, cv=20, eps=0.001, fit_intercept=True,
       l1_ratio=0.5, max_iter=1000, n_alphas=100, n_jobs=None,
       normalize=False, positive=False, precompute='auto', random_state=0,
       selection='cyclic', tol=0.0001, verbose=0).fit(X_ne_treino,y_ne_treino)
    print(model_ne9.coef_) 

    R2_ne9 = model_ne9.score(X_ne_treino, y_ne_treino) 
    coef9_ne=model_ne9.coef_
    
    y_ne_predictions9 = model_ne9.predict(X_ne_teste)
    y_ne_predictions9= pd.DataFrame(y_ne_predictions9, index=teste)   #previsão

    EQM_ne9 = mean_squared_error(y_ne_teste, y_ne_predictions9)
    resid_ne9 = np.sqrt(EQM_ne9)
    print('Test MSE: %.3f' % EQM_ne9,resid_ne9)
    
    accuracy_ne_9 = r2_score(y_ne_teste, y_ne_predictions9)
    R2_ne_9_teste = model_ne9.score(X_ne_teste, y_ne_teste)  
    print ('accuracy_ne, R2_ne_teste: %.3f' % accuracy_ne_9, R2_ne_9_teste)
        
    plt.figure()    
    pyplot.plot(y_ne_treino, label='Treino')
    pyplot.plot(y_ne_teste, color='black', label='Teste')
    pyplot.plot(y_ne_predictions9, color='red', label='Previsão')
    #dados['CEE_NE_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (ElasticNetCV')
    plt.grid()
    pyplot.show() 
    
                                    #Random Forest  (MELHOR EQM_ne)

    
    from sklearn.ensemble import RandomForestRegressor
    
    model_ne10 = RandomForestRegressor(n_estimators = 1000, random_state = 0).fit(X_ne_treino, y_ne_treino)
    
    print(model_ne10.feature_importances_)
    coef10_ne=model_ne10.feature_importances_
    
    
    R2_ne10 = model_ne10.score(X_ne_treino, y_ne_treino) 
    
    y_ne_predictions10 = model_ne10.predict(X_ne_teste)
    y_ne_predictions10= pd.DataFrame(y_ne_predictions10, index=teste)   #previsão

    EQM_ne10 = mean_squared_error(y_ne_teste, y_ne_predictions10)
    resid_ne10 = np.sqrt(EQM_ne10)
    print('Test MSE: %.3f' % EQM_ne10,resid_ne10)
    
    
    accuracy_ne_10 = r2_score(y_ne_teste, y_ne_predictions10)
    R2_ne_10_teste = model_ne10.score(X_ne_teste, y_ne_teste)  
    print ('accuracy_ne, R2_ne_teste: %.3f' % accuracy_ne_10, R2_ne_10_teste)
    
        
    plt.figure()    
    pyplot.plot(y_ne_treino, label='Treino')
    pyplot.plot(y_ne_teste, color='black', label='Teste')
    pyplot.plot(y_ne_predictions10, color='red', label='Previsão')
    #dados['CEE__TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Random Forest)')
    plt.grid()
    pyplot.show() 
    
    

    
     colunas_ne2 =  ['DM','DS','MÊS','ANO','ESTAC','FER','NEB_NE_9','NEB_NE_15','NEB_NE_21',
                'NEB_NE_MED','PA_NE_9','PA_NE_15','PA_NE_21','PA_NE_MED','TEMP_BS_NE_9','TEMP_BS_NE_15','TEMP_BS_NE_21','TEMP_BS_NE_MED','TEMP_BU_NE_9','TEMP_BU_NE_15','TEMP_BU_NE_21','TEMP_BU_NE_MED','UMID_NE_9', 
                'UMID_NE_15','UMID_NE_21','UMID_NE_MED','DV_NE_9','DV_NE_15','DV_NE_21','DV_NE_MED','VV_NE_9','VV_NE_15','VV_NE_21','VV_NE_MED','TAR_NE_CSO','TAR_NE_CP','TAR_NE_IP','TAR_NE_IND','TAR_NE_PP','TAR_NE_RED',
                'TAR_NE_RR','TAR_NE_RRA','TAR_NE_RRI','TAR_NE_SP1','TAR_NE_SP2','TAR_NE_MED','Meta_Selic', 'Taxa_Selic','CDI','DolarC','DolarC_var','DolarV','DolarV_var','EuroC','EuroC_var','EuroV','EuroV_var','IBV_Cot',
                'IBV_min','IBV_max','IBV_varabs','IBV_varperc','IBV_vol','INPC_m','INPC_ac','IPCA_m','IPCA_ac','IPAM_m','IPAM_ac','IPADI_m', 'IPADI_ac' , 'IGPM_m','IGPM_ac','IGPDI_m','IGPDI_ac','PAB_o','PAB_d',
                'TVP_o','TVP_d','PICV_o','ICV_d','CCU_o','CCU_d','CS_o','CS_d','UCPIIT_FGV_o','UCPIIT_FGV_d','CPCIIT_CNI_o','CPCIIT_CNI_d','VIR_o','VIR_d','HTPIT_o','HTPIT_d','SRIT_o','SRIT_d','PPOB','PGN','PIG_o','PIG_d','PIBCa_o','PIBCa_d',
                'PIBI_o','PIBI_d','PIBCo_o','PIBCo_d','PIA_o','PIA_d','ICC','INEC','ICEI','DBNDES','IEG_o','IEG_d','IETIT_o','IETIT_d','IETC_o','IETC_d','IETS_o','IETS_d','IETCV_o','IETCV_d', 'PO','TD','BM','PME',
                'TEMPaj_BS_NE_9','TEMPaj_BS_NE_15','TEMPaj_BS_NE_21','TEMPaj_BS_NE_MED','TEMPaj_BU_N_MED','TEMPaj_BU_NE_9','TEMPaj_BU_NE_15','TEMPaj_BU_NE_21','TEMPaj_BU_NE_MED'] 


    coef_ne = pd.DataFrame(coef2_ne, index=colunas_ne2)
    coef_ne.columns = ['OLS']
    coef_ne['LinearRegression']=coefreg_ne
    coef_ne['Lasso']=coef3_ne
    coef_ne['LassoCV']=coef4_ne
    coef_ne['Lars']=coef5_ne
    coef_ne['LassoLars']=coef6_ne
    coef_ne['LassoLarsCV']=coef7_ne
    coef_ne['Ridge']=coef8_ne
    coef_ne['ElasticNet']=coef90_ne    
    coef_ne['ElasticNetCV']=coef9_ne
    coef_ne['RandomForest']=coef10_ne
    
    
    
    
    R2_ne_list       = [R2_ne_AR,R2_ne2,R2_nereg, R2_ne3,R2_ne4,R2_ne5,R2_ne6,R2_ne7,R2_ne8,R2_ne90,R2_ne9,R2_ne10] 
    EQM_ne_list      = [EQM_ne,EQM_ne2,EQM_nereg, EQM_ne3,EQM_ne4,EQM_ne5,EQM_ne6,EQM_ne7,EQM_ne8,EQM_ne90,EQM_ne9,EQM_ne10]
    resid_ne_list    = [resid_ne,resid_ne2,resid_nereg, resid_ne3,resid_ne4,resid_ne5,resid_ne6,resid_ne7,
                          resid_ne8,resid_ne90,resid_ne9,resid_ne10]
    accuracy_ne_list = [accuracy_ne_AR_ne,accuracy_ne_2,accuracy_ne_reg,accuracy_ne_3,accuracy_ne_4,accuracy_ne_5,
                          accuracy_ne_6,accuracy_ne_7,accuracy_ne_8,accuracy_ne_90,accuracy_ne_9,accuracy_ne_10]
    R2_ne_test_list  = [R2_ne_AR_tese,R2_ne_2_teste,R2_ne_reg_teste, R2_ne_3_teste,R2_ne_4_teste,R2_ne_5_teste,
                          R2_ne_6_teste,R2_ne_7_teste,R2_ne_8_teste,R2_ne_90_teste,R2_ne_9_teste,R2_ne_10_teste]   
    
    
    
    index=['R2_ne', 'EQM_ne', 'Resíduo_ne','accuracy_ne','R2_ne teste']
    
    colunas3 = ['AR','OLS','LinearRegression','Lasso','LassoCV','Lars','LassoLars',
                'LassoLarsCV','Ridge','ElasticNet','ElasticNetCV','RandomForest']
    

    
    previsao_ne = pd.DataFrame([R2_ne_list, EQM_ne_list, resid_ne_list, accuracy_ne_list,
                             R2_ne_test_list],index=index, columns=colunas3)                  
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      #definindo as matrizes X e y
    
   import patsy as ps
   
   y_n = dados['CEE_N_TOT']




                    #Região Sudeste/CO
    
     y_n_treino,X_n_treino = ps.dmatrices('CEE_N_TOT ~ DM+DS	+MÊS+ANO+ESTAC+FER + NEB_N_9 +\
                                    NEB_N_15 + NEB_N_21 + NEB_N_MED+PA_N_9+PA_N_15+PA_N_21+PA_N_MED+TEMP_BS_N_9	+ TEMP_BS_N_15 + \
                                    TEMP_BS_N_21+TEMP_BS_N_MED	+TEMP_BU_N_9	+TEMP_BU_N_15+TEMP_BU_N_21+TEMP_BU_N_MED	+UMID_N_9 + \
                                    UMID_N_15 + UMID_N_21 + UMID_N_MED	+ DV_N_9 + DV_N_15 + DV_N_21	+ DV_N_MED + VV_N_9	+ VV_N_15 + \
                                    VV_N_21	+VV_N_MED+TAR_N_CSO+TAR_N_CP+TAR_N_IP+TAR_N_IND+TAR_N_PP	+TAR_N_RED+TAR_N_RR + \
                                    TAR_N_RRA	+TAR_N_RRI	+TAR_N_SP1	+TAR_N_SP2	+TAR_N_MED	+Meta_Selic+ Taxa_Selic+CDI+DolarC+DolarC_var + \
                                    DolarV + DolarV_var+EuroC + EuroC_var+EuroV + EuroV_var	+ IBV_Cot	+IBV_min	+ IBV_max	+IBV_varabs	+ \
                                    IBV_varperc	+ IBV_vol	+INPC_m+INPC_ac+IPCA_m+IPCA_ac + IPAM_m + IPAM_ac + IPADI_m + IPADI_ac	+ \
                                    IGPM_m+IGPM_ac + IGPDI_m	+IGPDI_ac	+PAB_o	+PAB_d	+ TVP_o	+ TVP_d	+PICV_o	+ICV_d	+CCU_o + \
                                    CCU_d + CS_o + CS_d+UCPIIT_FGV_o+UCPIIT_FGV_d+CPCIIT_CNI_o+CPCIIT_CNI_d+VIR_o	+VIR_d	+HTPIT_o	+ \
                                    HTPIT_d	+ SRIT_o	+SRIT_d	+PPOB	+PGN	+PIG_o	+PIG_d	+PIBCa_o+PIBCa_d + PIBI_o+PIBI_d+PIBCo_o + \
                                    PIBCo_d + PIA_o	+PIA_d+ICC+INEC+ICEI+DBNDES	+IEG_o	+IEG_d	+IETIT_o	+IETIT_d	+IETC_o	+ \
                                    IETC_d + IETS_o	+IETS_d	+IETCV_o+IETCV_d + PO+TD+BM + PME + \
                                    TEMPaj_BS_N_9	+ TEMPaj_BS_N_15 + TEMPaj_BS_N_21+TEMPaj_BS_N_MED	+TEMPaj_BU_N_9	+TEMPaj_BU_N_15+TEMPaj_BU_N_21+TEMPaj_BU_N_MED ', 
                                    data=dados_treino, return_type='dataframe')

                                    #X_N_treino = X_N_treino.drop(['Intercept'], axis=1)

    y_n_teste,X_n_teste = ps.dmatrices('CEE_N_TOT ~ DM+DS	+MÊS+ANO+ESTAC+FER + NEB_N_9 +\
                                    NEB_N_15 + NEB_N_21 + NEB_N_MED+PA_N_9+PA_N_15+PA_N_21+PA_N_MED+TEMP_BS_N_9	+ TEMP_BS_N_15 + \
                                    TEMP_BS_N_21+TEMP_BS_N_MED	+TEMP_BU_N_9	+TEMP_BU_N_15+TEMP_BU_N_21+TEMP_BU_N_MED	+UMID_N_9 + \
                                    UMID_N_15 + UMID_N_21 + UMID_N_MED	+ DV_N_9 + DV_N_15 + DV_N_21	+ DV_N_MED + VV_N_9	+ VV_N_15 + \
                                    VV_N_21	+VV_N_MED+TAR_N_CSO+TAR_N_CP+TAR_N_IP+TAR_N_IND+TAR_N_PP	+TAR_N_RED+TAR_N_RR + \
                                    TAR_N_RRA	+TAR_N_RRI	+TAR_N_SP1	+TAR_N_SP2	+TAR_N_MED	+Meta_Selic+ Taxa_Selic+CDI+DolarC+DolarC_var + \
                                    DolarV + DolarV_var+EuroC + EuroC_var+EuroV + EuroV_var	+ IBV_Cot	+IBV_min	+ IBV_max	+IBV_varabs	+ \
                                    IBV_varperc	+ IBV_vol	+INPC_m+INPC_ac+IPCA_m+IPCA_ac + IPAM_m + IPAM_ac + IPADI_m + IPADI_ac	+ \
                                    IGPM_m+IGPM_ac + IGPDI_m	+IGPDI_ac	+PAB_o	+PAB_d	+ TVP_o	+ TVP_d	+PICV_o	+ICV_d	+CCU_o + \
                                    CCU_d + CS_o + CS_d+UCPIIT_FGV_o+UCPIIT_FGV_d+CPCIIT_CNI_o+CPCIIT_CNI_d+VIR_o	+VIR_d	+HTPIT_o	+ \
                                    HTPIT_d	+ SRIT_o	+SRIT_d	+PPOB	+PGN	+PIG_o	+PIG_d	+PIBCa_o+PIBCa_d + PIBI_o+PIBI_d+PIBCo_o + \
                                    PIBCo_d + PIA_o	+PIA_d+ICC+INEC+ICEI+DBNDES	+IEG_o	+IEG_d	+IETIT_o	+IETIT_d	+IETC_o	+ \
                                    IETC_d + IETS_o	+IETS_d	+IETCV_o+IETCV_d + PO+TD+BM + PME+\
                                    TEMPaj_BS_N_9	+ TEMPaj_BS_N_15 + TEMPaj_BS_N_21+TEMPaj_BS_N_MED	+TEMPaj_BU_N_9	+TEMPaj_BU_N_15+TEMPaj_BU_N_21+TEMPaj_BU_N_MED ', 
                                    data=dados_teste, return_type='dataframe')  
                

    
    #Correlação
    
    from pandas import DataFrame
    from pandas import concat
    
    values_n= DataFrame(dados['CEE_N_TOT'].values)
    dataframe = concat([values_n.shift(1), values_n], axis=1)
    dataframe.columns = ['t-1','t+1']
    result = dataframe.corr()
    print(result)
    
    
    
    
    from plotly.plotly import plot_mpl
    from statsmodels.tsa.seasonal import seasonal_decompose
   
    dec_seas_n = seasonal_decompose(y_n, model='multiplicative')
    fig_n = dec_seas_n.plot()


    #Checando estacionariedade da série de consumo de EE    
    
    from statsmodels.tsa.stattools import adfuller, kpss


    def adf_test(y):
        # perform Augmented Dickey Fuller test
        print('Results of Augmented Dickey-Fuller test:')
        dftest = adfuller(y, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value ({})'.format(key)] = value
        print(dfoutput)
       

     data_n = y_n_treino.iloc[:,0].values
     data2_n = y_n_teste.iloc[:,0].values
     y_n= dados['CEE_N_TOT']
     
     adf_test(data_n)
     adf_test(data2_n)
     adf_test(y_n)
    
    
    #Não podemos rejeitas a hipótese nula a 10%, então a série é não estacionária (série toda e separada).

       
    
    #1st difference
    y_n_treino_diff = np.diff(data)
    ts_diagnostics(y_n_treino_diff, lags=30, title='International Airline Passengers diff', filename='adf_diff')
    adf_test(y_n_treino_diff)     
    
    y_n_teste_diff=np.diff(data2)
    adf_test(y_n_teste_diff)     

    #1a diferença é estacionária
    
     
                                            #AR


    from statsmodels.tsa.ar_model import AR
    from statsmodels.tsa.arima_model import ARIMA
    from sklearn.metrics import mean_squared_error

    from sklearn.metrics import accuracy_score




    #Modelo 1 (Sem tirar a primeira diferenca)
    
    model_n = AR(y_n_treino)                                #modelo
    model_n_fit = model_n.fit()                             #lags
    print('Lag: %s' % model_n_fit.k_ar)
    print('Coefficients: %s' % model_n_fit.params)        #coeficientes
    
    R2_n_AR=0 
    accuracy_n_AR_n=0
    R2_n_AR_tese=0


    # make predictions
    y_n_predictions = model_n_fit.predict(start=len(y_n_treino), end=len(y_n_treino)+len(y_n_teste)-1, dynamic=False)

    EQM_n = mean_squared_error(y_n_teste, y_n_predictions)
    resid_n=np.sqrt(EQM_n)
    print('Test MSE: %.3f' % EQM_n, resid_n)
    

 """   
           #outro modelo arima (consegue decidir o numero de lags - 20 no caso)
    model_n1 = ARIMA(y_n_treino, order=(17,0,0))
    model_n_fit1 = model_n1.fit()
    print('Lag: %s' % model_n_fit1.k_ar)
    print('Coefficients: %s' % model_n_fit1.params)
    # make predictions
    y_n_predictions1 = model_n_fit1.predict(start=len(y_n_treino), end=len(y_n_treino)+len(y_n_teste)-1, dynamic=False)
    EQM_n1 = mean_squared_error(y_n_teste, y_n_predictions1)
    resid_n1 = np.sqrt(EQM_n1)
    print('Test MSE: %.3f' % EQM_n1, resid_n1)
    print(model_n_fit1.summary())
"""


    
    # plot results
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    
    plt.figure()    
    pyplot.plot(y_n_treino, label='Treino')
    pyplot.plot(y_n_teste, color='black', label='Teste')
    pyplot.plot(y_n_predictions, color='red', label='Previsão')
    #dados['CEE_n_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (AR)')
    plt.grid()
    pyplot.show()

    
 """   
    #Modelo 2 (Tirando a primeira diferenca) - Não consegui, deve faltar detalhe
    
    #1)
    y_diff=np.diff(y)
    periodo_diff = pd.date_range('2/1/2017', periods=545)
    y_diff = pd.DataFrame(y_diff,index=periodo_diff)
    y_n_treino_diff, y_n_teste_diff = y_diff[0:train_size], y_diff[train_size:len(dados)]
    
    #2)
    treino_diff = pd.date_range('2/1/2017', periods=364)
    teste_diff = pd.date_range('1/31/2018', periods=180)
    y_n_treino_diff = pd.DataFrame(y_n_treino_diff,index=treino_diff)
    y_n_teste_diff = pd.DataFrame(y_n_teste_diff,index=teste_diff)
    
    
    model_ndiff = AR(y_n_treino_diff)                          #modelo
    model_ndiff_fit = model_ndiff.fit()                             #lags
    print('Lag: %s' % model_ndiff_fit.k_ar)
    print('Coefficients: %s' % model_ndiff_fit.params)        #coeficientes

    # make predictions
    y_n_predictions_diff = model_ndiff_fit.predict(start=len(y_n_treino_diff), end=len(y_n_treino_diff)+len(y_n_teste_diff)-1, dynamic=False)

    EQM_n_diff = mean_squared_error(y_n_teste_diff, y_n_predictions_diff)
    resid_n_diff =np.sqrt(EQM_n_diff)

    print('Test MSE: %.3f' % EQM_n_diff, resid_n_diff)
  
    # plot results
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    
    plt.figure()    
    pyplot.plot(y_n_treino_diff, label='Treino')
    pyplot.plot(y_n_teste_diff, color='black', label='Teste')
    pyplot.plot(y_n_predictions_diff, color='red', label='Previsão')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (AR - 1a Diferença')
    plt.grid()
    pyplot.show()

"""


                           #OLS                
    
    import statsmodels.api as sm
    from sklearn.metrics import r2_score



    model_n2 = sm.OLS(y_n_treino,X_n_treino)                  #modelo
    model_n_fit2 = model_n2.fit() 
    print (model_n_fit2.summary())                        #sumário do modelo
    coef2_n=model_n_fit2.params
    
    R2_n2=model_n_fit2.rsquared
    
        
    # make predictions
    y_n_predictions2 = model_n_fit2.predict(X_n_teste)          #previsão

    EQM_n2 = mean_squared_error(y_n_teste, y_n_predictions2)    #EQM_n
    resid_n2 = np.sqrt(EQM_n2)                                #Resíduo

    print('Test MSE, resid_nual: %.3f' % EQM_n2, resid_n2)
    
    
    accuracy_n_2 = r2_score(y_n_teste, y_n_predictions2)
    R2_n_2_teste = sm.OLS(y_n_teste,X_n_teste).fit().rsquared
    print ('accuracy_n, R2_n_teste: %.3f' % accuracy_n_2, R2_n_2_teste)
    
    plt.figure()    
    pyplot.plot(y_n_treino, label='Treino')
    pyplot.plot(y_n_teste, color='black', label='Teste')
    pyplot.plot(y_n_predictions2, color='red', label='Previsão')
    #dados['CEE_n_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (OLS)')
    plt.grid()
    pyplot.show()   
    
    
    
    #2)Linear Regression
  
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    reg = LinearRegression().fit(X_n_treino, y_n_treino)
    print(reg.score(X_n_treino, y_n_treino))                        #R2_n fora da amostra
    print(reg.coef_)                                            #coeficientes
    coefreg_n=np.transpose(reg.coef_)

    R2_nreg=reg.score(X_n_treino, y_n_treino)    

    predictionsreg_n = reg.predict(X_n_teste)
    y_n_predictionsreg= pd.DataFrame(predictionsreg_n, index=teste)   #previsão

    EQM_nreg = mean_squared_error(y_n_teste, y_n_predictionsreg)      #EQM_n
    resid_nreg = np.sqrt(EQM_nreg)                                #resid_nuo
    print('Test MSE, resid_nuo: %.3f' % EQM_nreg,resid_nreg)
    
    accuracy_n_reg = r2_score(y_n_teste, y_n_predictionsreg)
    R2_n_reg_teste = reg.score(X_n_teste, y_n_teste)  
    print ('accuracy_n, R2_n_teste: %.3f' % accuracy_n_reg, R2_n_reg_teste)
    
    
    
    plt.figure()    
    pyplot.plot(y_n_treino, label='Treino')
    pyplot.plot(y_n_teste, color='black', label='Teste')
    pyplot.plot(y_n_predictionsreg, color='red', label='Previsão')
    #dados['CEE_n_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (LinearRegression)')
    plt.grid()
    pyplot.show()   



                                            #Lasso
    
    #1)Lasso normal
    
    from sklearn import linear_model
    
    model_n3 = linear_model.Lasso( alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
    normalize=False, positive=False, precompute=False, random_state=None,
    selection='cyclic', tol=0.0001, warm_start=False)
    model_n_fit3=model_n3.fit(X_n_treino,y_n_treino)
    coef3_n=model_n3.coef_
    
    print(model_n_fit3.coef_)
    print(model_n_fit3.intercept_) 
    print(model_n_fit3.score(X_n_treino,y_n_treino))

    R2_n3 = model_n_fit3.score(X_n_treino,y_n_treino)
    
        # make predictions
    y_n_predictions3 = model_n_fit3.predict(X_n_teste)
    y_n_predictions3= pd.DataFrame(y_n_predictions3, index=teste)   #previsão


    EQM_n3 = mean_squared_error(y_n_teste, y_n_predictions3)
    resid_n3 = np.sqrt(EQM_n3)
    print('Test MSE, resid_nuo: %.3f' % EQM_n3,resid_n3)
    
    accuracy_n_3 = r2_score(y_n_teste, y_n_predictions3)
    R2_n_3_teste = model_n_fit3.score(X_n_teste, y_n_teste)  
    print ('accuracy_n, R2_n_teste: %.3f' % accuracy_n_3, R2_n_3_teste)
    


      
    plt.figure()    
    pyplot.plot(y_n_treino, label='Treino')
    pyplot.plot(y_n_teste, color='black', label='Teste')
    pyplot.plot(y_n_predictions3, color='red', label='Previsão')
    #dados['CEE_n_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lasso)')
    plt.grid()
    pyplot.show() 
    

    
    #2) Lasso CV (DESASTRE)
    
    from sklearn.linear_model import LassoCV

    
    model_n4 = LassoCV(cv=365, random_state=0).fit(X_n_treino, y_n_treino)
    print(model_n4.coef_)
    coef4_n=model_n4.coef_
    
    R2_n4 = model_n4.score(X_n_treino, y_n_treino) 


        # make predictions
    y_n_predictions4 = model_n4.predict(X_n_teste)
    y_n_predictions4= pd.DataFrame(y_n_predictions4, index=teste)   #previsão


    EQM_n4 = mean_squared_error(y_n_teste, y_n_predictions4)
    resid_n4 = np.sqrt(EQM_n4)
    print('Test MSE: %.3f' % EQM_n4,resid_n4)
    
    accuracy_n_4 = r2_score(y_n_teste, y_n_predictions4)
    R2_n_4_teste = model_n4.score(X_n_teste, y_n_teste)  
    print ('accuracy_n, R2_n_teste: %.3f' % accuracy_n_4, R2_n_4_teste)
    
        
    plt.figure()    
    pyplot.plot(y_n_treino, label='Treino')
    pyplot.plot(y_n_teste, color='black', label='Teste')
    pyplot.plot(y_n_predictions4, color='red', label='Previsão')
    #dados['CEE_n_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lasso CV)')
    plt.grid()
    pyplot.show() 
    
    
    
                                        #Lars
                                        
    #1) Lars normal (DESASTRE)
     
    model_n5 = linear_model.Lars(n_nonzero_coefs=6)
    model_n5_fit=model_n5.fit(X_n_treino, y_n_treino)
    print(model_n5_fit.coef_) 
    coef5_n=model_n5_fit.coef_
    
    R2_n5 = model_n5_fit.score(X_n_treino, y_n_treino) 

     
         # make predictions
    y_n_predictions5 = model_n5_fit.predict(X_n_teste)
    y_n_predictions5= pd.DataFrame(y_n_predictions5, index=teste)   #previsão


    EQM_n5 = mean_squared_error(y_n_teste, y_n_predictions5)
    resid_n5 = np.sqrt(EQM_n5)
    print('Test MSE: %.3f' % EQM_n5,resid_n5)
     
    accuracy_n_5 = r2_score(y_n_teste, y_n_predictions5)
    R2_n_5_teste = model_n5_fit.score(X_n_teste, y_n_teste)  
    print ('accuracy_n, R2_n_teste: %.3f' % accuracy_n_5, R2_n_5_teste)
       
        
    plt.figure()    
    pyplot.plot(y_n_treino, label='Treino')
    pyplot.plot(y_n_teste, color='black', label='Teste')
    pyplot.plot(y_n_predictions5, color='red', label='Previsão')
    #dados['CEE_n_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lars)')
    plt.grid()
    pyplot.show() 
     
    
    #2) Lasso Lars (3 MENOR EQM_n)
    model_n6 = linear_model.LassoLars(alpha=0.01).fit(X_n_treino,y_n_treino)
    print(model_n6.coef_) 
    coef6_n=model_n6.coef_
    
    R2_n6 = model_n6.score(X_n_treino, y_n_treino) 

    
    y_n_predictions6 = model_n6.predict(X_n_teste)
    y_n_predictions6= pd.DataFrame(y_n_predictions6, index=teste)   #previsão


    EQM_n6 = mean_squared_error(y_n_teste, y_n_predictions6)
    resid_n6 = np.sqrt(EQM_n6)
    print('Test MSE: %.3f' % EQM_n6,resid_n6)
    
    accuracy_n_6 = r2_score(y_n_teste, y_n_predictions6)
    R2_n_6_teste = model_n6.score(X_n_teste, y_n_teste)  
    print ('accuracy_n, R2_n_teste: %.3f' % accuracy_n_6, R2_n_6_teste)
    
        
    plt.figure()    
    pyplot.plot(y_n_treino, label='Treino')
    pyplot.plot(y_n_teste, color='black', label='Teste')
    pyplot.plot(y_n_predictions6, color='red', label='Previsão')
    #dados['CEE_n_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (lasso Lars)')
    plt.grid()
    pyplot.show() 
    
    
    
    #3) Lasso Lars com Cross Validation (2 menor EQM_n)
    
    model_n7 = linear_model.LassoLarsCV(cv=50).fit(X_n_treino,y_n_treino)
    print(model_n7.coef_)
    
    coef7_n=model_n7.coef_

    
    R2_n7 = model_n7.score(X_n_treino, y_n_treino) 

    
    y_n_predictions7 = model_n7.predict(X_n_teste)
    y_n_predictions7= pd.DataFrame(y_n_predictions7, index=teste)   #previsão

    EQM_n7 = mean_squared_error(y_n_teste, y_n_predictions7)
    resid_n7 = np.sqrt(EQM_n7)
    print('Test MSE: %.3f' % EQM_n7,resid_n7)
    
    accuracy_n_7 = r2_score(y_n_teste, y_n_predictions7)
    R2_n_7_teste = model_n7.score(X_n_teste, y_n_teste)  
    print ('accuracy_n, R2_n_teste: %.3f' % accuracy_n_7, R2_n_7_teste)
    
        
    plt.figure()    
    pyplot.plot(y_n_treino, label='Treino')
    pyplot.plot(y_n_teste, color='black', label='Teste')
    pyplot.plot(y_n_predictions7, color='red', label='Previsão')
    #dados['CEE_n_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Lasso Lars CV)')
    plt.grid()
    pyplot.show() 
    
    
     
                                #Ridge Regression (MENOR EQM_n)
    
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
    
    model_n8 = Ridge(alpha=0.1,normalize=True)
    model_n8_fit=model_n8.fit(X_n_treino, y_n_treino)
    
    coef8_n=np.transpose(model_n8_fit.coef_)

        
    R2_n8 = model_n8_fit.score(X_n_treino, y_n_treino) 

    
    y_n_predictions8 = model_n8_fit.predict(X_n_teste)
    y_n_predictions8= pd.DataFrame(y_n_predictions8, index=teste)   #previsão

    EQM_n8 = mean_squared_error(y_n_teste, y_n_predictions8)
    resid_n8 = np.sqrt(EQM_n8)
    print('Test MSE: %.3f' % EQM_n8,resid_n8)
    
    
    accuracy_n_8 = r2_score(y_n_teste, y_n_predictions8)
    R2_n_8_teste = model_n8_fit.score(X_n_teste, y_n_teste)  
    print ('accuracy_n, R2_n_teste: %.3f' % accuracy_n_8, R2_n_8_teste)
    
        
    plt.figure()    
    pyplot.plot(y_n_treino, label='Treino')
    pyplot.plot(y_n_teste, color='black', label='Teste')
    pyplot.plot(y_n_predictions8, color='red', label='Previsão')
    #dados['CEE_n_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Ridge)')
    plt.grid()
    pyplot.show() 
    

    
                                #ElasticNet (4 MENOR EQM_n)
                                
    #1) ElasticNet 
    
      from sklearn.linear_model import ElasticNet

    
    model_n90 = ElasticNet().fit(X_n_treino,y_n_treino)
    print(model_n90.coef_) 

    R2_n90 = model_n90.score(X_n_treino, y_n_treino) 
    coef90_n=model_n90.coef_
    
    y_n_predictions90 = model_n90.predict(X_n_teste)
    y_n_predictions90= pd.DataFrame(y_n_predictions90, index=teste)   #previsão

    EQM_n90 = mean_squared_error(y_n_teste, y_n_predictions90)
    resid_n90 = np.sqrt(EQM_n90)
    print('Test MSE: %.3f' % EQM_n90,resid_n90)
    
    accuracy_n_90 = r2_score(y_n_teste, y_n_predictions90)
    R2_n_90_teste = model_n90.score(X_n_teste, y_n_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_n_90, R2_n_90_teste)
        
    plt.figure()    
    pyplot.plot(y_n_treino, label='Treino')
    pyplot.plot(y_n_teste, color='black', label='Teste')
    pyplot.plot(y_n_predictions90, color='red', label='Previsão')
    #dados['CEE_BR_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (ElasticNet)')
    plt.grid()
    pyplot.show() 
    
    
    
    
    
    
    
    
    #2) ElasticNetCV
    
        from sklearn.linear_model import ElasticNetCV

    model_n9 = ElasticNetCV(alphas=None, copy_X=True, cv=20, eps=0.001, fit_intercept=True,
       l1_ratio=0.5, max_iter=1000, n_alphas=100, n_jobs=None,
       normalize=False, positive=False, precompute='auto', random_state=0,
       selection='cyclic', tol=0.0001, verbose=0).fit(X_n_treino,y_n_treino)
    print(model_n9.coef_) 

    R2_n9 = model_n9.score(X_n_treino, y_n_treino) 
    coef9_n=model_n9.coef_
    
    y_n_predictions9 = model_n9.predict(X_n_teste)
    y_n_predictions9= pd.DataFrame(y_n_predictions9, index=teste)   #previsão

    EQM_n9 = mean_squared_error(y_n_teste, y_n_predictions9)
    resid_n9 = np.sqrt(EQM_n9)
    print('Test MSE: %.3f' % EQM_n9,resid_n9)
    
    accuracy_n_9 = r2_score(y_n_teste, y_n_predictions9)
    R2_n_9_teste = model_n9.score(X_n_teste, y_n_teste)  
    print ('accuracy_n, R2_n_teste: %.3f' % accuracy_n_9, R2_n_9_teste)
        
    plt.figure()    
    pyplot.plot(y_n_treino, label='Treino')
    pyplot.plot(y_n_teste, color='black', label='Teste')
    pyplot.plot(y_n_predictions9, color='red', label='Previsão')
    #dados['CEE_n_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (ElasticNetCV')
    plt.grid()
    pyplot.show() 
    
                                    #Random Forest  (MELHOR EQM_n)

    
    from sklearn.ensemble import RandomForestRegressor
    
    model_n10 = RandomForestRegressor(n_estimators = 1000, random_state = 0).fit(X_n_treino, y_n_treino)
    
    print(model_n10.feature_importances_)
    coef10_n=model_n10.feature_importances_
    
    
    R2_n10 = model_n10.score(X_n_treino, y_n_treino) 
    
    y_n_predictions10 = model_n10.predict(X_n_teste)
    y_n_predictions10= pd.DataFrame(y_n_predictions10, index=teste)   #previsão

    EQM_n10 = mean_squared_error(y_n_teste, y_n_predictions10)
    resid_n10 = np.sqrt(EQM_n10)
    print('Test MSE: %.3f' % EQM_n10,resid_n10)
    
    
    accuracy_n_10 = r2_score(y_n_teste, y_n_predictions10)
    R2_n_10_teste = model_n10.score(X_n_teste, y_n_teste)  
    print ('accuracy_n, R2_n_teste: %.3f' % accuracy_n_10, R2_n_10_teste)
    
        
    plt.figure()    
    pyplot.plot(y_n_treino, label='Treino')
    pyplot.plot(y_n_teste, color='black', label='Teste')
    pyplot.plot(y_n_predictions10, color='red', label='Previsão')
    #dados['CEE_n_TOT'].plot(color='black', label='Brasil')
    plt.legend(loc='best')
    plt.ylabel('KW/h')
    plt.xticks(rotation=30)
    plt.title('Previsão Consumo de Energia Elétrica (Random Forest)')
    plt.grid()
    pyplot.show() 
    
    

    
     colunas_n2 =  ['DM','DS','MÊS','ANO','ESTAC','FER','NEB_N_9','NEB_N_15','NEB_N_21',
                'NEB_N_MED','PA_N_9','PA_N_15','PA_N_21','PA_N_MED','TEMP_BS_N_9','TEMP_BS_N_15','TEMP_BS_N_21','TEMP_BS_N_MED','TEMP_BU_N_9','TEMP_BU_N_15','TEMP_BU_N_21','TEMP_BU_N_MED','UMID_N_9', 
                'UMID_N_15','UMID_N_21','UMID_N_MED','DV_N_9','DV_N_15','DV_N_21','DV_N_MED','VV_N_9','VV_N_15','VV_N_21','VV_N_MED','TAR_N_CSO','TAR_N_CP','TAR_N_IP','TAR_N_IND','TAR_N_PP','TAR_N_RED',
                'TAR_N_RR','TAR_N_RRA','TAR_N_RRI','TAR_N_SP1','TAR_N_SP2','TAR_N_MED','Meta_Selic', 'Taxa_Selic','CDI','DolarC','DolarC_var','DolarV','DolarV_var','EuroC','EuroC_var','EuroV','EuroV_var','IBV_Cot',
                'IBV_min','IBV_max','IBV_varabs','IBV_varperc','IBV_vol','INPC_m','INPC_ac','IPCA_m','IPCA_ac','IPAM_m','IPAM_ac','IPADI_m', 'IPADI_ac' , 'IGPM_m','IGPM_ac','IGPDI_m','IGPDI_ac','PAB_o','PAB_d',
                'TVP_o','TVP_d','PICV_o','ICV_d','CCU_o','CCU_d','CS_o','CS_d','UCPIIT_FGV_o','UCPIIT_FGV_d','CPCIIT_CNI_o','CPCIIT_CNI_d','VIR_o','VIR_d','HTPIT_o','HTPIT_d','SRIT_o','SRIT_d','PPOB','PGN','PIG_o','PIG_d','PIBCa_o','PIBCa_d',
                'PIBI_o','PIBI_d','PIBCo_o','PIBCo_d','PIA_o','PIA_d','ICC','INEC','ICEI','DBNDES','IEG_o','IEG_d','IETIT_o','IETIT_d','IETC_o','IETC_d','IETS_o','IETS_d','IETCV_o','IETCV_d', 'PO','TD','BM','PME',
                'TEMPaj_BS_N_9','TEMPaj_BS_N_15','TEMPaj_BS_N_21','TEMPaj_BS_N_MED','TEMPaj_BU_N_MED','TEMPaj_BU_N_9','TEMPaj_BU_N_15','TEMPaj_BU_N_21','TEMPaj_BU_N_MED'] 


    coef_n = pd.DataFrame(coef2_n, index=colunas_n2)
    coef_n.columns = ['OLS']
    coef_n['LinearRegression']=coefreg_n
    coef_n['Lasso']=coef3_n
    coef_n['LassoCV']=coef4_n
    coef_n['Lars']=coef5_n
    coef_n['LassoLars']=coef6_n
    coef_n['LassoLarsCV']=coef7_n
    coef_n['Ridge']=coef8_n
    coef_n['ElasticNet']=coef90_n    
    coef_n['ElasticNetCV']=coef9_n
    coef_n['RandomForest']=coef10_n
    
    
    
    
    R2_n_list       = [R2_n_AR,R2_n2,R2_nreg, R2_n3,R2_n4,R2_n5,R2_n6,R2_n7,R2_n8,R2_n90,R2_n9,R2_n10] 
    EQM_n_list      = [EQM_n,EQM_n2,EQM_nreg, EQM_n3,EQM_n4,EQM_n5,EQM_n6,EQM_n7,EQM_n8,EQM_n90,EQM_n9,EQM_n10]
    resid_n_list    = [resid_n,resid_n2,resid_nreg, resid_n3,resid_n4,resid_n5,resid_n6,resid_n7,
                          resid_n8,resid_n90,resid_n9,resid_n10]
    accuracy_n_list = [accuracy_n_AR_n,accuracy_n_2,accuracy_n_reg,accuracy_n_3,accuracy_n_4,accuracy_n_5,
                          accuracy_n_6,accuracy_n_7,accuracy_n_8,accuracy_n_90,accuracy_n_9,accuracy_n_10]
    R2_n_test_list  = [R2_n_AR_tese,R2_n_2_teste,R2_n_reg_teste, R2_n_3_teste,R2_n_4_teste,R2_n_5_teste,
                          R2_n_6_teste,R2_n_7_teste,R2_n_8_teste,R2_n_90_teste,R2_n_9_teste,R2_n_10_teste]   
    
    
    
    index=['R2_n', 'EQM_n', 'Resíduo_n','accuracy_n','R2_n teste']
    
    colunas3 = ['AR','OLS','LinearRegression','Lasso','LassoCV','Lars','LassoLars',
                'LassoLarsCV','Ridge','ElasticNet','ElasticNetCV','RandomForest']
    

    
    previsao_n = pd.DataFrame([R2_n_list, EQM_n_list, resid_n_list, accuracy_n_list,
                             R2_n_test_list],index=index, columns=colunas3)                  
      