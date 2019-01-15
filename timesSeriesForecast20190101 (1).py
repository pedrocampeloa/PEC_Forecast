# -*- coding: utf-8 -*-
"""
Created on Nov  5 22:52:48 2018

@author: daniel
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model

def replaceNanWithMean(a):
    col_mean = np.nanmean(a, axis=0)
    inds = np.where(np.isnan(a))
    a[inds] = np.take(col_mean, inds[1])
    return a 

def evalSquareAverage(x):
    #print(np.shape(x)[0])
    return np.sum(np.power(x,2))/(1.0*np.shape(x)[0])


def modelEvaluation(model,X, y):
    model.fit(X,y)
    #print(model.coef_)
    yHat=model.predict(X)    
    
    error=y-yHat

    return evalSquareAverage(error)


def generateDataframe(independentVariables,independentVariablesLags,uselessVariables,timeSeriesSize,sigma,seed):
    #independentVariables: Lista com os nomes das variaveis independentes
    #independentVariablesLags: Lista ordenada com os lags de cada variavel
    #uselessVariables: Lista com o nome das variaveis inuteis
    #timeSeriesSize: Inteiro com a dimensão da serie de tempo
    #sigma: Desvio padrao do erro da variavel dependente. Para as variaveis independent e useless foi assumido 
    #constante para todas as series - uma extensao simples eh incluir desvio padroes diferentes e explorar como isso afeta os algoritmos de machine learning)
    #seed: Fixa a semente 
    np.random.seed(seed)
    df = pd.DataFrame(index=range(timeSeriesSize))
    for var in uselessVariables:
        df[var]=np.random.randn(timeSeriesSize,1)
    for i,var in enumerate(independentVariables):
        theVar=np.random.randn(timeSeriesSize+independentVariablesLags[i],1)
        df[var]=theVar[independentVariablesLags[i]:timeSeriesSize+independentVariablesLags[i]]
        for lag in range(1,independentVariablesLags[i]+1):
            df[var+'lag'+str(lag)]=theVar[independentVariablesLags[i]-lag:timeSeriesSize+independentVariablesLags[i]-lag]

    
    count=0
    if(constant):
        df['constant']=1
        y=model[count]*np.ones([timeSeriesSize,1])+sigma*np.random.randn(timeSeriesSize,1)
        count=count+1
    else:    
        y=np.zeros(timeSeriesSize,1)+sigma*np.random.randn(timeSeriesSize,1)
    
    for i,var in enumerate(independentVariables):
        for lag in range(1,independentVariablesLags[i]+1):
            y=y+model[count]*df.as_matrix(columns=[var+'lag'+str(lag)])
            df.drop(columns=[var+'lag'+str(lag)],inplace=True)
            count=count+1
    df['y']=y

        
    return df    

def timeSeriesPreparation(df,varDict,constantName,dropNaN,forecastHorizon):
    # df: Dataframe real de dados com variaveis organizadas em colunas
    # varDict: Dicionario com o max lag das variaveis envolvidas
    # constantName: Nome da constante. Se '', o modelo eh sem constante
    # dropNaN: True ou False: Deleta NaN. Cuidado pode deletar linhas da serie e perder consistencia dos dados
    # ForecastHorizon: Horizonte de previsao
    #OBS: As datas mais antigas estao no inicio e as mais atuais estao no fim da planilha          
    
    varList=[]
    for var in df.columns:
        if ((var in varDict) and (var!='y')):
            #print(var)
            varList.append(var)
    
    if(constantName!=''):
        timeSeries=df[varList+['y']].copy()
        timeSeries[constantName]=1
    else:
        timeSeries=df[varList+['y']].copy()

    #print(timeSeries)
    for var in df.columns:
        if var in varDict:
            #print(var)
            maxLag=varDict[var]+1
            for lag in range(1, maxLag):
                #print(lag)
                timeSeries[var+'lag' + str(lag)] = timeSeries[var].shift(lag+forecastHorizon-1)
    if(dropNaN):
        timeSeries.dropna(inplace=True)   
    return timeSeries     

def generateListOfIndependentVariables(varDict,constantName):
    # varDict: Dicionario com os lags maximos das variaveis
    # constantName: Nome da constante. Se '', o modelo eh sem constante
    independentVariableList=[]
    if(constantName!=''):
        independentVariableList.append(constantName)
    for key,value in varDict.items():
        maxLag=value+1
        for lag in range(1, maxLag):
            independentVariableList.append(key+'lag' + str(lag))
    return independentVariableList

def runModel(timeSeries,listOfIndependentVariables,dependentVariable,normalize,model):
    # timeSeries: dataframe
    # listOfIndependentVariables: Variaveis a serem incluidas que foram geradas em generateListOfIndependentVariables ou manualmente
    # dependentVariable: string com o nome da variavel dependente
    # normalize: True ou False. Normalizacao da matriz X. Lembre "alguns modelos de machine learning precisam ser normalizados"
    # model: sklearn model    
    X=timeSeries.as_matrix(columns=listOfIndependentVariables)
    X=replaceNanWithMean(X)
    y=np.ravel(timeSeries[dependentVariable])
    if(normalize):
        numberOfColumnsOfX=np.shape(X)[0]
        for i in range(numberOfColumnsOfX):
            if(X[:,i].std(0)!=0):
                X[:,i] = ((X[:,i] - X[:,i].mean(0)) / X[:,i].std(0))
    
    mse=modelEvaluation(model,X, y) 
    return mse
        
def estimateModelCoefficients(timeSeries,listOfIndependentVariables,dependentVariable,normalize,model):        
    X=timeSeries.as_matrix(columns=listOfIndependentVariables)
    X=replaceNanWithMean(X)
    y=np.ravel(timeSeries[dependentVariable])
    if(normalize):
        numberOfColumnsOfX=np.shape(X)[0]
        for i in range(numberOfColumnsOfX):
            if(X[:,i].std(0)!=0):
                X[:,i] = ((X[:,i] - X[:,i].mean(0)) / X[:,i].std(0))
    
    model.fit(X,y)
    #mse=modelEvaluation(model,X, y) 
    return model


def testModel(timeSeries,listOfIndependentVariables,dependentVariable,normalize,model):
    X=timeSeries.as_matrix(columns=listOfIndependentVariables)
    X=replaceNanWithMean(X)
    y=np.ravel(timeSeries[dependentVariable])
    if(normalize):
        numberOfColumnsOfX=np.shape(X)[0]
        for i in range(numberOfColumnsOfX):
            if(X[:,i].std(0)!=0):
                X[:,i] = ((X[:,i] - X[:,i].mean(0)) / X[:,i].std(0))
    yHat=model.predict(X)    
    
    error=y-yHat

    return evalSquareAverage(error)
        

if __name__=='__main__':
    
    # Monte Carlo
    independentVariables=['var1','var2','var3']
    independentVariablesLags=[1,2,4]
    uselessVariables=['useless1','useless2','useless3']    
    timeSeriesSize=100
    constant=True
    model=[1,2,3,4,5,6,7,8] #Model parameters should be the constant + number of parameter lags
    sigma=0.1
    seed=0
    
    # Being god
    df=generateDataframe(independentVariables,independentVariablesLags,uselessVariables,timeSeriesSize,sigma,seed)
    #print(df)

    #Generating time series
    varDictTimeSeries={}  
    varDictTimeSeries['y']=2
    varDictTimeSeries['var1']=2
    varDictTimeSeries['var2']=2
    varDictTimeSeries['var3']=4       
    constantName='constant'        
    dropNaN=False
    forecastHorizon=1 
    #ForecastHorizon=1 "significa que vc esta usando dado de hoje para prever o futuro
    #Valores usuais: 1 (um dia),7 (1 semana),30 (1 mes),180
    timeSeries=timeSeriesPreparation(df,varDictTimeSeries,constantName,dropNaNforecastHorizon)
    #print(timeSeries)

    porcentTrainSize=0.2
    trainSize=int(porcentTrainSize*timeSeriesSize)
    testSize=timeSeriesSize-trainSize
    timeSeriesTrain=timeSeries.head(trainSize)
    timeSeriesTest=timeSeries.tail(testSize)

####### TRAIN #####################################################################

    estimationWindow=10 # It must be smaller than the number of available points

    
    # You may have a routine to include the relevant parameters test
    # Carregando todos os possiveis parameters
    dictParameters={}    


    # Parameters Set 1
    varDictModel={}
    varDictModel['var1']=2
    varDictModel['var2']=2
    varDictModel['var3']=2
    
    varDictParameters={}
    varDictParameters['lags']=varDictModel
    varDictParameters['constantName']='constant'
    varDictParameters['lambda']=None
    varDictParameters['allData']=False
    dictParameters['set1']=varDictParameters


    # Parameters set 2
    varDictModel={}
    varDictModel['var1']=1
    varDictModel['var2']=1
    varDictModel['var3']=1
    
    varDictParameters={}
    varDictParameters['lags']=varDictModel
    varDictParameters['constantName']='constant'
    varDictParameters['lambda']=None
    varDictParameters['allData']=False
    dictParameters['set2']=varDictParameters
    
    
    
    
    numberOfAvalableEstimations=trainSize-estimationWindow
    dictError={}
    for name,parameters in dictParameters.items():
        varDictModel=parameters['lags']
        constantName=parameters['constantName']
        allData=parameters['allData']
        dictError[name]=0
        for t in range(numberOfAvalableEstimations):
            
            listOfIndependentVariables=generateListOfIndependentVariables(varDictModel,constantName)
            dependentVariable='y' 
            normalize=False
            model=linear_model.LinearRegression(fit_intercept=False)
            if(allData):
                rowSelection=range(t+estimationWindow)
                theTimeSeries=timeSeriesTest.iloc[list(rowSelection),range(len(list(timeSeries.columns)))]
            else: 
                rowSelection=range(t,t+estimationWindow)
                theTimeSeries=timeSeriesTrain.iloc[list(rowSelection),range(len(list(timeSeries.columns)))]
            estimatedModel=estimateModelCoefficients(theTimeSeries,listOfIndependentVariables,dependentVariable,normalize,model)
            theTimeSeries=timeSeriesTrain.iloc[[t+estimationWindow],range(len(list(timeSeries.columns)))]
            error=testModel(theTimeSeries,listOfIndependentVariables,dependentVariable,normalize,estimatedModel)
            print(error)
            dictError[name]=dictError[name]+error
    
    # Select the parameters with minimal error
    minErrorParametersKey = min(dictError, key=lambda k: dictError[k])
    print(minErrorParametersKey)
    parameters=dictParameters[minErrorParametersKey]
    
###################TEST############################################################    

    varDictModel=parameters['lags']
    constantName=parameters['constantName']
    allData=parameters['allData']
    
    
    numberOfAvalableTests=testSize-estimationWindow
    errorTotal=0
    for t in range(numberOfAvalableTests):
        
        listOfIndependentVariables=generateListOfIndependentVariables(varDictModel,constantName)
        dependentVariable='y' 
        normalize=False
        model=linear_model.LinearRegression(fit_intercept=False)
        if(allData):
            rowSelection=range(t+estimationWindow)
            theTimeSeries=timeSeriesTest.iloc[list(rowSelection),range(len(list(timeSeries.columns)))]
        else: 
            rowSelection=range(t,t+estimationWindow)
            theTimeSeries=timeSeriesTest.iloc[list(rowSelection),range(len(list(timeSeries.columns)))]
        estimatedModel=estimateModelCoefficients(theTimeSeries,listOfIndependentVariables,dependentVariable,normalize,model)
        theTimeSeries=timeSeriesTest.iloc[[t+estimationWindow],range(len(list(timeSeries.columns)))]        
        error=testModel(theTimeSeries,listOfIndependentVariables,dependentVariable,normalize,estimatedModel)
        errorTotal=errorTotal+error
    
    print(errorTotal)    
    
    
    
    