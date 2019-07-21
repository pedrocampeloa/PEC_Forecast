PEC_Forecast

PEC_Forecast is a work that uses machine learning techniques to forecast Brazilian power electricity consumption (PEC) for short and medium terms developed in Python 3+. Its main goal is to show that machine learning models consistently outperform benchmark models.

PEC_forecast.py is the Python script that we use to set our data, generate plots, train the models and test them out of the sample. PEC_plots.py is the Python script that we use to generate some alternate graphics.

dadosconsumo.csv is the csv file that has all data on electricity consumption for the analyzed period. dadosx.csv is the csv file that contains all explanatory variables. PEC_forecast.py file merges the two databases to run our models.

In addition, we used 510 variables to predict the consumption of electricity lagged in 3 different time horizons, generating a total of 1510 variables. The Variables_Index.pdf file explains the meaning of each variable used.
