#Residuo = Serie_Original - SVR(Serie_Original)
#Previsão_Final = AR(Serie_Original) + AR(Resíduo)

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt


def corr_factor_plot(data, lags):
  values = []
  for i in range((lags)):
    values.append(data.autocorr(i))
  plt.clf()
  plt.plot(np.zeros_like(values))
  plt.plot(np.ones_like(values)*2*(1/np.sqrt(len(data))))
  plt.plot(np.ones_like(values)*(-2)*(1/np.sqrt(len(data))))
  plt.bar(range(lags), values, width=0.4)
  plt.title('Gráfico de autocorrelação')
  plt.xlabel('Lags')
  plt.ylabel('Autocorrelação')
  plt.show()

trainLimit = 0.6
validateLimit = 0.2
testLimit = 0.2

dataset = pd.read_excel('gold-2000.xlsx')
dataset = dataset.iloc[:,1]

#setar tamanhos
trainSize = int(np.floor(trainLimit*len(dataset)))
validateSize = int(np.floor(validateLimit*len(dataset)))


#normalizar a base
maxData = np.max(dataset[0:trainSize])
minData = np.min(dataset[0:trainSize])
normalized_dataset = (dataset - minData)/(maxData - minData)

#plotando a correlação
dataSeries = pd.Series(normalized_dataset)
corr_factor_plot(dataSeries, 10)



dimension = 20
stepAhead = 1
bestValue = 100**100;
model = 0
erroBar = []

for d in range(1, dimension+stepAhead, 1):

  shiftedData = pd.concat([dataSeries.shift(i) for i in range(d+stepAhead)], axis=1)
  train = shiftedData.iloc[d:trainSize, 1:]
  train = np.hstack((train, np.ones_like(train.iloc[:,0].values.reshape(len(train),1))))
  trainTarget = shiftedData.iloc[d:trainSize, 0]

  valid = shiftedData.iloc[trainSize:(validateSize+trainSize), 1:]
  valid = np.hstack((valid, np.zeros_like(valid.iloc[:,0].values.reshape(len(valid),1))))
  validTarget = shiftedData.iloc[trainSize:(validateSize+trainSize), 0]

  test = shiftedData.iloc[(validateSize+trainSize):, 1:]
  test = np.hstack((test, np.ones_like(test.iloc[:,0].values.reshape(len(test),1))))
  testTarget = shiftedData.iloc[(validateSize+trainSize):, 0]

  X_inv = np.linalg.pinv(train)
  coefficients = X_inv.dot(trainTarget)

  #validando o modelo
  predictedValues = coefficients.dot(valid.T)
  erro = mse(predictedValues, validTarget)

  erroBar.append(erro)
  if(erro<bestValue):
    bestValue = erro

#validação por dimensão, menor mean squared error
plt.clf()
plt.bar(range(1, dimension+stepAhead, 1), erroBar)
plt.xlabel('Dimensão da série temporal')
plt.ylabel('MSE Validation')
plt.show()

#aplicando a previsão ao teste
predTest = coefficients.dot(test.T)
mse_test_series = mse(predTest, testTarget)
print('o erro da previsão da série original é ',mse_test_series)

plt.clf()
plt.plot(predTest)
plt.plot(testTarget.values)
plt.legend(['Previsões da série temporal', 'Valor Real'])
plt.show()


###########################################################
#Residuo = Serie_Original - LIN(Serie_Original)

length = [i for i in range(len(predTest))]
predTest = pd.Series(predTest, index=length)

testTarget.reset_index(drop=True)
residuo = []
length = len(testTarget)

for (index, value) in enumerate(testTarget):
  residuo.append(value - predTest[index])
residuo = pd.Series(residuo)

trainLimit = 0.6
validateLimit = 0.2
testLimit = 0.2

#setar tamanhos
residueTrainSize = int(np.floor(trainLimit*len(residuo)))
residueValidateSize = int(np.floor(validateLimit*len(residuo)))


#normalizar a base
residueMaxData = np.max(residuo[0:trainSize])
residueMinData = np.min(residuo[0:trainSize])
normalizedResidueDataset = (residuo - minData)/(maxData - minData)

#plotando a correlação
residueDataSeries = pd.Series(normalizedResidueDataset)
corr_factor_plot(residueDataSeries, 20)


residueDimension = 10
bestResidueValue = 100**10
erroResidueBar = []

for d in range(1, residueDimension+stepAhead, 1):
  shiftedResidueData = pd.concat([residueDataSeries.shift(i) for i in range(d+stepAhead)], axis=1)
  residueTrain = shiftedResidueData.iloc[d:residueTrainSize, 1:]
  residueTrain = np.hstack((residueTrain, np.ones_like(residueTrain.iloc[:,0].values.reshape(len(residueTrain),1))))
  residueTrainTarget = shiftedResidueData.iloc[d:residueTrainSize, 0]

  residueValid = shiftedResidueData.iloc[residueTrainSize:(residueValidateSize+residueTrainSize), 1:]
  residueValid = np.hstack((residueValid, np.zeros_like(residueValid.iloc[:,0].values.reshape(len(residueValid),1))))
  residueValidTarget = shiftedResidueData.iloc[residueTrainSize:(residueValidateSize+residueTrainSize), 0]

  residueTest = shiftedResidueData.iloc[(residueValidateSize+residueTrainSize):, 1:]
  residueTest = np.hstack((residueTest, np.ones_like(residueTest.iloc[:,0].values.reshape(len(residueTest),1))))
  residueTestTarget = shiftedResidueData.iloc[(residueValidateSize+residueTrainSize):, 0]

  X_inv = np.linalg.pinv(residueTrain)
  residueCoefficients = X_inv.dot(residueTrainTarget)

  #validando o modelo
  residuePredictedValues = residueCoefficients.dot(residueValid.T)
  residueError = mse(residuePredictedValues, residueValidTarget)

  erroResidueBar.append(residueError)
  if(residueError < bestResidueValue):
    bestResidueValue = residueError


#validação por dimensão, menor mean squared error
plt.clf()
plt.bar(range(1, residueDimension+stepAhead, 1), erroResidueBar)
plt.xlabel('Dimensão')
plt.ylabel('MSE Validation')
plt.show()

#aplicando a previsão ao teste
residuePredTest = residueCoefficients.dot(residueTest.T)
mse_test_residuo = mse(residuePredTest, residueTestTarget)
print('o erro do resíduo é ',mse_test_residuo)

plt.clf()
plt.plot(residuePredTest)
plt.plot(residueTestTarget.values)
plt.legend(['Previsões do resíduo', 'Valor Real'])
plt.show()



##############################################################################
#Previsão_Final = AR(Serie_Original) + AR(Resíduo)

predTest = predTest.iloc[(len(predTest)-len(residuePredTest)):]

previsaoFinal = []


for (index, value) in enumerate(predTest):
  previsaoFinal.append((value + residuePredTest[index]**2))
previsaoFinal = pd.Series(previsaoFinal)

finalTarget = testTarget.iloc[(len(testTarget)-len(residuePredTest)):]

plt.clf()
plt.plot(previsaoFinal)
plt.plot(finalTarget.values)
plt.ylim(0,1.5)
plt.legend(['Previsão Final', 'Valor Real'])
plt.show()

mse_test_final = mse(previsaoFinal, finalTarget.values)
print('o erro da previsão final é ', mse_test_final)
