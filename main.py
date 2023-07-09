import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit import DataStructs

import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Fragments
from rdkit.Chem import rdMolDescriptors
import seaborn as sns


data = pd.read_csv('data.csv')
drug_descriptors = pd.read_csv(r'drug_descriptors.csv')[['drug', 'smiles']]
bacterial_descriptors = pd.read_csv(r'bacterial_descriptors.csv')

data = data.drop(['Unnamed: 0.1',
                  'Drug_dose',
                  'NP_concentration',
                  'fold_increase_in_antibacterial_activity (%)',],
                  axis=1)

data['ZOI_drug'][data.loc[:, 'ZOI_drug'] == '32+'] = 32
data['ZOI_drug'] = data['ZOI_drug'].astype(float)


data = data[~data.loc[:, 'Drug'].isna()]
data = data[~data.loc[:, 'ZOI_drug_NP'].isna()]
non_bacts = []
for i in data.loc[:,'Bacteria']:
   if (i == bacterial_descriptors.loc[:, 'Bacteria']).sum() == 0:
      non_bacts.append([i, (i == bacterial_descriptors.loc[:, 'Bacteria']).sum()])
#Соеденим все в один датасет
data = data.merge(drug_descriptors, how = 'left', left_on='Drug', right_on='drug')
data = data.merge(bacterial_descriptors, how = 'left', left_on='Bacteria', right_on='Bacteria')
#Выбросим слишком пустые и не несущие полезной информации столбы после объединения
data = data.drop(['subkingdom',
                  'clade' ,
                  'kingdom',
                  'phylum',
                  'class',
                  'order',
                  'family',
                  #'genus',
                  'species',
                  'Tax_id',
                  'isolated_from'
                  ],
                  axis=1)

data = data[~data.loc[:, 'smiles'].isna()]
#Заменим по каждому антибиотику пропущенные значения на среднее для этого антибиотика ZOI_drug
for i in np.unique(data[data.loc[:, 'ZOI_drug'] != data.loc[:, 'ZOI_drug']].loc[:, 'Drug']):
  iter = data[data.loc[:, 'Drug'] == i].loc[:, 'ZOI_drug']

  mean = np.mean(data[data.loc[:, 'Drug'] == i].loc[:, 'ZOI_drug'])
  index = (iter[iter.isna()].index)
  if (mean == mean):
    data.loc[index, 'ZOI_drug'] = mean
  else:
    data.loc[index, 'ZOI_drug'] = np.mean(data.loc[:, 'ZOI_drug'])

#Функция заменяет названия на инты и приписывает названия

def names_to_class(data):
  data1 = data
  for i in data1.columns:
    uniqs = np.unique(data1[i])

    print(i)
    print(pd.DataFrame(uniqs))

    counter = -1
    for j in uniqs:
      counter += 1
      data[i][data1[i] == j] = counter

  return data1

corr = data.loc[:, ['method', 'shape', 'NP_Synthesis', 'NP size_min', 'NP size_max', 'NP size_avg', 'ZOI_NP']].dropna(axis=0, how='any')
corr.loc[:, 'ZOI_NP'][corr.loc[:, 'ZOI_NP'] == '50+'] = 50
corr.loc[:, 'ZOI_NP'] = corr.loc[:, 'ZOI_NP'].astype(float)

corr.loc[:, ['method',	'shape',	'NP_Synthesis']] = names_to_class(corr.loc[:, ['method',	'shape',	'NP_Synthesis']])

sum = 0
n = 30
for i in range(n):
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(corr.iloc[:, :-1], corr.iloc[:, -1], test_size=0.3)
  rg = RandomForestRegressor(max_depth=4, n_estimators=100)
  rg.fit(x_train, y_train)
  sum += sklearn.metrics.r2_score(y_test, rg.predict(x_test))
print('R2 =', sum/n)

zoi_np_df = data.loc[:, ['Unnamed: 0', 'method', 'shape', 'NP_Synthesis', 'NP size_min', 'NP size_max', 'NP size_avg', 'ZOI_NP']][data.loc[:, ['Unnamed: 0', 'method', 'shape', 'NP_Synthesis', 'NP size_min', 'NP size_max', 'NP size_avg']][~data.loc[:, ['Unnamed: 0', 'method', 'shape', 'NP_Synthesis', 'NP size_min', 'NP size_max', 'NP size_avg']].isna()][data.loc[:, ['ZOI_NP']].isna()].isna()]
zoi_np_df.loc[:, ['method',	'shape',	'NP_Synthesis']] = names_to_class(zoi_np_df.loc[:, ['method',	'shape',	'NP_Synthesis']])

rg = RandomForestRegressor(max_depth=4, n_estimators=100)
rg.fit(corr.iloc[:, :-1], corr.iloc[:, -1])

zoi_np_df.loc[:, 'ZOI_NP'] = rg.predict(zoi_np_df.iloc[:, 1:-1])

data.index = data.loc[:, 'Unnamed: 0']
zoi_np_df.index = zoi_np_df.loc[:, 'Unnamed: 0']
data.loc[:, 'ZOI_NP'] = zoi_np_df.loc[:, 'ZOI_NP']

#Все пропущенные бактерии грамотрицательные <<ссылка>>
ind = data[data.loc[:, 'gram'].isna()].index
data.loc[ind, 'gram'] = 'n'

#Оставшиеся пропущенные значения в столбцах заменим средним значением
cols = ['min_Incub_period, h',
        'avg_Incub_period, h',
        'max_Incub_period, h',
        'growth_temp, C' ,
        'biosafety_level' ]

for i in cols:
  (data.loc[:, i][data.loc[:, i].isna()]) = np.mean((data.loc[:, i][~data.loc[:, i].isna()]))

#Немного поменяем местами колонки и оставим потенциально выгодные для модели регрессии и SMILES который потом перейдет в дескрипторы
data = data.loc[:, ['smiles', 'genus', 'NP_Synthesis', 'shape', 'NP size_min', 'NP size_max', 'NP size_avg', 'min_Incub_period, h', 'avg_Incub_period, h',
       'max_Incub_period, h', 'growth_temp, C', 'MDR_check', 'ZOI_drug', 'ZOI_NP', 'ZOI_drug_NP']]

data = data[~data.loc[:, 'genus'].isna()]


def normalize(dataset):
    dataset.iloc[:, :-1] = (dataset.iloc[:, :-1]-dataset.iloc[:, :-1].mean ())/dataset.iloc[:, :-1].std()
    return dataset

def getMolDescriptors(mol, missingVal=None):

    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res


def data_desc(dataset):
  dataset.index = np.array(range(len(dataset.loc[:, 'smiles'])))
  descriptors = pd.DataFrame([getMolDescriptors(Chem.MolFromSmiles(data.iloc[0, :]['smiles']))]).T
  for mol in dataset.loc[1:, 'smiles']:

    d = pd.DataFrame([getMolDescriptors(Chem.MolFromSmiles(mol))]).T
    descriptors = pd.concat([descriptors, d], axis = 1)

  return descriptors.T

#Заменим ZOI_drug_NP на флоты попутно заменив неккоректые значения
data.loc[:, 'ZOI_drug_NP'][data.loc[:, 'ZOI_drug_NP'] == '32+'] = 32
data.loc[:, 'ZOI_drug_NP'][data.loc[:, 'ZOI_drug_NP'] == '17+2'] = 18
data.loc[:, 'ZOI_drug_NP'] = data.loc[:, 'ZOI_drug_NP'].astype(float)

#Дропнем коррелирующие признаки
data = data.drop(['max_Incub_period, h', 'min_Incub_period, h', 'NP size_max',], axis=1)

#Считаем ВСЕ дескрипторы для всех молекул антибиотика

descriptors = data_desc(data)
descriptors = normalize(descriptors) #нормализация
descriptors = descriptors.dropna(axis=1) #убираем неудачано рассчитаные дескрипторы
descriptors.index = data.index

#Заменим строки на инты по следующей схеме
data.loc[:, ['NP_Synthesis',	'shape', 'genus']] = names_to_class(data.loc[:, ['NP_Synthesis',	'shape', 'genus']])

dataset = pd.concat([descriptors.loc[:, ['FpDensityMorgan1', 'fr_C_O_noCOO', 'fr_Al_COO', 'EState_VSA7', 'fr_COO', 'fr_nitro_arom', 'SlogP_VSA2', 'SlogP_VSA10', 'LabuteASA']], data.drop(['smiles'], axis=1)], axis=1)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.3)
rg = sklearn.ensemble.GradientBoostingRegressor()
rg.fit(x_train, y_train)
rmse = (sklearn.metrics.mean_squared_error(rg.predict(x_test), y_test)**0.5).astype(float).round(3)
r2 = (sklearn.metrics.r2_score(rg.predict(x_test), y_test)**0.5).astype(float).round(3)
# plt.title('RMSE = '+str(rmse)+', R2 = '+str(r2))
# plt.xlabel('experimental')
# plt.ylabel('predicted')
# plt.ylim(0, 50)
# plt.xlim(0, 50)
# plt.plot(rg.predict(x_test), y_test, 'ro'),  plt.plot([0, 50], [0, 50])
print(r2)