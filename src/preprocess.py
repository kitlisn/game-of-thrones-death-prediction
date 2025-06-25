import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_data(train_path: str, test_path: str):
    data = pd.read_csv(Path(train_path)) #train df
    data1 = pd.read_csv(Path(test_path)) #test df
    return data, data1

load_data('!YOUR PATH!', '!YOUR PATH!')

data.fillna({'dateOfBirth': data['dateOfBirth'].median(), 'age': data['age'].median()}, inplace=True) #заполнил столбцы с датой рождения и возрастом медианой (использовал словари, потому что без них могла быть ошибка при создании копий таблицы)
data['house'] = data['house'].fillna('Unk')

data1.fillna({'dateOfBirth': data['dateOfBirth'].median(), 'age': data['age'].median()}, inplace=True) #аналогично для тестовых данных
data1['house'] = data1['house'].fillna('Unk')

data.loc[(data.popularity >= 0.5), ['isPopular']] = 1 #создание признака isPopular
data.loc[(data.popularity < 0.5), ['isPopular']] = 0
data.loc[(data.numDeadRelations > 0), ['boolDeadRelations']] = 1 #создание признака boolDeadRelations
data.loc[(data.numDeadRelations == 0), ['boolDeadRelations']] = 0

data1.loc[(data1.popularity >= 0.5), ['isPopular']] = 1 #аналогично для тестовых данных
data1.loc[(data1.popularity < 0.5), ['isPopular']] = 0
data1.loc[(data1.numDeadRelations > 0), ['boolDeadRelations']] = 1
data1.loc[(data1.numDeadRelations == 0), ['boolDeadRelations']] = 0

cult = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],
    'Asshai': ["asshai'i", 'asshai'],
    'Lysene': ['lysene', 'lyseni'],
    'Andal': ['andal', 'andals'],
    'Braavosi': ['braavosi', 'braavos'],
    'Dornish': ['dornishmen', 'dorne', 'dornish'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Westerosi': ['westeros', 'westerosi'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvoshi': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Free Folk': ['wildling', 'first men', 'free folk'],
    'Qartheen': ['qartheen', 'qarth'],
    'Reach': ['the reach', 'reach', 'reachmen'],
    'lhazareen': ['lhazareen', 'lhazarene'],
}

data['culture'] = data['culture'].fillna('Unk') #заполнение NaN'ов заглушкой
data['culture'] = data['culture'].str.lower() #перевод данных в столбце сulture к нижнему регистру для упрощения обработки
for cults in cult:
    data['culture'] = data['culture'].replace(cult[cults], cults) #упрощение признака culture
valid_cultures = set(cult.keys()) #cоздал набор уникальных значений из словаря с культурами

data['culture'] = data['culture'].apply(lambda x: x if x in valid_cultures else 'other') #оставлю в словаре только культуры, которые есть в словаре (остальные и пропуски Unk меняю на other)

data1['culture'] = data1['culture'].fillna('Unk') #аналогично для тестовых данных
data1['culture'] = data1['culture'].str.lower()
for cults in cult:
    data1['culture'] = data1['culture'].replace(cult[cults], cults)
valid_cultures = set(data['culture'])

data1['culture'] = data1['culture'].apply(lambda x: x if x in valid_cultures else 'other')

data.drop(columns=['name', 'title', 'mother',	'father', 'heir', 'spouse', 'isAliveMother', 'isAliveFather', 'isAliveHeir', 'isAliveSpouse','numDeadRelations','popularity'], inplace = True)
#столбцы name, mother, father, heir, spouse содержат уникальные значения (для каждого объекта разные), не повлияют на целевую переменную
#в столбцах isAliveMother, isAliveFather, isAliveHeir, isAliveSpouse много пропусков, тоже удаляю
#столбцы numDeadRelations,popularity заменил на новые
data1.drop(columns=['name', 'title', 'mother',	'father', 'heir', 'spouse', 'isAliveMother', 'isAliveFather', 'isAliveHeir', 'isAliveSpouse','numDeadRelations','popularity'], inplace = True) #удаление ненужных столбцов

data['age_plus_date'] = data['age'] + data['dateOfBirth'] #создаю новый признак - сумму возраста и года рождения
data1['age_plus_date'] = data1['age'] + data1['dateOfBirth']

unprocessed = ['culture', 'house'] #признаки для one-hot кодировки

encoder = OneHotEncoder(sparse_output=False) #one-hot кодировка
one_hot_encoded = encoder.fit_transform(data[unprocessed])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(unprocessed), index=data.index)
data_encoded = pd.concat([data, one_hot_df], axis=1)
data_encoded.drop(columns=unprocessed, inplace = True) #сконкатенировал с новыми столбцами, старые удалил

for col in unprocessed: #по значениям тестовых данных проходимся с тем же кодировщиком, чтобы не было ошибок если появятся новые значения
    data1[col] = data1[col].apply(lambda x: x if x in encoder.categories_[unprocessed.index(col)] else 'Unk') #если появляются новые значение и NaN'ы меняем на Unk
one_hot_encoded_1 = encoder.transform(data1[unprocessed])
one_hot_df_1 = pd.DataFrame(one_hot_encoded_1, columns=encoder.get_feature_names_out(unprocessed), index=data1.index)
data1_encoded = pd.concat([data1, one_hot_df_1], axis=1)
data1_encoded.drop(columns=unprocessed, inplace=True)

data_encoded.drop(columns=['book2', 'book3', 'book5', 'isMarried', 'isNoble'], inplace = True) #удаление ненужных столбцов

data1_encoded.drop(columns=['book2', 'book3', 'book5', 'isMarried', 'isNoble'], inplace = True) #аналогично для тестовых данных

X = data_encoded.drop(columns=['isAlive']).values
y = data_encoded['isAlive'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)