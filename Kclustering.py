import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly import express
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import random
import time

start_time = time.time()

df = pd.read_csv('drug_consumption.csv')
df = df.drop(df.columns[[0,14,15,16,17,19,20,21,22,23,24,25,26,27,28,30,31]], axis = 1)

usage_col = {
    'CL0': 1, #'Never Used'
    'CL1': 1, #'Used over a Decade Ago'
    'CL2': 2, #'Used in Last Decade'
    'CL3': 3, #'Used in Last Year'
    'CL4': 4, #'Used in Last Month'
    'CL5': 4, #'Used in Last Week'
    'CL6': 4, #'Used in Last Day'
    }
df['Alcohol'] = df['Alcohol'].replace(usage_col)
df['Cannabis'] = df['Cannabis'].replace(usage_col)
df['Nicotine'] = df['Nicotine'].replace(usage_col)

for i, row in df.iterrows():
    us = 0
    if row['Nicotine'] == 1:
        us = random.randrange(0,25)
    elif row['Nicotine'] == 2:
        us = random.randrange(26,50)
    elif row['Nicotine'] == 3:
        us = random.randrange(51,75)
    elif row['Nicotine'] == 4:
        us = random.randrange(76,100)
    df.at[i,'Nicotine'] = us

for i, row in df.iterrows():
    usc = 0
    if row['Cannabis'] == 1:
        usc = random.randrange(0,25)
    elif row['Cannabis'] == 2:
        usc = random.randrange(26,50)
    elif row['Cannabis'] == 3:
        usc = random.randrange(51,75)
    elif row['Cannabis'] == 4:
        usc = random.randrange(76,100)
    df.at[i,'Cannabis'] = usc

for i, row in df.iterrows():
    usa = 0
    if row['Alcohol'] == 1:
        usa = random.randrange(0,25)
    elif row['Alcohol'] == 2:
        usa = random.randrange(26,50)
    elif row['Alcohol'] == 3:
        usa = random.randrange(51,75)
    elif row['Alcohol'] == 4:
        usa = random.randrange(76,100)
    df.at[i,'Alcohol'] = usa

# Spliting

x = df['Nicotine']
y = df['Cannabis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

# train k mean

data = list(zip(x_train, y_train))

inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters = 4)
kmeans.fit(data)

scatter = plt.scatter(x_train, y_train, c = kmeans.labels_)
plt.title('K = 4 (Train)')
plt.xlabel('Nicotine')
plt.ylabel('Cannabis')
plt.legend(*scatter.legend_elements(), loc = 'center right', title = 'Clusters')
plt.show()

# test k mean

data_t = list(zip(x_test, y_test))

inertias_t = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data_t)
    inertias_t.append(kmeans.inertia_)

plt.plot(range(1,11), inertias_t, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters = 4)
kmeans.fit(data_t)

scatter_t = plt.scatter(x_test, y_test, c = kmeans.labels_)
plt.title('K = 4 (Test)')
plt.xlabel('Nicotine')
plt.ylabel('Cannabis')
plt.legend(*scatter_t.legend_elements(), loc = 'center right', title = 'Clusters')
plt.show()


kmeans = KMeans(n_clusters = 4)
predicted = kmeans.fit_predict(df[['Nicotine','Cannabis']])
df['Cluster'] = predicted

print(df)

print("--- %s seconds ---" % (time.time() - start_time))

