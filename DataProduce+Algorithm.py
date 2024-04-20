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

df.head(5)

# Drop Columns

df = df.drop(df.columns[[0,14,15,16,17,19,20,21,22,23,24,25,26,27,28,30,31]], axis = 1)

# Drop NA

df.isna().sum()

# df Transfrom

age_col = {
          -0.95197: '18-24',
          -0.07854: '25 - 34',
          0.49788: '35 - 44',
          1.09449: '45 - 54',
          1.82213: '55 - 64',
          2.59171: '65+'
          }
df['Age'] = df['Age'].replace(age_col)

gender_col = {
            0.48246: 'Female',
            -0.48246: 'Male'
            }
df['Gender'] = df['Gender'].replace(gender_col)

education_col = {
            -2.43591: 'Left School Before 16 years',
            -1.73790: 'Left School at 16 years',
            -1.43719: 'Left School at 17 years',
            -1.22751: 'Left School at 18 years',
            -0.61113: 'Some College,No Certificate Or Degree',
            -0.05921: 'Professional Certificate/ Diploma',
            0.45468: 'University Degree',
            1.16365: 'Masters Degree',
            1.98437: 'Doctorate Degree',
            }
df['Education'] = df['Education'].replace(education_col)

country_col = {
            -0.09765: 'Australia',
            0.24923: 'Canada',
            -0.46841: 'New Zealand',
            -0.28519: 'Other',
            0.21128: 'Republic of Ireland',
            0.96082: 'UK',
            -0.57009: 'USA'
            }
df['Country'] = df['Country'].replace(country_col)

ethnicity_col = {
            -0.50212: 'Asian',
            -1.10702: 'Black',
            1.90725: 'Mixed-Black/Asian',
            0.12600: 'Mixed-White/Asian',
            -0.22166: 'Mixed-White/Black',
            0.11440: 'Other',
            -0.31685: 'White'
            }
df['Ethnicity'] = df['Ethnicity'].replace(ethnicity_col)

usage_col = {
    'CL0': 0, #'Never Used'
    'CL1': 0, #'Used over a Decade Ago'
    'CL2': 0, #'Used in Last Decade'
    'CL3': 1, #'Used in Last Year'
    'CL4': 1, #'Used in Last Month'
    'CL5': 1, #'Used in Last Week'
    'CL6': 1, #'Used in Last Day'
    }
df['Alcohol'] = df['Alcohol'].replace(usage_col)
df['Cannabis'] = df['Cannabis'].replace(usage_col)
df['Nicotine'] = df['Nicotine'].replace(usage_col)

# Data visualization
float_columns =  ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 'Alcohol', 'Cannabis', 'Nicotine']

for column in float_columns:
    express.histogram(data_frame=df, x=column).show()
# Drop Columns

df = df.drop(columns = ['Country', 'Ethnicity'])

N_float_columns =  ['Age', 'Gender', 'Education', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 'Alcohol', 'Cannabis', 'Nicotine']

for column in N_float_columns:
    express.histogram(data_frame=df, x=column).show()

# Transform values by using LabelEncoder

le = LabelEncoder()
N_df = df
L_col = ['Age', 'Gender', 'Education']

for c in L_col:
    N_df[c] = le.fit_transform(df[c])

N_df.describe()

# Minmax scaling

mms = MinMaxScaler()
mms.fit(N_df)
df_mms = N_df

df_mms[N_float_columns] = mms.transform(N_df[N_float_columns])


# Correlation Matrix

sns.heatmap(df_mms[N_float_columns].corr(),annot=True, cmap='Reds')
plt.show()

# Test split

x = df_mms.drop('Cannabis', axis = 1)
y = df_mms['Cannabis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

# KNN

knn10 = KNeighborsClassifier(n_neighbors = 10)

knn10.fit(x_train, y_train)

y_pred_10 = knn10.predict(x_test)

print("Accuracy with k=10", accuracy_score(y_test, y_pred_10)*100)

# LD

LD_model = LinearDiscriminantAnalysis()
LD_model.fit(x, y)

cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 2, random_state = 4)
scores = cross_val_score(LD_model, x, y, scoring = 'accuracy', cv = cv, n_jobs=-1)
print("Accuracy with n_splits = 10 and n_repeats = 2 ",np.mean(scores))   

# Gaussian NB 

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
NB_accuracy = accuracy_score(y_test, y_pred)

print("Accuracy with Gaussian NB", NB_accuracy*100)

# Logistic Regression 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

LR_model = LogisticRegression()
LR_model.fit(x_train, y_train)

y_pred = LR_model.predict(x_test)
LR_accuracy = accuracy_score(y_test, y_pred)

print("Accuracy with Logistic Regression",(LR_accuracy * 100))

# Feature importance

forest = RandomForestRegressor(random_state=0)
forest.fit(x_train, y_train)

importances = forest.feature_importances_

for i in range(len(importances)):
    print(f"{x.columns[i]}:"f"{importances[i]:.3f}")

# Visualization

barx = list(x.columns)
FT = pd.DataFrame(zip(barx, importances))
FT.columns = ['Feature', 'Percentage']
FT['Percentage'] = FT['Percentage'].apply(lambda x: round(x, 3))

px.bar(FT, x = 'Feature', y = 'Percentage', color = 'Feature', text_auto = True)

fig = px.pie(FT , values = 'Percentage',names= 'Feature',title = 'Feature importance')
fig.update_traces(textposition='outside',textinfo='label + percent')


print("--- %s seconds ---" % (time.time() - start_time))

