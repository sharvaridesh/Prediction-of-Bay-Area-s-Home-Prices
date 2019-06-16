# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 18:48:18 2019

@author: Sharvari
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor

#Importing dataset using pandas
df = pd.read_csv('final_data.csv')
df.info()

#Dropping columns 
df.drop(df.columns[[0, 2, 3, 15, 17, 18]], axis=1, inplace=True)
df.info()

#Changing datatype of column 'zindexvalue'
df['zindexvalue'] = df['zindexvalue'].str.replace(',','')
df['zindexvalue'] = df['zindexvalue'].astype('float64')
df.info()

print(df.describe())

#Plotting histogram of all features 
df.hist(bins=50, figsize=(15,20))
plt.show()

df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(10,7),
    c="lastsoldprice", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)

#Correlation Matrix
corr_matrix = df.corr()
corr_matrix["lastsoldprice"].sort_values(ascending=False)

df['price_per_sqft'] = df['lastsoldprice']/df['finishedsqft']
corr_matrix = df.corr()
corr_matrix["lastsoldprice"].sort_values(ascending=False)

l = len(df['neighborhood'].value_counts())
print(l)

#Grouping by neighbourhood into low price, high frequency low price, high frequency high price
freq = df.groupby('neighborhood').count()['address']
mean = df.groupby('neighborhood').mean()['price_per_sqft']
cluster = pd.concat([freq, mean], axis=1)
cluster['neighborhood'] = cluster.index
cluster.columns = ['freq', 'price_per_sqft','neighborhood']

cluster1 = cluster[cluster.price_per_sqft < 756]
print('\n',cluster1.index)

cluster_temp = cluster[cluster.price_per_sqft >= 756]
cluster2 = cluster_temp[cluster_temp.freq <123]
print('\n',cluster2.index)

cluster3 = cluster_temp[cluster_temp.freq >=123]
print('\n',cluster3.index)

def get_cluster(c):
    if c in cluster1.index:
        return 'low price'
    elif c in cluster2.index:
        return 'high price low frequency'
    else:
        return 'high price low frequency'

df['group']=df.neighborhood.apply(get_cluster)
df.drop(df.columns[[0, 4, 6, 7, 8, 13]], axis=1, inplace=True)
df = df[['bathrooms', 'bedrooms', 'finishedsqft', 'totalrooms', 'usecode', 'yearbuilt','zindexvalue', 'group', 'lastsoldprice']]
print(df.head())


X = df[['bathrooms', 'bedrooms', 'finishedsqft', 'totalrooms', 'usecode', 'yearbuilt', 
         'zindexvalue', 'group']]
Y = df[['lastsoldprice']]

n = pd.get_dummies(df.group)
X = pd.concat([X,n],axis=1)

m = pd.get_dummies(df.usecode)
X = pd.concat([X,m],axis=1)

X.drop(['group','usecode'],inplace=True, axis=1)

#Splitting dataframe into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Linear Regression Model
lm= LinearRegression()
lm.fit(X_train,Y_train)

Y_pred = lm.predict(X_test)
r = lm.score(X_test, Y_test)
print('\nLinear Regression R squared: %.4f' %r)

#RMSE metric
lm_mse = mean_squared_error(Y_pred, Y_test)
lm_rmse = np.sqrt(lm_mse)
print('\nLinear Regression RMSE: %.4f' %lm_rmse)

lm_mae = mean_absolute_error(Y_pred, Y_test)
print('\nLinear Regression MAE: %.4f\n' %lm_mae)

#RANDOM FOREST 
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(X_train, Y_train)

print('\nRandom Forest R squared: %.4f' % forest_reg.score(X_test, Y_test))

Y_pred1 = forest_reg.predict(X_test)
forest_mse = mean_squared_error(Y_pred1, Y_test)
forest_rmse = np.sqrt(forest_mse)
print('\nRandom Forest RMSE: %.4f\n' % forest_rmse)

#GRADIENT BOOSTING
model1 = ensemble.GradientBoostingRegressor()
model1.fit(X_train, Y_train)
print('\nGradient Boosting R squared": %.4f' % model1.score(X_test, Y_test))

Y_pred2 = model1.predict(X_test)
model_mse = mean_squared_error(Y_pred2, Y_test)
model_rmse = np.sqrt(model_mse)
print('\nGradient Boosting RMSE: %.4f\n' % model_rmse)







