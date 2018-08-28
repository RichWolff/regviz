# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


# TODO
# Load a useful data set, like car value, home value, or own data set

# TODO
# Show off the ability to change the scale of features 
# (think call center, brandtracker to 100bps or 10,000 google impressions)

## Load a data set
from sklearn.datasets import california_housing

## Set random seed for reproducability 
np.random.seed(21)

## Create pandas dataframe from random values
df = pd.DataFrame()
df['y']=np.random.randint(1,1000,size=1000)
df['x1']=np.random.randint(1,1000,size=1000)
df['x2']=np.random.randint(1,1000,size=1000)
df['x3']=np.random.randint(1,1000,size=1000)
df['x4']=np.random.randint(1,1000,size=1000) 
df['x6']=df['y']*np.random.normal(loc=.5,scale=1,size=1000)


# initialize and fit OLS model from random data
import statsmodels.api as sm

model_0 = sm.OLS(df['y'],df.drop('y',axis=1))
res = model_0.fit()

# Import and create regviz object from a fitted sm.OLS object
from stat_graph import regviz
viz = regviz(res)

# Fit the dataframe that will supply information for vizualization
viz.fit()

# display raw data from fitted data frame
print('Original Statistic list')
print(viz.get_feature_data(),end ='\n\n')

# sort features by statistic
# use coef to sort by coefficients, pval to sort by pvalue
print('Sorted Feature List By Pvals and descending')
viz.sort_features(sort_by='pval',ascending='True')
print(viz.get_feature_data(),end='\n\n')

print('Sorted Feature List By Coefs and ascending')
viz.sort_features(sort_by='coefs',ascending='True')
print(viz.get_feature_data(),end='\n\n')


#Update feature names on fit dataframe
viz.set_feature_names(['Worst Coef','Mid Coef 1',
                       'Mid Coef 2','Mid Coef 3','Best Coef'])
print('Renamed Feature List')
print(viz.get_feature_data(),end='\n\n')

# Plt the fitted data
viz.plot()

# plt the data but hide a feature
viz.plot(hide_features=['Mid Coef 1','Mid Coef 2'])

# Create your own axes and plot to that
fig = plt.figure(figsize=(5,2))
ax = fig.add_axes([0,0,1,1])
viz.plot(ax=ax, hide_features=['Mid Coef 1','Mid Coef 2'])
