import sys
import csv
import numpy as np
import pandas as pd



features = ['id','origin','destination','origin_city','destination_city','distance','year','quarter','origin_population','destination_population','passengers','flights','seats','oil_price','flight_fare']
categorical_features = ['id','origin','destination','origin_city','destination_city']

dataset = 'flight_edges.tsv'
myData = pd.read_csv(dataset)


myDate['year'] = pd.DatetimeIndex(myDate['date']).year
myDate['month'] = pd.DatetimeIndex(myDate['date']).month
myDate['day'] = pd.DatetimeIndex(myDate['date']).day
myDate['quarter'] = pd.DatetimeIndex(myDate['date']).quarter
del myDate['date']

myDate = myDate[myDate.price != '.'] # delete any row with price == '.'
myDate = myDate[['year', 'month', 'day', 'quarter', 'price']]
myDate[['price']] = myDate[['price']].astype(float)


# create id column
myDate['id'] = myDate['year'].astype(str) + myDate['quarter'].astype(str)
myDate.set_index('id', inplace=True)

myData.to_csv('flight_edges_1.csv', encoding='utf-8')

dataset = 'flight_edges.tsv'
myData = pd.read_csv(dataset)

df = myData[["price"]]
df = df.groupby(df.index, as_index=True).agg({'price': 'mean'})

df.to_csv('oil_prices.csv', encoding='utf-8')
