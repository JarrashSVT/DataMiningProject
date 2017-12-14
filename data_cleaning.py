import sys
import csv
import numpy as np
import pandas as pd



features = ['id','origin','destination','origin_city','destination_city','distance','year','quarter','origin_population','destination_population','passengers','flights','seats','oil_price','flight_fare']
categorical_features = ['id','origin','destination','origin_city','destination_city']
 
target = ['flight_fare']
dataset = 'DS1.csv'
myData = pd.read_csv(dataset, usecols=features)

#Remove incomplete rows
#drop rows that have all NA values
myData.dropna(how='all')
myData.dropna(thresh=5)
myData.dropna(subset=['origin'])
myData.dropna(subset=['destination'])
myData.dropna(subset=['origin_city'])
myData.dropna(subset=['destination_city'])


#missing data
myData.distance = myData.distance.fillna(myData.distance.mean())
myData.quarter = myData.quarter.fillna(1)
myData.origin_population = myData.origin_population.fillna(myData.origin_population.mean())
myData.destination_population = myData.destination_population.fillna(myData.destination_population.mean())
myData.passengers = myData.passengers.fillna(myData.passengers.mean())
myData.flights = myData.flights.fillna(myData.flights.mean())
myData.seats = myData.seats.fillna(myData.seats.mean())
myData.oil_price = myData.oil_price.fillna(myData.oil_price.mean())
myData.flight_fare = myData.flight_fare.fillna(myData.flight_fare.mean())


for feature in categorical_features:
    myData[feature].str.strip()


myData.to_csv('DS1-cleaned.csv', encoding='utf-8')


#
features = ['Year','quarter','citymarketid_1','citymarketid_2','city1','city2','airportid_1','airportid_2','airport_1','airport_2','nsmiles','passengers','fare','carrier_lg','large_ms','fare_lg','carrier_low','lf_ms','fare_low']
categorical_features = ['city1','city2','airport_2','carrier_lg','carrier_low']

 
dataset = '1996_2017-Table1a_0.csv'
myData = pd.read_csv(dataset, usecols=features)

#Remove incomplete rows
#drop rows that have all NA values
myData.dropna(how='all')
myData.dropna(thresh=5)
myData.dropna(subset=['airport_1'])
myData.dropna(subset=['airport_2'])
myData.dropna(subset=['city1'])
myData.dropna(subset=['city2'])
myData.dropna(subset=['Year'])


#missing data
myData.quarter = myData.quarter.fillna(1)
myData.nsmiles = myData.nsmiles.fillna(myData.nsmiles.mean())
myData.passengers = myData.passengers.fillna(myData.passengers.mean())
myData.fare = myData.fare.fillna(myData.fare.mean())
myData.large_ms = myData.large_ms.fillna(myData.large_ms.mean())
myData.fare_lg = myData.fare_lg.fillna(myData.fare_lg.mean())
myData.lf_ms = myData.lf_ms.fillna(myData.lf_ms.mean())
myData.fare_low = myData.fare_low.fillna(myData.fare_low.mean())


for feature in categorical_features:
    myData[feature].str.strip()


myData.to_csv('1996_2017-Table1a_0-cleaned.csv', encoding='utf-8')


#
features = ['DATE','DCOILBRENTEU']

 
dataset = 'oil price DCOILBRENTEU.csv'
myData = pd.read_csv(dataset, usecols=features)

#Remove incomplete rows
#drop rows that have all NA values
myData.dropna(how='all')
myData.dropna(subset=['DCOILBRENTEU'])
myData.dropna(subset=['DATE'])
myData = myData.drop(myData.index[myData.DCOILBRENTEU == '.'])


myData.to_csv('oil price DCOILBRENTEU.csv', encoding='utf-8')