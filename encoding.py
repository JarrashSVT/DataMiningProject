import sys
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

#features = ['origin', 'destination', 'oil_price', 'quarter', 'year', 'flight_fare']
features = ['origin', 'destination', 'oil_price', 'flight_fare']

features2 = ['origin', 'destination', 'oil_price']
#featuresToDrop = ['origin', 'destination','distance', 'quarter', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'flight_fare']
target = ['flight_fare']
dataset = 'DS1.csv'
theData = pd.read_csv(dataset, usecols=features)


df = pd.DataFrame(theData)
#df_ = pd.DataFrame(mydata)

origin_df = df['origin']
origin_arr = np.array(np.array(origin_df))
print(origin_arr)
print(type(origin_arr))
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(origin_arr)
#print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
print(type(onehot_encoded))


with open('origin.csv', 'a') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["Origin", "Origin_Encoded"])

with open('destination.csv', 'a') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["Destination", "Destination_Encoded"])

origin_encoded_list = [] 
for origin_abbr, origin_encoded in zip(origin_arr,onehot_encoded):
    #print("{} --> {}".format(origin_abbr,origin_encoded))
    #origin_encoded_list.append([origin_abbr,origin_encoded])
    #origin_encoded = origin_encoded.replace('\n','')
    with open('origin.csv', 'a') as file:
        writer = csv.writer(file, delimiter = ",")
        writer.writerow([origin_abbr,origin_encoded])



#===================================

destination_df = df['destination']
destination_arr = np.array(remove_duplicates(np.array(destination_df)))
print(destination_arr)
print(type(destination_arr))
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(destination_arr)
#print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
print(type(onehot_encoded))

origin_encoded_list = [] 
for origin_abbr, origin_encoded in zip(destination_arr,onehot_encoded):
    #print("{} --> {}".format(origin_abbr,origin_encoded))
    #origin_encoded_list.append([origin_abbr,origin_encoded])
    with open('destination.csv', 'a') as file:
        writer = csv.writer(file, delimiter = ",")
        writer.writerow([origin_abbr,origin_encoded])

#print(origin_encoded_list)
#print(df)