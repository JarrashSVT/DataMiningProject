import numpy as np
import pandas as pd 
import scipy.stats as stats
import matplotlib.pyplot as plt 
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
#from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

#mydata = pd.read_csv('DS1.csv', usecols=['origin', 'destination','oil_price','flight_fare'])
mydata = pd.read_csv('DS1.csv', usecols=['origin', 'destination','distance', 'quarter', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'oil_price','flight_fare'])

#mydata = pd.read_csv('DS1.csv')

df = pd.DataFrame(mydata)

print(type(mydata))
print("num of objects: ", len(mydata))

## encoding categorical values using label encoding
df['origin'] = df['origin'].astype('category')
df['destination'] = df['destination'].astype('category')
print(df.dtypes)

## assign the encoded variable to a new column using the cat.codes accessor:

df['origin_cat'] = df['origin'].cat.codes
df['destination_cat'] = df['destination'].cat.codes
#print(df.head())
#print(df['destination'])

X = df.drop(['flight_fare','origin', 'destination' ], axis=1)
Y = pd.DataFrame(mydata['flight_fare'])
#print('X', X)
#print('Y', Y)

lm = LinearRegression()

x_train,x_test,y_train,y_test= train_test_split(X, Y, test_size=0.2, random_state =4)

# train the model
lm.fit(x_train, y_train)
#LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
#lm.score()

for idx, col_name in enumerate(x_train.columns):
    print("The coefficient for {} is {}".format(col_name, lm.coef_[0][idx]))


intercept = lm.intercept_[0]
print("The intercept for our model is {}".format(intercept))

y_predicted = lm.predict(x_test)

print('y_predicted: ',y_predicted[100])
print( 'y_test: ', y_test.iloc[[100]])
print('y_predicted type: ',type(y_predicted))
print('y_test type: ',type(y_test[:0]))
print('#y_test =', len(y_test))
print('#y_predicted =' ,len(y_predicted))

score = lm.score(x_test, y_test)
#print(accuracy_score(Y, predicted))
print('score = ', score)


print('***** the mean squared error *****')


regression_model_mse = mean_squared_error(y_predicted, y_test)

print('regression_model_mse = ',regression_model_mse)
print('math.sqrt(regression_model_mse) = ',math.sqrt(regression_model_mse))

print(y_test)
print(y_predicted)

#y_test.to_csv('y_test.csv')
#pd.DataFrame(y_predicted).to_csv('y_predicted.csv')
#y_test['predicted'] = pd.DataFrame(y_predicted)
#df_result = pd.concat([y_test,pd.DataFrame(y_predicted)], axis=1)
#print(df_result)

col_name = 'origin_cat'
#plt.scatter(x_test[[col_name]], y_test,  color='black')
plt.plot(x_test[[col_name]], y_predicted, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
"""        
for idx, col_name in enumerate(x_train.columns):
    #print("The coefficient for {} is {}".format(col_name, lm.coef_[0][idx]))
    #print(col_name)
    #print(x_test[['oil_price']])

    if col_name == 'oil_price':
        print(x_test[[col_name]])
"""
    


print("=========== BayesianRidge =============")

lm = BayesianRidge()
#x_train,x_test,y_train,y_test= train_test_split(X, Y, test_size=0.2, random_state =4)

# train the model
lm.fit(x_train, y_train)

y_predicted = lm.predict(x_test)

print('y_predicted: ',y_predicted[100])
print( 'y_test: ', y_test.iloc[[100]])
print('y_predicted type: ',type(y_predicted))
print('y_test type: ',type(y_test[:0]))
print('#y_test =', len(y_test))
print('#y_predicted =' ,len(y_predicted))

score = lm.score(x_test, y_test)
#print(accuracy_score(Y, predicted))
print('score = ', score)
"""
print("=========== BayesianRidge =============")

lm = BayesianRidge()
#x_train,x_test,y_train,y_test= train_test_split(X, Y, test_size=0.2, random_state =4)

# train the model
lm.fit(x_train, y_train)

y_predicted = lm.predict(x_test)

print('y_predicted: ',y_predicted[100])
print( 'y_test: ', y_test.iloc[[100]])
print('y_predicted type: ',type(y_predicted))
print('y_test type: ',type(y_test[:0]))
print('#y_test =', len(y_test))
print('#y_predicted =' ,len(y_predicted))

score = lm.score(x_test, y_test)
#print(accuracy_score(Y, predicted))
print('score = ', score)

print("=========== SVM -> SVR =============")

clf = SVR('linear')
x_train,x_test,y_train,y_test= train_test_split(X, Y, test_size=0.2, random_state =4)

for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)

    """