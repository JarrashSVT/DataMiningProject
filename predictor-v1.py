

"""
=========================================================
Flight Fare Predictor
=========================================================

1- Analyze dataset
2- Analyze dataset for specific origin and destination
3- Predict a Flight Fare


Linear Models Options:              Encoding Options:
0- LinearRegression                 0- Lable Encoding
1- Lasso                            1- One-Hot Encoding
2- Ridge

"""
print(__doc__)

import argparse
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer


def execute_model(X, Y, model_,encoding ,debug):
    #model = LinearRegression()
    model = get_model(model_)
    x_train,x_test,y_train,y_test= train_test_split(X, Y, test_size=0.2, random_state =4)

    # train the model
    model.fit(x_train, y_train)
    logToTrace('\n')
    if(encoding == 0):
        print()
        for idx, col_name in enumerate(x_train.columns):
            print("The coefficient for {} is {}".format(col_name, model.coef_[0][idx]))
            logToTrace("The coefficient for {} is {}".format(col_name, model.coef_[0][idx]))

    print()
    intercept = model.intercept_[0]
    print("The intercept for our model is {}".format(intercept))
    logToTrace("The intercept for our model is {}".format(intercept))

    y_predicted = model.predict(x_test)


    score = model.score(x_test, y_test)
    #print(accuracy_score(Y, predicted))
    print()
    print('score = ', score)
    logToTrace('\n')
    logToTrace('score = '+ str(score))
    print()
    print('***** The Mean Squared Error *****')
    print()

    logToTrace('\n')
    logToTrace('***** The Mean Squared Error *****')

    regression_model_mse = mean_squared_error(y_predicted, y_test)

    print('regression_model_mse = ',regression_model_mse)
    logToTrace('regression_model_mse = ' + str(regression_model_mse))
    print('math.sqrt(regression_model_mse) = ',math.sqrt(regression_model_mse))
    logToTrace('math.sqrt(regression_model_mse) = ' + str(math.sqrt(regression_model_mse)))

#==============================================================================================================

def execute_model_to_predict_fare(user_input,X, Y, model,debug):
    model = get_model(model)
    x_input = np.array(user_input)

    if(debug):
        print("User Input:", x_input)
    #x_train,x_test,y_train,y_test= train_test_split(X, Y, test_size=0.2, random_state =4)

    # train the model
    model.fit(X, Y)

    #for idx, col_name in enumerate(X.columns):
    #    print("The coefficient for {} is {}".format(col_name, model.coef_[0][idx]))

    if(debug):
        print()
        intercept = model.intercept_[0]
        print("The intercept for our model is {}".format(intercept))

    if(debug):
        print('x_input type is ',type(x_input))
    y_predicted = model.predict(x_input.reshape(1,-1))
    print()
    print('Predicted flight fare is ', y_predicted.item(0))
    logToTrace('\n')
    logToTrace('Predicted flight fare is ' + str(y_predicted.item(0)))

    score = model.score(X, Y)
    #print(accuracy_score(Y, predicted))
    print()
    print('score = ', score)
    logToTrace('\n')
    logToTrace('score = ' + str(score))

    

#==============================================================================================================

def prepare_model(model, dataset, debug):

    print('dataset is ', dataset)
    logToTrace('dataset is ' + dataset)
    print()
    logToTrace('\n')
    print('Model is ', get_model(model))
    logToTrace('Model is ' + str(get_model(model)))

    features = ['origin', 'destination','distance', 'quarter','year', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'oil_price','flight_fare']
    features2 = ['origin', 'destination','distance', 'quarter', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'oil_price','origin_cat','destination_cat']
    featuresToDrop = ['origin', 'destination','distance', 'quarter', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'flight_fare']
    target = ['flight_fare']
    mydata = pd.read_csv('DS1.csv', usecols=features)


    df = pd.DataFrame(mydata)
    #df_ = pd.DataFrame(mydata)

    print()
    logToTrace('\n')
    print('***** Dataset Summary *****')
    logToTrace('\n')
    print('***** Dataset Summary *****')
    logToTrace('\n')
    print(df.describe())
    logToTrace(df.describe())


    #print(df)
    ## encoding categorical values using label encoding
    df['origin'] = df['origin'].astype('category')
    df['destination'] = df['destination'].astype('category')
    #print(df.dtypes)

    ## assign the encoded variable to a new column using the cat.codes accessor:
    df['origin_cat'] = df['origin'].cat.codes
    df['destination_cat'] = df['destination'].cat.codes
    #print(df.head())
    #print(df['destination'])


    X = df.drop(['flight_fare','origin', 'destination' ], axis=1)
    #Y = pd.DataFrame(mydata['flight_fare'])
    Y = df.drop(features2, axis=1)
    #Y = df['flight_fare']
    if(debug):
        print('X', X)
        print('Y', Y)

        pd.DataFrame(X).to_csv('X.csv')


    execute_model(X, Y, model ,debug)

#==============================================================================================================
def prepare_model_with_encoding( model, encoding ,dataset, debug):
    
    print()
    print('Analyzing {} dataset '.format(dataset))
    logToTrace('Analyzing {} dataset '.format(dataset))
    print('Model is ', get_model(model))
    logToTrace('Model is '+ str(get_model(model)))
    if(encoding == 0):
        print('Encoding is Label Encoding')
        logToTrace('Encoding is Label Encoding')
    else:
        print('Encoding is One-Hot Encoding')
        logToTrace('Encoding is One-Hot Encoding')

        #features = ['origin', 'destination', 'oil_price', 'quarter', 'year', 'flight_fare']

    features = ['origin', 'destination','distance', 'quarter','year', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'oil_price','flight_fare']
    featuresToDropY = ['origin', 'destination','distance', 'quarter', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'oil_price','origin_cat','destination_cat']
    featuresToEncode = ['origin','destination', 'quarter', 'year']
    #featuresToDrop = ['origin', 'destination','distance', 'quarter', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'flight_fare']
    target = 'flight_fare'
    mydata = pd.read_csv('DS1.csv', usecols=features)


    df = pd.DataFrame(mydata)
    #df_ = pd.DataFrame(mydata)


    if(encoding == 0):
        ## encoding categorical values using label encoding
        df['origin'] = df['origin'].astype('category')
        df['destination'] = df['destination'].astype('category')
        #print(df.dtypes)

        ## assign the encoded variable to a new column using the cat.codes accessor:
        df['origin_cat'] = df['origin'].cat.codes
        df['destination_cat'] = df['destination'].cat.codes


        featuresToDropX = ['origin', 'destination',target]
        #print(df.head())
        #print(df['destination'])
    else:
        df = pd.get_dummies(df, columns=featuresToEncode)
        featuresToDropY = df.columns.values.tolist()
        featuresToDropY.remove(target)

        if(debug):
            pd.DataFrame(df).to_csv('_X.csv')
            print("Headers: ", featuresToDropY)

        featuresToDropX = [target]

    print()
    print('***** Dataset Summary *****')
    logToTrace('\n')
    logToTrace('***** Dataset Summary *****')
    logToTrace('\n')

    print()
    print(df.describe())
    logToTrace(str(df.describe()))

    X = df.drop(featuresToDropX, axis=1)
    #Y = pd.DataFrame(mydata['flight_fare'])
    Y = df.drop(featuresToDropY, axis=1)
    #Y = df['flight_fare']
    if(debug):
        print('X', X)
        print('Y', Y)

        #pd.DataFrame(X).to_csv('X.csv')


    execute_model(X, Y, model,encoding ,debug)

#==============================================================================================================



def prepare_model_with_origin_and_destination(origin, destination, model, dataset, debug):

    print()
    print('dataset is ', dataset)
    print('Model is ', get_model(model))

    features = ['origin', 'destination','distance', 'quarter','year', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'oil_price','flight_fare']
    features2 = ['origin', 'destination','distance', 'quarter', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'oil_price','origin_cat','destination_cat']
    featuresToDrop = ['origin', 'destination','distance', 'quarter', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'flight_fare']
    target = ['flight_fare']
    mydata = pd.read_csv('DS1.csv', usecols=features)


    df = pd.DataFrame(mydata)
    #df_ = pd.DataFrame(mydata)


    #print(df)
    ## encoding categorical values using label encoding
    df['origin'] = df['origin'].astype('category')
    df['destination'] = df['destination'].astype('category')
    #print(df.dtypes)

    ## assign the encoded variable to a new column using the cat.codes accessor:
    df['origin_cat'] = df['origin'].cat.codes
    df['destination_cat'] = df['destination'].cat.codes
    #print(df.head())
    #print(df['destination'])


    df = df.loc[df['origin'] == origin]
    if(len(df) == 0):
        print('entered origin not found')
        sys.exit()
    
    df = df.loc[df['destination'] == destination]
    if(len(df) == 0):
        print('no records found for the entered rout {} -> {}'.format(origin,destination))
        sys.exit()
    
    print()
    print('{} records found for rout {} -> {}'.format(len(df), origin, destination))
    

    print()
    print('***** Dataset Summary *****')
    print()
    print(df.describe())

    X = df.drop(['flight_fare','origin', 'destination' ], axis=1)
    #Y = pd.DataFrame(mydata['flight_fare'])
    Y = df.drop(features2, axis=1)
    #Y = df['flight_fare']
    if(debug):
        print('X', X)
        print('Y', Y)

        pd.DataFrame(X).to_csv('X.csv')


    execute_model(X, Y, model ,debug)

#==============================================================================================================



def prepare_model_with_origin_and_destination_with_encoding(origin, destination, model, encoding ,dataset, debug):

    print()
    print('Analyzing {} dataset '.format(dataset))
    logToTrace('Analyzing {} dataset '.format(dataset))
    print('Model is ', get_model(model))
    logToTrace('Model is '+ str(get_model(model)))
    if(encoding == 0):
        print('Encoding is Label Encoding')
        logToTrace('Encoding is Label Encoding')
    else:
        print('Encoding is One-Hot Encoding')
        logToTrace('Encoding is One-Hot Encoding')

        #features = ['origin', 'destination', 'oil_price', 'quarter', 'year', 'flight_fare']

    features = ['origin', 'destination','distance', 'quarter','year', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'oil_price','flight_fare']
    featuresToDropY = ['origin', 'destination','distance', 'quarter', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'oil_price','origin_cat','destination_cat']
    featuresToEncode = ['origin','destination', 'quarter', 'year']
    #featuresToDrop = ['origin', 'destination','distance', 'quarter', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'flight_fare']
    target = 'flight_fare'
    mydata = pd.read_csv('DS1.csv', usecols=features)


    df = pd.DataFrame(mydata)
    #df_ = pd.DataFrame(mydata)


    if(encoding == 0):
        ## encoding categorical values using label encoding
        df['origin'] = df['origin'].astype('category')
        df['destination'] = df['destination'].astype('category')
        #print(df.dtypes)

        ## assign the encoded variable to a new column using the cat.codes accessor:
        df['origin_cat'] = df['origin'].cat.codes
        df['destination_cat'] = df['destination'].cat.codes

        df = df.loc[df['origin'] == origin]
        df = df.loc[df['destination'] == destination]

        featuresToDropX = ['origin', 'destination',target]
        #print(df.head())
        #print(df['destination'])
    else:
        df = pd.get_dummies(df, columns=featuresToEncode)
        featuresToDropY = df.columns.values.tolist()
        featuresToDropY.remove(target)

        if(debug):
            pd.DataFrame(df).to_csv('_X.csv')
            print("Headers: ", featuresToDropY)

        df = df.loc[df['origin_'+origin] == 1]
        df = df.loc[df['destination_'+destination] == 1]

        featuresToDropX = [target]

    #df = df.loc[df['origin'] == origin]
    #if(len(df) == 0):
    #    print('entered origin not found')
    #    sys.exit()
    
    #df = df.loc[df['destination'] == destination]
    if(len(df) == 0):
        print('no records found for the entered rout {} -> {}'.format(origin,destination))
        sys.exit()
    
    print()
    print('{} records found for rout {} -> {}'.format(len(df), origin, destination))
    logToTrace('{} records found for rout {} -> {}'.format(len(df), origin, destination))

    print()
    print('***** Dataset Summary *****')
    logToTrace('\n')
    logToTrace('***** Dataset Summary *****')
    logToTrace('\n')

    print()
    print(df.describe())
    logToTrace(str(df.describe()))

    X = df.drop(featuresToDropX, axis=1)
    #Y = pd.DataFrame(mydata['flight_fare'])
    Y = df.drop(featuresToDropY, axis=1)
    #Y = df['flight_fare']
    if(debug):
        print('X', X)
        print('Y', Y)

        #pd.DataFrame(X).to_csv('X.csv')


    execute_model(X, Y, model,encoding ,debug)

#==============================================================================================================

def predict_fare(origin, destination, oil_price, model,dataset, debug):

    print()
    print('origin is ' , origin)
    print('destination is ', destination)
    print('dataset is ', dataset)
    print('Model is ', get_model(model))

    #features = ['origin', 'destination', 'oil_price', 'quarter', 'year', 'flight_fare']
    features = ['origin', 'destination', 'oil_price', 'flight_fare']

    features2 = ['origin', 'destination', 'oil_price','origin_cat','destination_cat']
    #featuresToDrop = ['origin', 'destination','distance', 'quarter', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'flight_fare']
    target = ['flight_fare']
    mydata = pd.read_csv(dataset, usecols=features)


    df = pd.DataFrame(mydata)
    #df_ = pd.DataFrame(mydata)


    #print(df)
    ## encoding categorical values using label encoding
    df['origin'] = df['origin'].astype('category')
    df['destination'] = df['destination'].astype('category')
    #print(df.dtypes)

    ## assign the encoded variable to a new column using the cat.codes accessor:
    df['origin_cat'] = df['origin'].cat.codes
    df['destination_cat'] = df['destination'].cat.codes
    #print(df.head())
    #print(df['destination'])

    df = df.loc[df['origin'] == origin]
    if(len(df) == 0):
        print('entered origin not found')
        sys.exit()
    
    df = df.loc[df['destination'] == destination]
    if(len(df) == 0):
        print('no records found for the entered rout {} -> {}'.format(origin,destination))
        sys.exit()
    
    print()
    print('***** Dataset Summary *****')
    print()
    print(df.describe())
    print()
    print('{} records found for rout {} -> {}'.format(len(df), origin, destination))
    

    X = df.drop(['flight_fare','origin', 'destination' ], axis=1)
    #Y = pd.DataFrame(mydata['flight_fare'])
    Y = df.drop(features2, axis=1)
    #Y = df['flight_fare']
    if(debug):
        print()
        print('X', X)
        print('Y', Y)

        pd.DataFrame(X).to_csv('X.csv')

    origin_cat = df['origin_cat'].iloc[0]
    destination_cat = df['destination_cat'].iloc[0]
    if(debug):
        print()
        print(origin_cat)
        print(destination_cat)
    ##execute_model(X, Y, debug)
    execute_model_to_predict_fare([oil_price, origin_cat, destination_cat],X, Y, model, debug)
    #execute_model_to_predict_fare([oil_price, quarter, year, origin_cat, destination_cat],X, Y, model, debug)

#==============================================================================================================
def predict_fare2(origin, destination, oil_price, model, quarter, year, dataset, debug):
    
    print()
    print('origin is ' , origin)
    print('destination is ', destination)
    print('dataset is ', dataset)
    print('Model is ', model)

    #features = ['origin', 'destination', 'oil_price', 'quarter', 'year', 'flight_fare']
    features = ['origin', 'destination', 'oil_price', 'flight_fare']

    features2 = ['origin', 'destination', 'oil_price']
    #featuresToDrop = ['origin', 'destination','distance', 'quarter', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'flight_fare']
    target = ['flight_fare']
    mydata = pd.read_csv(dataset, usecols=features)

    origin_encoded = pd.DataFrame(pd.read_csv('origin.csv'))
    destination_encoded = pd.DataFrame(pd.read_csv('destination.csv'))
    #print(origin_encoded['Origin_Encoded'])
    if(debug):
        print(origin_encoded)
        print(destination_encoded)

    df = pd.DataFrame(mydata)
    #df_ = pd.DataFrame(mydata)

    df_origin = df['origin']
    arr_origin = np.array(df_origin)
   

    df = df.loc[df['origin'] == origin]
  

   # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~> ",origin_encoded)
    if(len(df) == 0):
        print('entered origin not found')
        sys.exit()
    
    df = df.loc[df['destination'] == destination]
    if(len(df) == 0):
        print('no records found for the entered rout {} -> {}'.format(origin,destination))
        sys.exit()
    
    origin_encoded_ = origin_encoded.loc[origin_encoded['Origin'] == origin]
    destination_encoded_ = destination_encoded.loc[destination_encoded['Destination'] == destination]
    
    if(debug):
        print(origin_encoded_.iat[0,1])
        print(destination_encoded_.iat[0,1])

    print()
    print('***** Dataset Summary *****')
    print()
    print(df.describe())
    print()
    print('{} records found for rout {} -> {}'.format(len(df), origin, destination))
    

    X = df.drop(['flight_fare','origin', 'destination' ], axis=1)
    X['origin'] = origin_encoded['Origin_Encoded']
    X['destination'] = destination_encoded['Destination_Encoded']
    #Y = pd.DataFrame(mydata['flight_fare'])
    Y = df.drop(features2, axis=1)
    #Y = df['flight_fare']
    if(debug):
        print()
        print('X', X)
        print('Y', Y)

        pd.DataFrame(X).to_csv('X.csv')

    #origin_cat = df['origin_cat'].iloc[0]
    #destination_cat = df['destination_cat'].iloc[0]
    if(debug):
        print()
        #print(origin_cat)
        #print(destination_cat)
    ##execute_model(X, Y, debug)
    #execute_model_to_predict_fare([oil_price, origin_encoded_.iat[0,1], destination_encoded_.iat[0,1]],X, Y, debug)
    #execute_model_to_predict_fare([oil_price, quarter, year, origin_cat, destination_cat],X, Y, debug)

#==============================================================================================================
def predict_fare3(origin, destination, oil_price, model,quarter, year,dataset, debug):

    print()
    logToTrace('\n')
    print('origin is ' , origin)
    logToTrace('origin is ' + origin)
    print('destination is ', destination)
    logToTrace('destination is '+ destination)
    print('dataset is ', dataset)
    logToTrace('dataset is '+ dataset)
    print('Model is ', get_model(model))
    logToTrace('Model is '+ str(get_model(model)))

    if(encoding == 0):
        print('Encoding is Label Encoding')
        logToTrace('Encoding is Label Encoding')
    else:
        print('Encoding is One-Hot Encoding')
        logToTrace('Encoding is One-Hot Encoding')

    features = ['origin', 'destination', 'oil_price', 'quarter', 'year', 'flight_fare']
    #features = ['origin', 'destination', 'oil_price', 'flight_fare']
    features2 = ['origin', 'destination', 'oil_price']
    featuresToEncode = ['origin','destination', 'quarter', 'year']
    #featuresToDrop = ['origin', 'destination', 'quarter', 'year', 'flight_fare']


    #featuresToDrop = ['origin', 'destination','distance', 'quarter', 'origin_population', 'destination_population' ,'passengers', 'flights' ,'seats', 'flight_fare']
    target = 'flight_fare'
    mydata = pd.read_csv(dataset, usecols=features)


    df = pd.DataFrame(mydata)
    df = pd.get_dummies(df, columns=featuresToEncode)


    featuresToDrop = df.columns.values.tolist()
    featuresToDrop.remove(target)
    if(debug):
        pd.DataFrame(df).to_csv('_X.csv')
        print("Headers: ", featuresToDrop)

    df = df.loc[df['origin_'+origin] == 1]
    if(len(df) == 0):
        print('entered origin not found')
        sys.exit()
    
    df = df.loc[df['destination_'+destination] == 1]
    if(len(df) == 0):
        print('no records found for the entered rout {} -> {}'.format(origin,destination))
        sys.exit()
    
    user_input = df
    if(debug):
        print()
        print('user_input before =',user_input)
        print('oil price before', user_input['oil_price'])

    user_input = user_input.loc[user_input['year_'+str(year)] == 1]
    if(len(user_input) == 0):
        print('entered Year not found')
        sys.exit()
    
    user_input = user_input.loc[user_input['quarter_'+str(quarter)] == 1]
    if(len(user_input) == 0):
        print('entered quarter not found')
        sys.exit()

    print()
    print('***** Dataset Summary *****')
    logToTrace('\n')
    logToTrace('***** Dataset Summary *****')
    logToTrace('\n')

    print()
    print(df.describe())
    logToTrace(str(df.describe()))
    
    logToTrace('\n')
    logToTrace('{} records found for rout {} -> {}'.format(len(df), origin, destination))

    X = df.drop([target], axis=1)
    user_input = user_input.drop([target], axis=1)
    user_input.loc[user_input['oil_price'] == user_input['oil_price'], 'oil_price'] = oil_price

    #Y = pd.DataFrame(mydata['flight_fare'])
    Y = df.drop(featuresToDrop, axis=1)
    #Y = df['flight_fare']
    if(debug):
        print()
        print('X', X)
        print('Y', Y)
        print('user_input updated =',user_input)
        print('oil price after', user_input['oil_price'])
        pd.DataFrame(X).to_csv('X.csv')
        pd.DataFrame(user_input).to_csv('user_input.csv')

    #origin_cat = df['origin_cat'].iloc[0]
    #destination_cat = df['destination_cat'].iloc[0]
    if(debug):
        print()
    #    print(origin_cat)
     #   print(destination_cat)
    ##execute_model(X, Y, debug)
    execute_model_to_predict_fare(user_input,X, Y, model, debug)
    #execute_model_to_predict_fare([oil_price, quarter, year, origin_cat, destination_cat],X, Y, model, debug)
#==============================================================================================================

def get_model(model):
    
    #print('model = ', model)
    if(model == 0):
        #print('LinearRegression is selected')
        return LinearRegression()
    elif(model == 1):
        #print('Lasso is selected')
        return Lasso()
    else:
        #print('Ridge is selected')    
        return Ridge()
#==============================================================================================================

def logToTrace(str):
    filename = 'trace-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + '.txt'
    f = open(filename,'a')
    f.write('\n' + str)
    f.close()
#==============================================================================================================



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Mining Project')


    parser.add_argument("--dataset", type=str, 
                            help="dataset name",
                            default="DS1.CSV", 
                            required=False)        
    parser.add_argument("--debug", type=bool, 
                            default=False, 
                            required=False) 

    args = parser.parse_args()
    
    choice= int(input('Choose an operation:'))
    model = int(input('Enter model type: '))
    encoding = int(input('Enter encoding type: '))
   #
    if(choice == 1):
        prepare_model_with_encoding(model,encoding, args.dataset,args.debug)
    elif (choice == 2): 
        origin = input('Enter origin airport code: ')
        destination = input('Enter destination airport code: ')
        prepare_model_with_origin_and_destination_with_encoding(origin.upper(), destination.upper(), model, encoding, args.dataset, args.debug)
    elif(choice == 3): 
        origin = input('Enter the origin airport code: ')
        destination = input('Enter the destination airport code: ')
        oil_price = int(input('Enter the oil price: '))
        year = int(input('Enter the year: '))
        quarter = int(input('Enter the quarter: '))
        #predict_fare(origin.upper(), destination.upper(), oil_price, model, args.dataset, args.debug)
        predict_fare3(origin.upper(), destination.upper(), oil_price, model, quarter, year, args.dataset, args.debug)

        #predict_fare2(origin.upper(), destination.upper(), oil_price, quarter, year, model, args.dataset, args.debug)

    elif(choice == 99): # option 99 is for testing :)
        origin = 'ABQ'##input('Enter the origin airport code: ')
        destination = 'BWI'## input('Enter the destination airport code: ')
        oil_price = 33#int(input('Enter the oil price: '))
        year = 2003 #int(input('Enter the year: '))
        quarter = 3 # int(input('Enter the quarter: '))
        predict_fare3(origin.upper(), destination.upper(), oil_price, model, quarter, year, args.dataset, args.debug)


