import numpy as np
import pandas as pd
import os
import env
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


def acquire_zillow():
    '''
    This function checks if the zillow data is saved locally. 
    If it is not local, this function reads the zillow data from 
    the CodeUp MySQL database and return it in a DataFrame.
    
    The prepare portion of this function removes outliers via 
    quantiles. 
        - Renames the features
        - Feature engineers a readable location
        - Feature engineers year_built into decade bins
        - Feature engineers tax_value percentiles(quadrants), split for location
    '''
    
    # Acquire
    # Set file name
    filename = 'zillow.csv'
    # if the file is saved locally... grab that
    if os.path.isfile(filename):
        df = pd.read_csv(filename).iloc[:,1:]
    # if the file is not local, pull it via SQL from the CodeUp database
    else:
        q = '''SELECT bathroomcnt, 
                        bedroomcnt, 
                        calculatedfinishedsquarefeet,
                        fips, 
                        lotsizesquarefeet, 
                        taxvaluedollarcnt,
                        yearbuilt,
                        regionidzip,
                        latitude,
                        longitude,
                        poolcnt,
                        numberofstories,
                        garagetotalsqft
                FROM properties_2017
                JOIN predictions_2017
                    USING (parcelid)
                WHERE transactiondate LIKE '2017%%'
                AND propertylandusetypeid IN (261, 279)
                ;
                '''
        df = pd.read_sql(q, env.conn('zillow'))
        # Save it locally for future use
        df.to_csv(filename)
    # return the file
    return df
        
def prepare_zillow(df):
    '''
    This function takes in the zillow DataFrame and completes the following
    before returning the DataFrame.
        - renames columns
        - handles nulls and outliers
        - feature engineers
        - splits the data into train, validate, and test datasets
        - creates a scaled DataFrame pre-set for modeling later
    '''
    
    # Prepare
    # rename columns
    df.columns = ['baths', 'beds', 'area', 'city', 'lot_size', 'value', 'year_built', 'zipcode', 'lat', 'long', 'pool', 'stories', 'garage_sqft']
        
    # remove outliers
    df.pool = df.pool.fillna(0)
    df.garage_sqft = df.garage_sqft.fillna(0)
    df.stories = df.stories.fillna(1)
    df = df.dropna()

    # create readable location column from city/fips codes
    df['location'] = df.city.map({6037: 'Los Angeles', 6059: 'Orange', 6111:'Ventura'})
    
    # create dummies for the cities and drop the fips/city column
    dummies = pd.get_dummies(df.city)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns='city')
    
    # rename columns so the dummies have human readable names, resave them as integer types
    df.columns = ['baths', 'beds', 'area', 'lot_size', 'value', 'year_built', 'zipcode', 'lat', 'long', 'pool', 'stories', 'garage_sqft', 'location', 'los_angeles', 'orange', 'ventura']
    df.los_angeles = df.los_angeles.astype(int)
    df.orange = df.orange.astype(int)
    df.ventura = df.ventura.astype(int)
    
    # remove outliers using quantiles
    things = ['lot_size', 'value', 'area']
    for col in things:
        q1,q3 = df[col].quantile([.25,.75])
        iqr = q3-q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr

        df = df[(df[col] > lower) & (df[col] < upper)]
    
    # Take year_built and bin into a decades feature
    df['decade'] = pd.cut(df.year_built, 
                         bins=[1800,1850,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020], 
                         labels=['1800', '1850', '1900', '1910', '1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010'])
    df.decade = df.decade.astype(int)
    
    # create a measurement in sqft for a home's yard size
    df['yard_size'] = df.lot_size - df.area
    
    # create a measurement in sqft for a home's living space, use average bedroom and average bathroom sqft
    df['living_space'] = df.area - df.beds*180 - df.baths*40 + df.garage_sqft
    
    # create a tag feature for homes with a half bathroom
    df['half_bath'] = df.baths%1 != 0
    df['half_bath'] = df.half_bath.map({True:1,
                                        False:0})
    
    # create a feature counting how many amenities or extras a home has
    df['extras'] = df.baths + df.beds + df.stories + df.pool + df.garage_sqft/df.garage_sqft
    df['extras'] = df.extras.fillna(0)
    
    # combo of latititude and longitude
    df['lat_long'] = df.lat/df.long

    # drop the one weird zipcode 
    df = df.drop(index=26047)
    
    # return it!
    return df

def split_data(df):
    '''
    Takes in a dataframe and target (as a string). Returns train, validate, and test subset 
    dataframes with the .2/.8 and .25/.75 splits to create a final .2/.2/.6 split between datasets
    '''
    # split the data into train and test. 
    train, test = train_test_split(df, test_size = .2, random_state=123)
    
    # split the train data into train and validate
    train, validate = train_test_split(train, test_size = .25, random_state=123)
    
    return train, validate, test


def scale_data(train, val, test, cols_to_scale):
    '''
    This function takes in train, validate, and test dataframes as well as a
    list of features to be scaled via the MinMaxScalar. It then returns the 
    scaled versions of train, validate, and test in new dataframes. 
    '''
    # create copies to not mess with the original dataframes
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    
    # create the scaler and fit it
    scaler = MinMaxScaler()
    scaler.fit(train[cols_to_scale])
    
    # use the scaler to scale the data and resave
    train_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(train[cols_to_scale]),
                                               columns = train[cols_to_scale].columns.values).set_index([train.index.values])
    val_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(val[cols_to_scale]),
                                               columns = val[cols_to_scale].columns.values).set_index([val.index.values])
    test_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(test[cols_to_scale]),
                                               columns = test[cols_to_scale].columns.values).set_index([test.index.values])
    
    return train_scaled, val_scaled, test_scaled

