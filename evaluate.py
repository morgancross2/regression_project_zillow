# standard ds imports
import numpy as np
import pandas as pd

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# for modeling and evaluation
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
        

def model_setup(train_scaled, train, val_scaled, val, test_scaled, test):
    ''' 
    This function takes in the train, validate, test datasets and their scaled counterparts
    and then returns the X and y objects with appropriate feature selection and ready for modeling. 
    '''
    # Set up X and y values for modeling
    X_train, y_train = train_scaled.drop(columns=[#'baths',
                                                 # 'beds',
                                                 # 'area',
                                                 # 'lot_size',
                                                 'value',
                                                 # 'year_built',
                                                 # 'zipcode',
                                                 # 'lat',
                                                 # 'long',
                                                 'location',
                                                 # 'los_angeles',
                                                 # 'orange',
                                                 # 'ventura',
                                                 'decade',
                                                 'yard_size',
                                                 # 'living_space',
                                                 # 'half_bath'
                                                 ]), train.value
    X_val, y_val = val_scaled.drop(columns=[#'baths',
                                                 # 'beds',
                                                 # 'area',
                                                 # 'lot_size',
                                                 'value',
                                                 # 'year_built',
                                                 # 'zipcode',
                                                 # 'lat',
                                                 # 'long',
                                                 'location',
                                                 # 'los_angeles',
                                                 # 'orange',
                                                 # 'ventura',
                                                 'decade',
                                                 'yard_size',
                                                 # 'living_space',
                                                 # 'half_bath'
                                                ]), val.value
    X_test, y_test = test_scaled.drop(columns=[#'baths',
                                                 # 'beds',
                                                 # 'area',
                                                 # 'lot_size',
                                                 'value',
                                                 # 'year_built',
                                                 # 'zipcode',
                                                 # 'lat',
                                                 # 'long',
                                                 'location',
                                                 # 'los_angeles',
                                                 # 'orange',
                                                 # 'ventura',
                                                 'decade',
                                                 'yard_size',
                                                 # 'living_space',
                                                 # 'half_bath'
                                              ]), test.value

    # make them into dataframes
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test
        

def get_models_with_results(X_train, y_train, X_val, y_val):
    ''' 
    This function takes in the X and y objects and then runs the following models:
    - Baseline model using y_train mean
    - LarsLasso model with alpha=1
    - Quadratic Linear Regression
    - Cubic Linear Regression
    
    Returns a DataFrame with the results.
    '''
    # Baseline Model
    # run the model
    pred_mean = y_train.value.mean()
    y_train['pred_mean'] = pred_mean
    y_val['pred_mean'] = pred_mean
    rmse_train = mean_squared_error(y_train.value, y_train.pred_mean, squared=False)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_mean, squared=False)

    # save the results
    metrics = pd.DataFrame(data=[{
        'model': 'baseline_mean',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.value, y_train.pred_mean),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.value, y_val.pred_mean)}])

    # LassoLars Model
    # run the model
    lars = LassoLars(alpha=1)
    lars.fit(X_train, y_train.value)
    y_train['pred_lars'] = lars.predict(X_train)
    rmse_train = mean_squared_error(y_train.value, y_train.pred_lars, squared=False)
    y_val['pred_lars'] = lars.predict(X_val)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_lars, squared=False)

    # save the results
    metrics = metrics.append({
        'model': 'LarsLasso, alpha 1',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.value, y_train.pred_lars),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.value, y_val.pred_lars)}, ignore_index=True)

    # Polynomial Models
    # set up the model
    pf = PolynomialFeatures(degree=2)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)
    
    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.value)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.value, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_lm2, squared=False)

    # save the results
    metrics = metrics.append({
        'model': 'd2',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.value, y_train.pred_lm2),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.value, y_val.pred_lm2)}, ignore_index=True)

    # set up the model
    pf = PolynomialFeatures(degree=3)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)

    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.value)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.value, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_lm2, squared=False)

    # save the results
    metrics = metrics.append({
        'model': 'd3',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.value, y_train.pred_lm2),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.value, y_val.pred_lm2)}, ignore_index=True)

    return metrics


def run_best_test(X_train, y_train, X_val, y_val, X_test, y_test):
    ''' 
    This function takes in the X and y objects and then runs and returns a DataFrame of
    results for the Quadradic Linear Regression Model. 
    '''
    # set up the model
    pf = PolynomialFeatures(degree=2)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)
    X_test_d2 = pf.transform(X_test)

    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.value)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.value, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_lm2, squared=False)
    y_test['pred_lm2'] = lm2.predict(X_test_d2)
    rmse_test = mean_squared_error(y_test.value, y_test.pred_lm2, squared=False)

    # save the results
    results = pd.DataFrame({'train': 
                               {'rmse': rmse_train, 
                                'r2': explained_variance_score(y_train.value, y_train.pred_lm2)},
                           'validate': 
                               {'rmse': rmse_val, 
                                'r2': explained_variance_score(y_val.value, y_val.pred_lm2)},
                           'test': 
                               {'rmse': rmse_test, 
                                'r2': explained_variance_score(y_test.value, y_test.pred_lm2)}
                          })
    
    return results