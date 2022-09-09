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

def plot_residuals(y_train, yhat):
    '''
    This function takes in the actual and predicted values and 
    plots them. 
    '''
    df = pd.DataFrame({'y_train':y_train,'yhat':yhat})

    
    df['residuals'] = df.y_train - df.yhat
    sns.scatterplot(data=df, x=df.y_train, y=df.residuals)
    plt.xlabel('x = y_train')
    plt.ylabel('y = residuals')
    plt.title('OLS linear model residuals')
    plt.show()
    

def regression_errors(y_train, yhat):
    '''
    This function takes in the actual (y_train) and predicted (yhat) values
    and returns the MSE, SSE, RMSE, TSS, and EES values
    '''
    df = pd.DataFrame({'y_train':y_train,'yhat':yhat})
    df['baseline'] = df.y_train.mean()
    
    MSE = mean_squared_error(df.y_train, df.yhat)
    SSE = MSE*len(df)
    RMSE = mean_squared_error(df.y_train, df.yhat, squared=False)
    TSS = mean_squared_error(df.y_train, df.baseline)*len(df)
    EES = TSS - SSE
        
    return MSE, SSE, RMSE, TSS, EES


def baseline_mean_errors(y_train):
    '''
    This function takes in the actual values (y_train) and returns
    the MSE, SSE, and RMSE for the baseline model.
    '''
    df = pd.DataFrame({'y_train':y_train})
    df['baseline'] = df.y_train.mean()
    
    MSE = mean_squared_error(df.y_train, df.baseline)
    SSE = MSE*len(df)
    RMSE = mean_squared_error(df.y_train, df.baseline, squared=False)
    
    return MSE, SSE, RMSE


def better_than_baseline(y_train, yhat):
    '''
    This function takes in the predicted and actual values and returns
    if the SSE, RMSE and R2 values are better for the model or baseline.
    '''

    MSE, SSE, RMSE, TSS, EES = regression_errors(y_train, yhat)

    MSE_baseline, SSE_baseline, RMSE_baseline = baseline_mean_errors(y_train)

    df = pd.DataFrame({'y_train':y_train,'yhat':yhat})
    df['baseline'] = df.y_train.mean()
    r2_baseline = r2_score(df.y_train, df.baseline)
    r2 = r2_score(df.y_train, df.yhat)

    if SSE < SSE_baseline:
        print(f'''The model SSE performs better than the baseline.
            Baseline SSE: {SSE_baseline}
            Model SSE: {SSE}''')
    else:
        print(f'''The baseline SSE performs better than the model.
            Baseline SSE: {SSE_baseline}
            Model SSE: {SSE}''')
    print()
        
    if RMSE < RMSE_baseline:
        print(f'''The model RMSE performs better than the baseline.
            Baseline RMSE: {RMSE_baseline}
            Model RMSE: {RMSE}''')
    else:
        print(f'''The baseline RMSE performs better than the model.
            Baseline RMSE: {RMSE_baseline}
            Model RMSE: {RMSE}''')
    print()
        
    if r2 > r2_baseline:
        print(f'''The model R2 performs better than the baseline.
            Baseline R2: {r2_baseline}
            Model R2: {r2}''')
    else:
        print(f'''The baseline R2 performs better than the model.
            Baseline R2: {r2_baseline}
            Model R2: {r2}''')
        

def model_setup(train_scaled, train, val_scaled, val, test_scaled, test):
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

    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test
        

def get_models_with_results(X_train, y_train, X_val, y_val):
    # Baseline Model
    pred_mean = y_train.value.mean()
    y_train['pred_mean'] = pred_mean
    y_val['pred_mean'] = pred_mean
    rmse_train = mean_squared_error(y_train.value, y_train.pred_mean, squared=False)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_mean, squared=False)

    metrics = pd.DataFrame(data=[{
        'model': 'baseline_mean',
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.value, y_val.pred_mean)}])

    # OLS Model
    # lm = LinearRegression()
    # lm.fit(X_train, y_train.value)
    # y_train['pred_lm'] = lm.predict(X_train)
    # rmse_train = mean_squared_error(y_train.value, y_train.pred_lm, squared=False)
    # y_val['pred_lm'] = lm.predict(X_val)
    # rmse_val = mean_squared_error(y_val.value, y_val.pred_lm, squared=False)

    # metrics = metrics.append({
    #     'model': 'OLS',
    #     'rmse_val': rmse_val,
    #     'r2_val': explained_variance_score(y_val.value, y_val.pred_lm)}, ignore_index=True)

    # LassoLars Model
    lars = LassoLars(alpha=1)
    lars.fit(X_train, y_train.value)
    y_train['pred_lars'] = lars.predict(X_train)
    rmse_train = mean_squared_error(y_train.value, y_train.pred_lars, squared=False)
    y_val['pred_lars'] = lars.predict(X_val)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_lars, squared=False)

    metrics = metrics.append({
        'model': 'LarsLasso, alpha 1',
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.value, y_val.pred_lars)}, ignore_index=True)

    # Tweedie Models
    # glm = TweedieRegressor(power=1, alpha=0)
    # glm.fit(X_train, y_train.value)
    # y_train['pred_glm'] = glm.predict(X_train)
    # rmse_train = mean_squared_error(y_train.value, y_train.pred_glm, squared=False)
    # y_val['pred_glm'] = glm.predict(X_val)
    # rmse_val = mean_squared_error(y_val.value, y_val.pred_glm, squared=False)

    # metrics = metrics.append({
    #     'model': 'poisson',
    #     'rmse_val': rmse_val,
    #     'r2_val': explained_variance_score(y_val.value, y_val.pred_glm)}, ignore_index=True)

    # glm = TweedieRegressor(power=2, alpha=0)
    # glm.fit(X_train, y_train.value)
    # y_train['pred_glm'] = glm.predict(X_train)
    # rmse_train = mean_squared_error(y_train.value, y_train.pred_glm, squared=False)
    # y_val['pred_glm'] = glm.predict(X_val)
    # rmse_val = mean_squared_error(y_val.value, y_val.pred_glm, squared=False)

    # metrics = metrics.append({
    #     'model': 'gamma',
    #     'rmse_val': rmse_val,
    #     'r2_val': explained_variance_score(y_val.value, y_val.pred_glm)}, ignore_index=True)

    # glm = TweedieRegressor(power=3, alpha=0)
    # glm.fit(X_train, y_train.value)
    # y_train['pred_glm'] = glm.predict(X_train)
    # rmse_train = mean_squared_error(y_train.value, y_train.pred_glm, squared=False)
    # y_val['pred_glm'] = glm.predict(X_val)
    # rmse_val = mean_squared_error(y_val.value, y_val.pred_glm, squared=False)

    # metrics = metrics.append({
    #     'model': 'inverse gaussian',
    #     'rmse_val': rmse_val,
    #     'r2_val': explained_variance_score(y_val.value, y_val.pred_glm)}, ignore_index=True)

    # Polynomial Models
    pf = PolynomialFeatures(degree=2)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)
    
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.value)

    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.value, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_lm2, squared=False)

    metrics = metrics.append({
        'model': 'd2',
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.value, y_val.pred_lm2)}, ignore_index=True)

    pf = PolynomialFeatures(degree=3)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)

    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.value)

    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.value, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_lm2, squared=False)

    metrics = metrics.append({
        'model': 'd3',
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.value, y_val.pred_lm2)}, ignore_index=True)

    # pf = PolynomialFeatures(degree=4)
    # X_train_d2 = pf.fit_transform(X_train)
    # X_val_d2 = pf.transform(X_val)
    # X_test_d2 = pf.transform(X_test)

    # lm2 = LinearRegression()
    # lm2.fit(X_train_d2, y_train.value)

    # y_train['pred_lm2'] = lm2.predict(X_train_d2)
    # rmse_train = mean_squared_error(y_train.value, y_train.pred_lm2, squared=False)
    # y_val['pred_lm2'] = lm2.predict(X_val_d2)
    # rmse_val = mean_squared_error(y_val.value, y_val.pred_lm2, squared=False)

    # metrics = metrics.append({
    #     'model': 'd4',
    #     'rmse_val': rmse_val,
    #     'r2_val': explained_variance_score(y_val.value, y_val.pred_lm2)}, ignore_index=True)

    return metrics


def run_best_test(X_train, y_train, X_val, y_val, X_test, y_test):
    pf = PolynomialFeatures(degree=3)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)
    X_test_d2 = pf.transform(X_test)

    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.value)

    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.value, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_lm2, squared=False)
    y_test['pred_lm2'] = lm2.predict(X_test_d2)
    rmse_test = mean_squared_error(y_test.value, y_test.pred_lm2, squared=False)

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