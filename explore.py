import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression


def explore_nums(train, nums):
    '''
    This function takes in:
            train = train DataFrame
            nums = numerical columns (as a list of strings)
    '''
    for col in nums:
        sns.histplot(x=col, data=train)
        plt.show()
    
    
def select_kbest(X_train, y_train, k_features):
    '''
    This function takes in X_train, y_train, and the number of features
    to select and returns the names of the selected features using SelectKBest
    from sklearn. 
    '''
    kbest = SelectKBest(f_regression, k=k_features)
    kbest.fit(X_train, y_train)
    
    print(X_train.columns[kbest.get_support()].tolist())
    
    
def select_rfe(X_train, y_train, k_features):
    '''
    This function takes in X_train, y_train, and the number of features
    to select and returns the names of the selected features using Recursive
    Feature Elimination from sklearn. 
    '''
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=k_features)
    rfe.fit(X_train, y_train)
    
    print(X_train.columns[rfe.support_].tolist())
    
    
def select_sfs(X_train, y_train, k_features):
    '''
    This function takes in X_train, y_train, and the number of features
    to select and returns the names of the selected features using Sequential
    Feature Selector from sklearn. 
    '''
    model = LinearRegression()
    sfs = SequentialFeatureSelector(model, n_features_to_select=k_features)
    sfs.fit(X_train, y_train)
    
    print(X_train.columns[sfs.support_].tolist())
    
    
def question1_viz(train):
    plt.figure(figsize=(12,8))
    sns.histplot(data=train, x='value')
    plt.title('Distribution of Home Values')
    plt.show()
    
    print()
    
    plt.figure(figsize=(12,8))

    sns.histplot(data=train[train.area > train.area.median()], x='value')
    sns.histplot(data=train[train.area <= train.area.median()], x='value', color='#3f6f21')

    above = train[train.area > train.area.median()].value.median()
    plt.axvline(x=above, color='darkblue', linestyle='--')
    below = train[train.area <= train.area.median()].value.median()
    plt.axvline(x=below, color='darkgreen', linestyle='--')
    middle = train.value.median()
    plt.axvline(x=middle, color='red', linestyle='--')

    plt.title('More Space has More Value')
    plt.show()
    

def question2_viz(train):
    plt.figure(figsize=(12,8))
    sns.histplot(data=train[train.beds == 0], x='value', label = '0 bedrooms', color='#eef6ec')
    plt.axvline(x=train[train.beds == 0].value.median(), color='#eef6ec', linestyle='--')
    sns.histplot(data=train[train.beds == 1], x='value', label = '1 bedroom', color='#7bc86c')
    plt.axvline(x=train[train.beds == 1].value.median(), color='#7bc86c', linestyle='--')
    sns.histplot(data=train[train.beds == 2], x='value', label = '2 bedrooms', color='#519839')
    plt.axvline(x=train[train.beds == 2].value.median(), color='#519839', linestyle='--')
    sns.histplot(data=train[train.beds == 3], x='value', label = '3 bedrooms', color='#3f6f21')
    plt.axvline(x=train[train.beds == 3].value.median(), color='#3f6f21', linestyle='--')
    sns.histplot(data=train[train.beds == 4], x='value', label = '4 bedrooms', color='#06373A')
    plt.axvline(x=train[train.beds == 4].value.median(), color='#06373A', linestyle='--')
    sns.histplot(data=train[train.beds == 5], x='value', label = '5 bedrooms', color='black')
    plt.axvline(x=train[train.beds == 5].value.median(), color='black', linestyle='--')

    plt.title('Value Increases as Number of Bedrooms Increases')
    plt.legend()
    plt.show()
    
    print()
    
    plt.figure(figsize=(12,8))
    sns.histplot(data=train[train.baths == 0], x='value', label = '0 bathrooms', color='#eef6ec')
    plt.axvline(x=train[train.baths == 0].value.median(), color='#eef6ec', linestyle='--')
    sns.histplot(data=train[train.baths == 1], x='value', label = '1 bathroom', color='#7bc86c')
    plt.axvline(x=train[train.baths == 1].value.median(), color='#7bc86c', linestyle='--')
    sns.histplot(data=train[train.baths == 2], x='value', label = '2 bathrooms', color='#519839')
    plt.axvline(x=train[train.baths == 2].value.median(), color='#519839', linestyle='--')
    sns.histplot(data=train[train.baths == 3], x='value', label = '3 bathrooms', color='#3f6f21')
    plt.axvline(x=train[train.baths == 3].value.median(), color='#3f6f21', linestyle='--')
    sns.histplot(data=train[train.baths == 4], x='value', label = '4 bathrooms', color='#06373A')
    plt.axvline(x=train[train.baths == 4].value.median(), color='#06373A', linestyle='--')
    sns.histplot(data=train[train.baths == 5], x='value', label = '5 bathrooms', color='black')
    plt.axvline(x=train[train.baths == 5].value.median(), color='black', linestyle='--')


    plt.ylim(0,700)
    plt.title('Value Increases as Number of Bathrooms Increases')
    plt.legend()
    plt.show()
    
    
def question2b_viz(train):
    plt.figure(figsize=(12,8))

    sns.kdeplot(data=train, x='beds', y='baths')
    plt.xlim(0,7)
    plt.ylim(0,5)
    plt.arrow(1.9,.4,3,3, head_width=.2, color='red')
    
    plt.title('Bathrooms and Bedrooms Most Often Increase Together')
    plt.show()
    

def question3_viz(train):
    train['brackets'] = pd.cut(train.value, 10, labels=[1,2,3,4,5,6,7,8,9,10])
    
    plt.figure(figsize=(12,8))
    sns.histplot(data=train, x='year_built', alpha=.8, hue='location',hue_order=['Ventura', 'Orange', 'Los Angeles'])
    plt.axvline(x=train[train.location == 'Los Angeles'].year_built.mean(), color='green', linestyle='--')
    plt.axvline(x=train[train.location == 'Orange'].year_built.mean(), color='orange', linestyle='--')
    plt.axvline(x=train[train.location == 'Ventura'].year_built.mean(), color='blue', linestyle='--')
    plt.title('Los Angeles is Older than Other Locations')
    plt.show()
    
    print()
    
    plt.figure(figsize=(12,8))
    sns.histplot(data=train, x='value', alpha=.8, hue='location', hue_order=['Ventura', 'Orange', 'Los Angeles'])
    plt.axvline(x=train[train.location == 'Los Angeles'].value.median(), color='green', linestyle='--')
    plt.axvline(x=train[train.location == 'Orange'].value.median(), color='orange', linestyle='--')
    plt.axvline(x=train[train.location == 'Ventura'].value.median(), color='blue', linestyle='--')
    plt.title('Los Angeles is Cheaper than Other Locations')
    plt.show()
    
    print()
    
    plt.figure(figsize=(12,12))
    sns.scatterplot(data=train, x='long', y='lat', hue='location', hue_order=['Ventura', 'Orange', 'Los Angeles'])
    plt.title('Visualization of Homes in County Lines')
    plt.show()
    
    print()
    
    plt.figure(figsize=(12,12))
    sns.scatterplot(data=train, x='long', y='lat', hue='brackets', palette='rocket_r')
    plt.title('Los Angeles has Large Pocket of Low Value Homes')
    plt.show()


def hyp1_test(train):
    # Create the samples
    area_above = train[train.area > train.area.median()].value
    area_below = train[train.area <= train.area.median()].value

    # Set alpha
    α = 0.05

    # Check for equal variances
    s, pval = stats.levene(area_above, area_below)

    # Run the two-sample, one-tail T-test.
    # Use the results from checking for equal variances to set equal_var
    t, p = stats.ttest_ind(area_above, area_below, equal_var=(pval >= α))

    # Evaluate results based on the t-statistic and the p-value
    if p/2 < α and t > 0:
        print('''Reject the Null Hypothesis.
        
Findings suggest there is more value in homes with above median area than homes with below median area.''')
    else:
        print('''Fail to reject the Null Hypothesis.
        
Findings sugget there is more value in homes with below median area than homes with above medain area.''')
        
        
def hyp2_test(train):
    # Create the samples
    baths_above = train[(train.baths > train.baths.median())&(train.beds < train.beds.median())].value
    baths_below = train[(train.baths < train.baths.median())&(train.beds > train.beds.median())].value

    # Set alpha
    α = 0.05

    # Check for equal variances
    s, pval = stats.levene(baths_above, baths_below)

    # Run the two-sample, one-tail T-test.
    # Use the results from checking for equal variances to set equal_var
    t, p = stats.ttest_ind(baths_above, baths_below, equal_var=(pval >= α))

    # Evaluate results based on the t-statistic and the p-value
    if p/2 < α and t > 0:
        print('''Reject the Null Hypothesis.
        
Findings suggest there is more value in homes with above median bathrooms and below median bedrooms than homes with below median bathrooms and above median bedrooms.''')
    else:
        print('''Fail to reject the Null Hypothesis.
        
Findings suggest there is more value in homes with below median bathrooms and above median bedrooms than homes with above median bathrooms and below median bedrooms.''')
        
        
def hyp2b_test(train):
    # Set alpha
    α = 0.05

    # Create observed data
    observed = pd.crosstab(train.beds, train.baths)

    # Run chi-square test
    chi2,pval,degf,expected = stats.chi2_contingency(observed)

    # Evaluate results by comparing the p-value with alpha
    if pval < α:
        print('''Reject the Null Hypothesis.
        
Findings suggest there is an association between bedrooms and bathrooms.''')
    else:
        print('''Fail to reject the Null Hypothesis.
        
Findings suggest there is not an association between bedrooms and bathrooms.''')
        
        
def hyp3_test(train):
    # Create the samples
    LA_homes = train[train.location == 'Los Angeles'].value
    VenturaOrange_homes = train[(train.location == 'Orange')|(train.location == 'Ventura') ].value

    # Set alpha
    α = 0.05

    # Check for equal variances
    s, pval = stats.levene(LA_homes, VenturaOrange_homes)

    # Run the two-sample, one-tail T-test.
    # Use the results from checking for equal variances to set equal_var
    t, p = stats.ttest_ind(LA_homes, VenturaOrange_homes, equal_var=(pval >= α))

    # Evaluate results based on the t-statistic and the p-value
    if p/2 < α and t < 0:
        print('''Reject the Null Hypothesis.
        
Findings suggest there is less value in Los Angeles homes than homes in Ventura or Orange.''')
    else:
        print('''Fail to reject the Null Hypothesis.
        
Findings suggest there is greater than or equal value in Los Angeles homes than homes in Ventura or Orange.''')