# Zillow Regression Project
by: Morgan Cross

This project is designed to identify key features and build a regression model to best predict a home's tax assessed value. This report will interchagably use 'cost' or 'value' to refer to a home's tax assessed value.  

-----
## Project Overview:

#### Objectives:
- Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook Final Report.
- Create modules (wrangle.py) that make the process repeateable and the report (notebook) easier to read and follow.
- Ask exploratory questions of the data that will help you understand more about the attributes and drivers of housing cost. Answer questions through charts and statistical tests.
- Construct a model to predict cost using regression techniques, and make predictions for a group of single family residences.
- Refine work into a Report, in the form of a jupyter notebook, that you will walk through in a 5 minute presentation to a group of collegues and managers about the work you did, why, goals, what you found, your methdologies, and your conclusions.
- Be prepared to answer panel questions about your code, process, findings and key takeaways, and model.

#### Business Goals:
- Find features that drive single family residence home cost.
- Construct a ML regression model that best predicts cost.
- Deliver a report that a non-data scientist can read through and understand what steps were taken, why and what was the outcome?

#### Audience:
- Zillow data science team

#### Project Deliverables:
- this README.md walking through the project details
- final_report.ipynb displaying the process, findings, models, key takeaways, recommendation and conclusion
- wrangle.py with all acquire and prepare functions
- working_report.ipynb showing all work throughout the pipeline

-----
## Executive Summary:
Goals:
- Identify drivers of home value
- Build a model to best predict single family residence value
- Minimize Root Square Mean Error (RSME) in order to best predict home value

Key Findings:
- Location data and inside the home area data is the most impactful for predicting home value. 
- All models (LarLasso, Quadratic Linear Regression, Cubic Linear Regression) predicted home value better than the baseline, but not by much.

Takeaways:
 - More in home features and/or quality of life by location data would greatly improve the model. 
 - My best model, Quadratic Linear Regression, only reduced the baseline error by \$30,000 or 13% of total baseline error. 

Recommendations:
- Develop a flow of how a home is assessed. Home tax value assessors have a policy and procedure they must follow. Being able to use their assessment process in predicting this value would be essential to building better models moving forward.
- Track assigned school ratings and nearby crime rates. These features directly impact quality of life for most single family residence buyers. 

-----
## Data Dictionary:
| Target | Type | Description |
| ---- | ---- | ---- |
| value | int | The assessed tax value amount of the home |


| Feature Name | Type | Description |
| ---- | ---- | ---- |
| area | float | Sum of square feet in the home |
| baths | float | Count of bathrooms in the home |
| beds | float | Count of bedrooms in the home |
| decade | int | The decade the home was built in |
| extras | float | Sum of the home's bathrooms, bedrooms, stories, pool, and if it has a garage |
| garage | int | Sum of square feet in the garage |
| half_bath | int | 1 if the home has a half bath, 0 if not |
| lat | float | The home's geographical latitude |
| lat_long | float | The home's latitude divided by its longitude |
| living_space | float | The home area in sqft minus 132sqft per bedroom and 40sqft per bathroom (average sqft per respective room) |
| location | object | The human-readable county name the home is in |
| long | float | The home's geographical longitude |
| los_angeles | int | 1 if the home is in Los Angeles County, 0 if not | 
| lot_size | float | Sum of square feet of the piece of land the home is on |
| orange | int | 1 if the home is in Orange County, 0 if not |
| pool | int | 1 if the home has a pool, 0 if not |
| stories | int | Count of how many levels or stories the home has |
| ventura | int | 1 if the home is in Ventura County, 0 if not|
| yard_size | float | The lot size minus the home area in sqft |
| year_built | float | The year the home was built |
| zipcode | float | The US postal service 5-digit code for the home's location |

-----
## Planning
 - Create deliverables:
     - README
     - final_report.ipynb
     - working_report.ipynb
 - Build functional wrangle.py, explore.py, and model.py files
 - Acquire the data from the Code Up database via the wrangle.acquire functions
 - Prepare and split the data via the wrangle.prepare functions
 - Explore the data and define hypothesis. Run the appropriate statistical tests in order to accept or reject each null hypothesis. Document findings and takeaways.
 - Create a baseline model in predicting home cost and document the RSME.
 - Fit and train three (3) regression models to predict cost on the train dataset.
 - Evaluate the models by comparing the train and validation data.
 - Select the best model and evaluate it on the train data.
 - Develop and document all findings, takeaways, recommendations and next steps. 

-----
## Data Aquisition and Preparation
Files used:
 - wrangle.py

Steps taken:
In this step, I started by calling my acquire_zillow function from wrangle.py. This function:
- grabs the data from the CodeUp database via a SQL query
- creates a local CSV of the table, if not already saved locally

Next, I called my prepare_zillow function from wrangle.py. This funciton:
- renames columns
- handles nulls and outliers
- feature engineers
- splits the data into train, validate, and test datasets
- creates a scaled DataFrame pre-set for modeling later

What happened to each feature?
- columns are renamed to be more human readable
- drops all homes with missing data
- creates dummies for the counties (Los Angeles, Orange, and Ventura)
- removes outliers for lot_size, value, and area using quantiles (see prepare_zillow in wrangle.py for specifics)
- removes a single zipcode outlier
- feature engineers:
    - decade -> bins year_built into 10 year segments
    - yard_size -> subtracts home area from the lot size
    - living_space -> subtract the average bedroom size and bathroom size from the home area
    - half_bath -> creates a True/False (1/0) tag for homes with a half bathroom
- The following features were scaled using the MinMaxScaler:
    - beds
    - baths
    - area
    - lot_size
    - year_built
    - zipcode
    - yard_size
    - living_space
    - half_bath

Takeaways:
- I feature engineered yard_size and living_space to better assess the aspect ratio of these features. More yard or living area may play a role in home value. 
- I feature engineered decades for the purpose of visualization during explore
- I feature engineered half_bath to isolate homes with this feature. 
- Before moving to exploration, I split the data into train, validate, and test datasets. Lastly, I created scaled versions of these datasets in preparation for modeling.

-----
## Data Exploration
Files used:
- explore.py

Questions Addressed:
1. Do larger homes have more value?
2. Is there more value in bedrooms or bathrooms?
3. Is the amount of bedrooms and bathrooms related?
4. What is location's role in a home's value?

### Test 1: T-Test - Above Median Home Area Value vs Below Median Home Area Value
- A T-Test evaluates if there is a difference in the means of two continuous variables. This test is looking at a two samples and one tail.
- This test returns a p-value and a t-statistic.
- This test will compare the mean value of homes under the median area and the mean value of homes above the median area.
- Confidence level is 95%
- Alpha is 0.05

Hypothesis:
 - The null hypothesis is homes with above median area have less than or equal value to homes with below median area.
 - The alternate hypothesis is homes with above median area have greater value to homes with below median area.

Results: 
- p-value is less than alpha
- t-statistic is positive
- I rejected the Null Hypothesis, suggesting there is more value in homes with above median area than homes with below median area.

### Test 2: T-Test - Above Median Bathrooms/Below Median Bedrooms Home Value vs. Below Median Bathrooms/Below Median Bedrooms Home Value
- A T-Test evaluates if there is a difference in the means of two continuous variables. This test is looking at a two samples and one tail.
- This test returns a p-value and a t-statistic.
- This test will compare the mean value of homes with above median bathrooms and below median bedrooms against the mean value of homes with below median bathrooms and above median bedrooms.
- Confidence level is 95%
- Alpha is 0.05

Hypothesis:
 - The null hypothesis is homes with above median bathrooms and below median bedrooms have lower or equal value than homes with below median bathrooms and above median bedrooms.
 - The alternate hypothesis is homes with above median bathrooms and below median bedrooms have greater value than homes with below median bathrooms and above median bedrooms.

Results: 
- p-value is less than alpha
- t-statistic is positive
- I rejected the Null Hypothesis, suggesting there is more value in homes with above median bathrooms and below median bedrooms than homes with below median bathrooms and above median bedrooms.

### Test 3: Chi-Square - Bedrooms vs. Bathrooms
- This test evaluates if there is an association between two categorical variables.
- This test returns a chi2-value, a p-value, the degrees of freedom, and the expected outcome.
- This test will compare the count of bedrooms and the count of bathrooms.
- Confidence level is 95%
- Alpha is 0.05

Hypothesis:
- The null hypothesis is there is no association between count of bedrooms and count of bathrooms.
- The alternative hypothesis is there is an association between count of bedrooms and count of bathrooms.

Results: 
- p-value is less than alpha
- I rejected the Null Hypothesis, suggesting there is an association between count of bedrooms and count of bathrooms.

### Test 4: T-Test - Los Angeles Home Value vs. Orange and Ventura Home Value
- A T-Test evaluates if there is a difference in the means of two continuous variables. This test is looking at a two samples and one tail.
- This test returns a p-value and a t-statistic.
- This test will compare the mean value of homes with above median bathrooms and below median bedrooms against the mean value of homes with below median bathrooms and above median bedrooms.
- Confidence level is 95%
- Alpha is 0.05

Hypothesis:
 - The null hypothesis is the value mean of Los Angeles homes is greater than or equal to the value mean of Ventura and Orange homes.
 - The alternate hypothesis is the value mean of Los Angeles homes is less than the value mean of Ventura and Orange homes.

Results: 
- p-value is less than alpha
- t-statistic is negative
- I rejected the Null Hypothesis, suggesting there is less value in Los Angeles homes compared to Ventura and Orange homes.

### Takeaways from exploration:
- There is more value in homes with above median area.
- There is more value in homes with above median bathrooms and below median bedrooms than the opposite. 
- There is an association between bedrooms and bathrooms.
- There is less value in Los Angeles homes than in Orange or Ventura.

-----
## Modeling:
### Model Preparation:

### Baseline:
Baseline Results
- The baseline model was built off of the train dataset's mean for home value at \$362,670.
- The baseline R<sup>2</sup> is 0.
- The baseline RMSE is \$235,784.

| Features Kept | Features Dropped |
| ---- | ---- |
| baths | location |
| beds | decade |
| area | yard_space |
| lot_size |  |
| zipcode |  |
| lat |  |
| long |  |
| lat_long |  |
| los_angeles |  |
| orange |  |
| ventura |  |
| living_space |  |
| half_bath |  |
| pool |  |
| stories |  |
| garage |  |
| extras |  |

### Modeling Results:
| Model | RMSE Train (dollars) | R<sup>2</sup> Train | RMSE Validate (dollars) | R<sup>2</sup> Validate |
| ---- | ---- | ---- | ---- | ---- |
| Baseline | 235,784 | 0 | 235,784 | 0 |
| LarsLasso alpha=1 | 206,664 | 0.257 | 203,881 | 0.252 |
| Quadratic Linear Regression | 198,529 | 0.314 | 196,592 | 0.305 |
| Cubic Linear Regression | 190,497 | 0.369 | 194,693 | .319 |

The Quadratic Linear Regression model minimized RMSE the most.

### Testing the Model:
| Model | RMSE (dollars) | R<sup>2</sup> |
| ---- | ---- | ---- |
| Train | 198,529 | 0.314 |
| Validate | 196,592 | 0.305 |
| Test | 203,822 | 0.295 |

-----
## Conclusion:
Home value is assessed through a myraid of metrics taken about the home. Location and area based information is the most valuable, but it is not enough. My best model only reduced the root mean squared error by \$35,000 from the baseline. The model's error of \$200,000  still covers most of a standard deviation and is not a good model to use in production.

#### Recommendations: 
- Add data or begin tracking school ratings and crime ratings for each neighborhood. I predict sections of homes with high school ratings and low crime rates will value for more than homes with low school ratings or high crime rates.
- Develop a flow of how a home is assessed. Home tax value assessors have a policy and procedure they must follow. Being able to use their assessment process in predicting this value would be essential to building better models moving forward. 

#### Next Steps:
- Add other home features such as pool, garage, stories, and fireplaces. These variables could greatly impact the value of a home.
- Feature engineer more detailed depictions of the use of the area inside the home. Specifically determine the kitchen vs living area sections of the home and see how this effects the model.
- Develop a model using different machine learning techniques focused on geographical distance. Home value is often geographically clusered as depicted in our finidngs. 

-----
## How to Recreate:
1. Utilize the following files found in this repository:
- final_report.ipynb
- wrangle.py
- explore.py
- evaluate.py

2. To access the data, you will need credentials to access to the CodeUp database saved in an env.py file.
Create your personal env.py file with your credentials saved as strings. (user, password, host)

3. Run the final_report.ipynb notebook.