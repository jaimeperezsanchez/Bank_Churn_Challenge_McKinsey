# Setup -------------------------------------------------------------------
# Package installation
import numpy as np
import xgboost as xgb
import pandas as pd
import random
import os
import datetime
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load auxiliar metrics functions
from aux_metrics import get_uplift, get_feature_importance



# Set seed for reproducibility
random.seed(42)

# Set data path
project_path = "/Drive/_Intercambio/RobertoPerez/20190909_ayuda_recruiting_Andres/"


# Load data and create target variable (TASK 1) --------------------------------
## Task 1a: Read in data in .csv format
churn_file_name = "bankingchurn_data.csv"
df_churn = pd.read_csv(project_path + churn_file_name, na_values=['-'])

# Make quick cross-checks
#print("Head")
#print(df_churn.head())
#print("Summary")
#print(df_churn.describe())
#print("Str")
#print(df_churn.info())
#print("Types")
#print(Counter(df_churn.dtypes))


## Task 1b: Create dependent variable from contract_end date and remove contract_end date
df_churn['churn_flag'] = "no_churn"
df_churn.loc[~df_churn['contract_end'].isna(), 'churn_flag'] = "churn"# Should be a factor with values "churn" and "no_churn"

df_churn['age'] = (datetime.datetime.strptime('2018-01-26', '%Y-%m-%d') - \
                   pd.to_datetime(df_churn['date_of_birth']))/np.timedelta64(1, 'Y')


# Data transformations and feature creation (TASK 2) ---------------------------
# Check distribution of the target variable
df_churn[['contract_start','churn_flag']].groupby(['churn_flag']).count()

# remove variables not needed (dates, helper veriables, row index...)
df_churn = df_churn.drop(['contract_start', 'contract_end',  'date_of_birth', 'ZIP'], axis = 1)
# df_churn[['contract_end', 'churn_flag']].head()
# df_churn[['contract_start', 'date_of_birth', 'age']].head()

# Missing data imputation - Replace each missing value with an own category, e.g. "NA -> Not_available"
df_churn.loc[df_churn['profession'].isna(), 'profession'] = 'unknown'
#df_churn[['age','profession']].groupby(['profession']).count()

# Modelling (TASK 3)------------------------------------------------------------
df_churn['churn_flag'] = np.where(df_churn['churn_flag']=='churn', 1, 0)

# Option 1: Logistic regression with manual variable selection

logit_model = LogisticRegression()
var_select  = ["size_household", 
               "income_deposits_per_year", 
               "main_account_flag", 
               "insurance_house_premium_per_year", 
               "income_securities_per_year", 
               "cash_withdraws_per_month"]

# Split data: leave out evaluation set
x_train = df_churn[var_select] 
y_train = df_churn['churn_flag']
x_train, x_eval_logit, y_train, y_eval_logit = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
# Split training data again into training and test set
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# fit the model with data
logit_model.fit(x_train,y_train)

# Predictions on test set
predictions_logreg = logit_model.predict(x_test)

# Check accuracy of predictions
accuracy_logreg = confusion_matrix(y_test.values, predictions_logreg)

sum(np.diagonal(accuracy_logreg))/sum(sum(accuracy_logreg))

print(accuracy_logreg)

print(get_uplift(predictions_logreg, y_test.values)[['percentile','acc_uplift']].head())

# Option 2: Xgboost without variable selection (all numerical variables are included)
# Split data: leave out evaluation set
var_select = np.setdiff1d(df_churn.columns, ['churn_flag', 'profession', 'segment'])
x_train = df_churn[var_select].apply(pd.to_numeric, errors='coerce')
y_train = df_churn['churn_flag']
x_train, x_eval_xgb, y_train, y_eval_xgb = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
# Split training data again into training and test set
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Specify the kind of model to develop
xgb_reg = xgb.XGBRegressor(objective ='binary:logistic', 
                          colsample_bytree = 0.3, 
                          learning_rate = 0.1,
                          max_depth = 5, 
                          n_estimators = 350)

xgb_reg.fit(x_train, y_train)
preds = xgb_reg.predict(x_test)

# Check accuracy of predictions
accuracy_xgb = confusion_matrix(y_test.values, np.round(preds))
print(sum(np.diagonal(accuracy_xgb))/sum(sum(accuracy_xgb)))
print(accuracy_xgb)

# Obtain the most important variables
scores_xgb = xgb_reg.get_booster().get_score(importance_type='gain')
most_imp_xgb = pd.DataFrame({'feature':list(scores_xgb.keys()), 'gain':list(scores_xgb.values())}).sort_values(by = ['gain'], ascending = False)
most_imp_xgb['gain'] = most_imp_xgb['gain']/most_imp_xgb['gain'].sum()
most_impt_xgb = most_imp_xgb.reset_index().drop('index', axis =1)
print(most_imp_xgb.head())


# Evaluation of your model (TASK 4)---------------------------------------------------------------
# Evaluate your best model on the top decile of the evaluation dataset
# Option 1: Linear regression 
preds_eval_logit  = logit_model.predict(x_eval_logit)
accuracy_logit_eval = confusion_matrix(y_eval_logit.values, preds_eval_logit)
print(sum(np.diagonal(accuracy_logit_eval))/sum(sum(accuracy_logit_eval)))
print(get_uplift(preds_eval_logit, y_eval_logit.values)[['percentile','acc_uplift']].head())

# Option 2: Xgboost
preds_eval_xgb = xgb_reg.predict(x_eval_xgb)
accuracy_xgb_eval = confusion_matrix(y_eval_xgb.values, np.round(preds_eval_xgb))
sum(np.diagonal(accuracy_xgb_eval))/sum(sum(accuracy_xgb_eval))
print(get_uplift(preds_eval_xgb, y_eval_xgb.values)[['percentile','acc_uplift']].head())