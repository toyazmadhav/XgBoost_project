import imp
from unicodedata import numeric
from catboost import CatBoostClassifier
import wget
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import os, glob, pickle
import re
import zipfile
import io
from plot_utils import plot_for_all_cols
from matplotlib import pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve

from sklearn.ensemble import VotingClassifier


from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
from utils import *



# ********Constants****************************************************
numeric_cols = [
   'age', 
   'dailyrate',
   'distancefromhome',
   'employeenumber',
   'hourlyrate',
   'monthlyincome',
   'monthlyrate',
   'numcompaniesworked',
   'percentsalaryhike',
   'totalworkingyears',
   'yearsatcompany',
   'yearsincurrentrole',
   'yearssincelastpromotion',
   'yearswithcurrmanager',
   'trainingtimeslastyear'
]

cat_cols = [
    'businesstravel', 
    'department', 
    'educationfield',
    'gender', 
    'jobrole',
    'maritalstatus',
]
# https://stats.stackexchange.com/questions/430636/is-a-rating-in-a-set-range-a-categorical-or-numerical-variable#:~:text=It's%20categorical%2C%20specifically%20ordinal.,what%20you%20mean%20by%20measurement.
#  https://stackoverflow.com/questions/29528628/how-to-specify-a-variable-in-pandas-as-ordinal-categorical
optional_cat_cols = [
  'education',
  'environmentsatisfaction',
  'jobinvolvement',
  'joblevel',
  'jobsatisfaction',
  'numcompaniesworked',
  'performancerating',
  'stockoptionlevel',
  'trainingtimeslastyear',
  'worklifebalance'
],
binnable_cols = [
  'yearsatcompany',
  'yearsincurrentrole',
  'yearssincelastpromotion',
  'yearswithcurrmanager',
]
binary_cols = [
  'overtime'
]
cols_to_be_dropeed = [
  'employeecount',
  'over18',
  'standardhours',
  'employeenumber'
  ]

space= {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': hp.quniform('max_depth', 5, 15, 1),
    'n_estimators': hp.quniform('n_estimators', 5, 35, 1),
    'num_leaves': hp.quniform('num_leaves', 5, 50, 1),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'num_iterations': 32,
    'num_leaves': hp.quniform('num_leaves', 2, 20, 2),

    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    # 'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1), #alias "subsample"
    'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
    # 'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
    # 'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
    'verbose': -1,
    #the LGBM parameters docs list various aliases, and the LGBM implementation seems to complain about
    #the following not being used due to other params, so trying to silence the complaints by setting to None
    'subsample': None, #overridden by bagging_fraction
    # 'reg_alpha': None, #overridden by lambda_l1
    # 'reg_lambda': None, #overridden by lambda_l2
    'min_sum_hessian_in_leaf': None, #overrides min_child_weight
    'min_child_samples': None, #overridden by min_data_in_leaf
    # 'colsample_bytree': None, #overridden by feature_fraction
#        'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'min_child_weight': hp.loguniform('min_child_weight', -16, 5), #also aliases to min_sum_hessian
    #'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
}
# ***************************************************************************

def download_data():
  url = 'https://cdn.iisc.talentsprint.com/CDS/MiniProjects/hr_employee_attrition_train.csv'
  file_name = url.split("/")[-1]
  if(not os.path.exists(file_name)):
    wget.download(url, file_name)
  return file_name

def load_data():
  file_name = download_data()
  df = pd.read_csv(file_name)
  st.write("# Data Exploration")
  st.write("## Data frame head")
  st.dataframe(df)
  return df

def handle_outliers(df, colm):
    '''Change the values of outlier to upper and lower whisker values '''
    q1 = df.describe()[colm].loc["25%"]
    q3 = df.describe()[colm].loc["75%"]
    iqr = q3 - q1
    lower_bound = q1
    upper_bound = q3

    # upper_bound=df[colm].mean()+2*df[colm].std()
    # lower_bound=df[colm].mean()-2*df[colm].std()

    df.loc[df[colm] < lower_bound,colm] = lower_bound
    df.loc[df[colm] > upper_bound,colm] = upper_bound

def write_train_and_test_reports(X_train, X_test, y_train, y_test, rf_clf):
    y_pred = rf_clf.predict(X_test)
    st.write("## Testing Report")
    st.write( accuracy_score(y_test, y_pred))
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

    y_pred_train = rf_clf.predict(X_train)
    st.write("## Training Report")
    st.write( accuracy_score(y_train, y_pred_train))
    st.dataframe(pd.DataFrame(classification_report(y_train, y_pred_train, output_dict=True)))

    fpr, tpr, thresh = roc_curve(y_test, y_pred, pos_label=1)
    random_probs = [1 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    # plt.style.use('seaborn')

    # plot roc curves
    fig, axis = plt.subplots(1, 1, figsize = (20, 20))
    axis.plot(fpr, tpr, linestyle='--',color='orange', label='Xg boost')
    axis.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    st.pyplot(fig)

def kaggle_predictions(train_model_path, out_res_name, is_cat_encode=True):
    clf = pickle.load(open(train_model_path, 'rb'))
    df_kgle = pd.read_csv('hr_employee_attrition_test.csv')
    df_kgle = df_kgle.drop(columns=cols_to_be_dropeed)
    df_kgle['overtime'].replace({'Yes': 1, 'No': 0}, inplace=True)
    if is_cat_encode:
      df_kgle = pd.get_dummies(df_kgle, columns=cat_cols, dtype='uint8')
    else:
      df_kgle[cat_cols] = df_kgle[cat_cols].astype('category')
    kaggle_pred_arr = clf.predict(df_kgle.drop(columns=['id']))
    kaggle_pred_df = pd.DataFrame(kaggle_pred_arr, columns=['label'])
    kaggle_pred_df['id'] = list(np.arange(1, len(kaggle_pred_arr) + 1))
    kaggle_pred_df = kaggle_pred_df[['id', 'label']]
    kaggle_pred_df.to_csv(f'submissions/{out_res_name}', index=False)

def load_and_pre_process_and_split_data(is_encode_cat=True):
    df = load_data()
    df['overtime'].replace({'Yes': 1, 'No': 0}, inplace=True)
    df['attrition'] = df['attrition'].replace({'Yes': 1, 'No': 0})
    for col in numeric_cols:
        handle_outliers(df, col)
    df_final = df.drop(columns=cols_to_be_dropeed)
    if is_encode_cat:
      df_final = pd.get_dummies(df_final, columns=cat_cols, dtype='uint8')
    else:
      df_final[cat_cols] = df_final[cat_cols].astype('category')
    X = df_final.drop('attrition', axis=1)
    y = df_final.attrition
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train,X_test,y_train,y_test

def main():
  df = load_data()
  buffer = io.StringIO()
  df.info(buf=buffer)
  st.text(buffer.getvalue())
  st.write("## Numerical Data")
  st.dataframe(df.describe())
  st.write("## Box plot")
  plot_for_all_cols(df, 'box')

  # https://www.kaggle.com/code/faressayah/ibm-hr-analytics-employee-attrition-performance#%F0%9F%93%8A-Exploratory-Data-Analysis
  fig, axis = plt.subplots(1,1, figsize=(20,20))

  non_heatmap_cols = cat_cols.copy()
  non_heatmap_cols.append('over18')
  # non_heatmap_cols.append('overtime')
  df['overtime'].replace({'Yes': 1, 'No': 0}, inplace=True)
  df['attrition'] = df['attrition'].replace({'Yes': 1, 'No': 0})
  sns.heatmap(df.drop(columns=non_heatmap_cols).corr(), annot=True, ax=axis)
  st.pyplot(fig)

  fig, axis = plt.subplots(1,1, figsize=(20,20))
  df.hist(figsize=(20,20), bins=50, ax=axis)
  st.pyplot(fig)

  # https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
  st.write("# Data Preprocessing")
  for col in numeric_cols:
    handle_outliers(df, col)
  
  st.write("## After outlier removal")
  # fig, axis = plt.subplots(1,1, figsize=(20,20))
  # df.hist(figsize=(20,20), bins=50, ax=axis)
  # st.pyplot(fig)

  plot_for_all_cols(df, 'box')
  
  st.write("## Dropping less corelatted columns: ")
  st.write(cols_to_be_dropeed)
  df_final = df.drop(columns=cols_to_be_dropeed)
  st.dataframe(df_final)
  st.write("## Categorical variables encoding")
  st.write(cat_cols)
  df_cat_one_hot = pd.get_dummies(df_final, columns=cat_cols, dtype='uint8')
  st.dataframe(df_cat_one_hot)
  buffer = io.StringIO()
  df_cat_one_hot.info(buf=buffer)
  st.text(buffer.getvalue())

  st.write("## Correlation Plot")
  fig, axis = plt.subplots(1,1, figsize=(10,30))
  df_cat_one_hot.drop('attrition', axis=1).corrwith(df_cat_one_hot.attrition).sort_values().plot(kind='barh', figsize=(10, 30), ax=axis)
  st.pyplot(fig)

  st.write("# Data Imbalance")
  fig, axis = plt.subplots(1,1, figsize=(20,20))
  df['attrition'].hist(ax=axis)
  st.pyplot(fig)
  scale = df['attrition'].value_counts()[0]/df['attrition'].value_counts()[1]

  st.write("# XgBoost Classifier")
  X = df_cat_one_hot.drop('attrition', axis=1)
  y = df_cat_one_hot.attrition
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
  xgb_clf = XGBClassifier(scale_pos_weight=scale, seed=42, num_boosting_rounds = 100, colsample_bytree = 0.7, learning_rate=0.1)
  xgb_clf.fit(X_train, y_train)
  write_train_and_test_reports(X_train, X_test, y_train, y_test, xgb_clf)
  st.write(xgb_clf.get_params())
  pickle.dump(xgb_clf, open('models/xgboost_balanced_outliers_removed_tuned.sav', 'wb'))
  kaggle_predictions('models/xgboost_balanced_outliers_removed_tuned.sav', 'xgboost_balanced_outliers_removed_tuned.csv')

  st.write('# Catboost Classifier')
  X_cat = df_final.drop('attrition', axis = 1)
  y_cat = df_final['attrition']
  X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
  cat_clf = CatBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=42, colsample_bylevel=0.7, class_weights={0: 1, 1:6}, reg_lambda=0.2)
  cat_clf.fit(X_train_cat, y_train_cat)
  write_train_and_test_reports(X_train_cat, X_test_cat, y_train_cat, y_test_cat, cat_clf)
  st.write(cat_clf.get_params())
  pickle.dump(cat_clf, open('models/catboost_balanced_outliers_removed.sav', 'wb'))
  kaggle_predictions('models/catboost_balanced_outliers_removed.sav', 'catboost_balanced_outliers_removed.csv')


def hyper_opt_best_model(X_train, y_train):
    space = { 'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.1, 0.5),
        'max_depth': hp.quniform('max_depth', 10, 50, 10),
        'n_estimators' : hp.choice('n_estimators', [10, 50, 100, 150, 200, 250, 300, 350, 400]),
        'scale_pos_weight': hp.uniform('scale_pos_weight', 4, 15),
        'min_child_weight': hp.uniform('min_child_weight', 1, 3),
      }
    def objective(params):
      int_types = [ "max_depth", "n_estimators"]
      params = convert_int_params(int_types, params)
      # float_types = ["min_samples_leaf", "min_samples_split"]
      # params = convert_float_params(float_types, params)
      params['random_state'] = 42
      model = XGBClassifier(**params)
    
      accuracy = cross_val_score(model, X_train, y_train, cv = 5).mean()
      # We aim to maximize accuracy, therefore we return it as a negative value
      return {'loss': -accuracy, 'status': STATUS_OK }
  
    trials = Trials()
    best = fmin(fn= objective,
              space= space,
              algo= tpe.suggest,
              max_evals = 80,
              trials= trials)
                
    return best

def hyper_opt_best_xgboost():
  X_train, X_test, y_train, y_test = load_and_pre_process_and_split_data()
  best_param = hyper_opt_best_model(X_train, y_train)
  int_types = [ "max_depth", "n_estimators"]
  best_param = convert_int_params(int_types, best_param)
  xgb_clf = XGBClassifier(**best_param)
  float_types = ["colsample_bytree", "reg_lambda", "learning_rate"]
  best_param = convert_float_params(float_types, best_param)
  xgb_clf.fit(X_train, y_train)
  write_train_and_test_reports(X_train, X_test, y_train, y_test, xgb_clf)
  name = 'xgboost_hyper_opt_' + "'" + best_param.__str__().replace(':', '_') + "'"
  pickle.dump(xgb_clf, open(f'models/{name}.sav', 'wb'))
  kaggle_predictions(f'models/{name}.sav', f'{name}.csv')
def xgboost_manual_tuning():
  st.write("# XgBoost Classifier")
  X_train, X_test, y_train, y_test = load_and_pre_process_and_split_data()
  params = {
'colsample_bytree': 0.75,
'reg_lambda': 0.5,
'learning_rate': 0.0213,
'n_estimators': 100,
'max_depth': 8,
'random_state': 42,
'scale_pos_weight': 4.5,
 'min_child_weight': 3,
}
  xgb_clf = XGBClassifier(**params)
  xgb_clf.fit(X_train, y_train)
  write_train_and_test_reports(X_train, X_test, y_train, y_test, xgb_clf)
  st.write(xgb_clf.get_params())
  name = 'xgboost_' + "'" + params.__str__().replace(':', '_') + "'"
  pickle.dump(xgb_clf, open(f'models/{name}.sav', 'wb'))
  kaggle_predictions(f'models/{name}.sav', f'{name}.csv')

def voting_classifier():
  X_train, X_test, y_train, y_test = load_and_pre_process_and_split_data(is_encode_cat=False)
  params = {
  # 'colsample_bytree': 0.75,
  # 'reg_lambda': 0.179,
  # 'learning_rate': 0.0213,
  'n_estimators': 100,
  # 'max_depth': 8,
  'random_state': 42,
  'scale_pos_weight': 4.5,
  # 'enable_categorical':True,
  # 'max_cat_to_onehot': 10
  }



  xgb_clf = XGBClassifier(**params,  enable_categorical=True)
  cat_clf = CatBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=42, class_weights={0: 1, 1:6}, cat_features=cat_cols)
  classifiers = [('lr', xgb_clf), ('svc', cat_clf)]
  voting_clf = VotingClassifier(estimators = classifiers, voting = 'soft')
  st.write("# Xgboost classifier")
  xgb_clf.fit(X_train, y_train)
  write_train_and_test_reports(X_train, X_test, y_train, y_test, xgb_clf)

  st.write("# Catboost classifier")
  cat_clf.fit(X_train, y_train)
  write_train_and_test_reports(X_train, X_test, y_train, y_test, cat_clf)

  st.write("# Voting classifier(Xgboost and catboost)")
  voting_clf.fit(X_train, y_train)
  write_train_and_test_reports(X_train, X_test, y_train, y_test, voting_clf)

  name = 'voting_clf_plain_xgboost_plain_est_100_catboost_est_100'
  pickle.dump(xgb_clf, open(f'models/{name}.sav', 'wb'))
  kaggle_predictions(f'models/{name}.sav', f'{name}.csv', False)



# plotting and plain xgboost and catboost
# main()
  
# xgboost_manual_tuning()

#hyperopt xgboost
# hyper_opt_best_xgboost()

#predicting using any saved model
# kaggle_predictions('models/xgboost_balanced_outliers_removed_tuned_hyperopt.sav', 'xgboost_balanced_outliers_removed_tuned_hyperopt.csv')

voting_classifier()



