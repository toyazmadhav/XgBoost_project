import imp
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
    lower_bound = q1 - (2 * iqr)
    upper_bound = q3 + (2 * iqr)

    upper_bound=df[colm].mean()+2*df[colm].std()
    lower_bound=df[colm].mean()-2*df[colm].std()

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

def kaggle_predictions(train_model_path, cols_to_be_dropeed, cat_cols):
    clf = pickle.load(open(train_model_path, 'rb'))
    df_kgle = pd.read_csv('hr_employee_attrition_test.csv')
    df_kgle = df_kgle.drop(columns=cols_to_be_dropeed)
    df_kgle['overtime'].replace({'Yes': 1, 'No': 0}, inplace=True)
    df_kgle_cat_one_hot = pd.get_dummies(df_kgle, columns=cat_cols, dtype='uint8')
    kaggle_pred_arr = clf.predict(df_kgle_cat_one_hot.drop(columns=['id']))
    kaggle_pred_df = pd.DataFrame(kaggle_pred_arr, columns=['label'])
    kaggle_pred_df['id'] = list(np.arange(1, len(kaggle_pred_arr) + 1))
    kaggle_pred_df = kaggle_pred_df[['id', 'label']]
    kaggle_pred_df.to_csv(f'submissions/xgboost_plain_balanced.csv', index=False)

def main():
  df = load_data()
  buffer = io.StringIO()
  df.info(buf=buffer)
  st.text(buffer.getvalue())
  st.write("## Numerical Data")
  st.dataframe(df.describe())
  st.write("## Box plot")
  plot_for_all_cols(df, 'box')

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
    'standardhours'
  ]

  # https://www.kaggle.com/code/faressayah/ibm-hr-analytics-employee-attrition-performance#%F0%9F%93%8A-Exploratory-Data-Analysis
  fig, axis = plt.subplots(1,1, figsize=(20,20))

  non_heatmap_cols = cat_cols.copy()
  non_heatmap_cols.append('over18')
  # non_heatmap_cols.append('overtime')
  df['overtime'].replace({'Yes': 1, 'No': 0}, inplace=True)
  df['attrition'] = df['attrition'].replace({'Yes': 1, 'No': 0})
  sns.heatmap(df.drop(columns=non_heatmap_cols).corr(), annot=True, ax=axis)
  st.pyplot(fig)

  # https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
  st.write("# Data Preprocessing")
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
  xgb_clf = XGBClassifier(scale_pos_weight=scale, seed=42)
  xgb_clf.fit(X_train, y_train)
  write_train_and_test_reports(X_train, X_test, y_train, y_test, xgb_clf)
  st.write(xgb_clf.get_params())
  pickle.dump(xgb_clf, open('models/xgboost_balanced.sav', 'wb'))
  kaggle_predictions('models/xgboost_balanced.sav', cols_to_be_dropeed, cat_cols)

main()


