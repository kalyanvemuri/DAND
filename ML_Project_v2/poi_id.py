#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

financial_features = ['salary', 'deferral_payments', 'total_payments',
                      'loan_advances', 'bonus', 'restricted_stock_deferred',
                      'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi']

new_features = ['fraction_to_poi', 'fraction_from_poi', 'bonus_to_salary', 'bonus_to_total']

features_list = ['poi'] + financial_features + email_features + new_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# convert to pandas dataframe for data manipulation
df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))

# set the index of df to be the employees series:
df.set_index(employees, inplace=True)

df = df.replace('NaN', np.nan)

# replace all NaN's with 0 for financial features
df[financial_features] = df[financial_features].fillna(0)

df_poi = df.loc[df['poi'] == True]
df_non_poi = df.loc[df['poi'] == False]

# impute NaN's in email features
imp = Imputer(missing_values='NaN', strategy='median', axis=0)

df.loc[df_poi.index, email_features] = imp.fit_transform(df_poi[email_features])
df.loc[df_non_poi.index, email_features] = imp.fit_transform(df_non_poi[email_features])

# Remove outliers
df.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'], inplace=True)

# add new features

#fraction of all messages sent from this person to POI
df['fraction_from_poi'] = df.from_this_person_to_poi / df.from_messages

#fraction of all messages sent to this person from POI
df['fraction_to_poi'] = df.from_poi_to_this_person / df.to_messages

#ratio of bonus to salary
df['bonus_to_salary'] = df.bonus / df.salary

#ratio of bonus to total payments
df['bonus_to_total'] = df.bonus / df.total_payments

df.fillna(value=0, inplace=True)

# create dataset back from dataframe
my_dataset = df.to_dict('index')

# classifier
clf = Pipeline([
    ('features', SelectKBest(k=19)),
    ('classifier', DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features=None, min_samples_split=20))
])

dump_classifier_and_data(clf, my_dataset, features_list)
