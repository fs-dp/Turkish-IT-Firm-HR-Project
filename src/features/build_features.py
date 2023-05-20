# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


# Load pickled dataframes
# These dataframes have been saved to disk from a previous analysis or data processing step
df_hr = pd.read_pickle("../visualization/df_hr.pkl")
emp_left = pd.read_pickle("../visualization/emp_left.pkl")
emp_stayed = pd.read_pickle("../visualization/emp_stayed.pkl")

# Define categorical variables 
cat_var = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus"]

# Select only the categorical variables from the dataframe
df_cat = df_hr[cat_var]

# Initialize OneHotEncoder
# This is a preprocessing step for machine learning algorithms which cannot handle categorical data
# OneHotEncoder converts categorical data into a format that can be used by these algorithms
onehotencoder = OneHotEncoder()

# Fit the OneHotEncoder and transform the categorical data, convert it to a DataFrame
df_cat = pd.DataFrame(onehotencoder.fit_transform(df_cat).toarray(), columns=onehotencoder.get_feature_names(cat_var))

# Select all columns from the dataframe that are not categorical (i.e., numerical data)
df_num = df_hr.loc[:, ~df_hr.columns.isin(cat_var)]

# Merge the processed categorical and numerical data back into one dataframe
df_merged = pd.concat([df_cat, df_num], axis=1)
df_merged['Attrition']=df_hr['Attrition']
df_merged.to_pickle('../models/df_merged.pkl')
