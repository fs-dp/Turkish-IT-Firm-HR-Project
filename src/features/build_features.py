# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load pickled dataframes
df_hr = pd.read_pickle("../visualization/df_hr.pkl")
emp_left = pd.read_pickle("../visualization/emp_left.pkl")
emp_stayed = pd.read_pickle("../visualization/emp_stayed.pkl")


cat_var = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus"]
df_cat = df_hr[cat_var]