# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load pickled dataframes
df_hr = pd.read_pickle("../visualization/df_hr.pkl")
emp_left = pd.read_pickle("../visualization/emp_left.pkl")
emp_stayed = pd.read_pickle("../visualization/emp_stayed.pkl")

# Compute and visualize correlations in the HR dataset
correlations = df_hr.corr()
f, ax = plt.subplots(figsize=(20,20))  # Create a new figure and a subplot
sns.heatmap(correlations, annot=True)  # Draw a heatmap with the correlation data

# Visualize employee attrition by age
sns.countplot(x="Age", hue="Attrition", data=df_hr)

# Visualize attrition by different job characteristics
plt.subplot(411)
sns.countplot(x='JobRole', hue='Attrition', data=df_hr)
plt.subplot(412)
sns.countplot(x='MaritalStatus', hue='Attrition', data=df_hr)
plt.subplot(413)
sns.countplot(x='JobInvolvement', hue='Attrition', data=df_hr)
plt.subplot(414)
sns.countplot(x='JobLevel', hue='Attrition', data=df_hr)

# KDE plot of employees' distance from home
plt.figure(figsize=(12,8))
sns.kdeplot(emp_left['DistanceFromHome'], label = "Employees who Left", shade=True, color='red')
sns.kdeplot(emp_stayed['DistanceFromHome'], label = "Employees who Stayed", shade=True, color='blue')
plt.xlabel("Distance From Home")

# KDE plot of years with current manager
plt.figure(figsize=(12,8))
sns.kdeplot(emp_left['YearsWithCurrManager'], label = "Employees who Left", shade=True, color='red')
sns.kdeplot(emp_stayed['YearsWithCurrManager'], label = "Employees who Stayed", shade=True, color='blue')
plt.xlabel("Years with Current Manager")

# KDE plot of total working years
plt.figure(figsize=(12,8))
sns.kdeplot(emp_left['TotalWorkingYears'], label = "Employees who Left", shade=True, color='red')
sns.kdeplot(emp_stayed['TotalWorkingYears'], label = "Employees who Stayed", shade=True, color='blue')
plt.xlabel("Total Working Years")

# Boxplot of monthly income by gender
sns.boxplot(x="MonthlyIncome", y="Gender", data=df_hr)

# Boxplot of monthly income by job role
plt.figure(figsize=(20,15))
sns.boxplot(x="MonthlyIncome", y="JobRole", data=df_hr)


