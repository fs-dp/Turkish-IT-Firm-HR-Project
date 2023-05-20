# Import the necessary libraries and load the dataset
import pandas as pd
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Inspect the first few rows of the data
df.head()

# View data structure and types
df.info()

# Statistical summary of the data
df.describe()

# Transform 'Yes'/'No' to '1'/'0' in 'Attrition' and 'OverTime' columns
df['Attrition'] = df["Attrition"].apply(lambda x:1 if x=="Yes" else 0)
df['OverTime'] = df["OverTime"].apply(lambda x:1 if x=="Yes" else 0)

# Generate histograms for all columns
df.hist(bins=30, figsize=(20,20), color='g')

# Drop columns that don't offer meaningful information
cols_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
df.drop(cols_to_drop, axis=1, inplace=True)

#Let's see the ratio of people who left.
left_df = df[df["Attrition"]==1]
stayed_df = df[df["Attrition"]==0]
print(f"Percentage of employees left is {round(100*len(left_df)/len(df),2)}")

#It seems like we are dealing with an imbalanced data which we will address later. 
#Now, let's just create pickle files of dataframes for further reference and move to visualization.py for further insights.
df.to_pickle('../visualization/df_hr.pkl')
left_df.to_pickle('../visualization/emp_left.pkl')
stayed_df.to_pickle('../visualization/emp_stayed.pkl')
