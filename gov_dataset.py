#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')
df=pd.read_csv('GOV dataset.csv')
df


# # Display dataset info

# In[57]:


print("Dataset Information:")
df.info()
print("\nFirst 5 Rows:")
print(df.head())


# # Handle missing values

# In[58]:


df['districtName'].fillna("Unknown", inplace=True)
df['mandalName'].fillna("Unknown", inplace=True)
df['villageName'].fillna("Unknown", inplace=True)


# In[59]:


df


# # Convert date column

# In[60]:


df['dataDate'] = pd.to_datetime(df['dataDate'], errors='coerce', format='%b-%y')


# # Display updated dataset info
# 

# In[61]:


print("\nUpdated Dataset Information:")
df.info()
print("\nUpdated First 5 Rows:")
print(df.head())


# # Remove duplicates

# In[62]:


df.drop_duplicates(inplace=True)


# In[63]:


df.info()


# # Normalize numerical data

# In[64]:


df_numeric = df.select_dtypes(include=[np.number])
df[df_numeric.columns] = (df_numeric - df_numeric.mean()) / df_numeric.std()
df_numeric


# # Data Visualization

# In[65]:


plt.figure(figsize=(10, 5))
sns.histplot(df['delCnt'], bins=30, kde=True, color='blue')
plt.title("Distribution of Total Deliveries")
plt.xlabel("Number of Deliveries")
plt.ylabel("Frequency")
plt.show()


# # This chart shows how frequently different numbers of deliveries occur. It helps us understand the overall spread and identify common values.

# In[66]:


plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='dataDate', y='delCnt', marker="o", color='red')
plt.title("Trend of Total Deliveries Over Time")
plt.xlabel("Date")
plt.ylabel("Total Deliveries")
plt.xticks(rotation=45)
plt.show()


# # This line chart shows how the number of deliveries changes over time. It helps to identify trends, seasonal patterns, and any sudden changes.
# 

# In[67]:


plt.figure(figsize=(8, 5))
sns.histplot(df[['govtDelCnt', 'pvtDelCnt']], bins=20, kde=True, element="step", palette=["blue", "red"])
plt.title("Comparison of Govt vs Private Deliveries")
plt.xlabel("Number of Deliveries")
plt.ylabel("Frequency")
plt.legend(["Govt Deliveries", "Private Deliveries"])
plt.show()


# # This chart compares the number of deliveries done by government and private institutions. It helps us see which type of institution is handling more deliveries.

# # Correlation Heatmap

# In[68]:


plt.figure(figsize=(10, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# # This heatmap shows how different numerical features in the dataset are related to each other. High values indicate strong relationships.

# In[69]:


pairplot_columns = [col for col in ['delCnt', 'govtDelCnt', 'pvtDelCnt', 'ancRegCnt', 'chImzCnt'] if col in df.columns]
if pairplot_columns:
    sns.pairplot(df[pairplot_columns])
    plt.show()
else:
    print("No valid columns available for pairplot.")


# # This plot helps visualize relationships between important numerical features in the dataset. It shows patterns and trends between different variables.

# In[70]:


plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['delCnt', 'govtDelCnt', 'pvtDelCnt']], palette="Set2")
plt.title("Boxplot of Deliveries")
plt.show()


# # A boxplot is useful for spotting outliers and understanding the spread of delivery numbers. It shows the minimum, maximum, and median values.

# In[71]:


plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['delCnt', 'govtDelCnt', 'pvtDelCnt']], palette="Set2")
plt.title("Boxplot of Deliveries")
plt.show()


# In[ ]:





# In[72]:


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# In[73]:


sns.pairplot(df)
plt.show()


# sns.violinplot(x=df['govtDelCnt'])
# plt.title("Violin Plot of Government Deliveries")
# plt.show()

# In[74]:


sns.violinplot(x=df['pvtDelCnt'])
plt.title("Violin Plot of Private Deliveries")
plt.show()


# # A violin plot combines a boxplot and a density plot. It shows the spread and shape of the distribution of private deliveries.

# sns.countplot(x=df['govtDelCnt'] > df['pvtDelCnt'])
# plt.title("Government vs Private Deliveries Count Comparison")
# plt.show()

# In[75]:


plt.figure(figsize=(8,5))
sns.barplot(x=df['dataDate'].dt.month, y=df['delCnt'], ci=None)
plt.title("Total Deliveries by Month")
plt.show()


# # This bar chart shows how deliveries change from month to month. It helps to see if there are seasonal patterns.

# plt.figure(figsize=(8,5))
# sns.kdeplot(df['delCnt'], shade=True, color='blue')
# plt.title("Kernel Density Estimation of Total Deliveries")
# plt.show()
# 

# In[76]:


plt.figure(figsize=(8,5))
sns.scatterplot(x=df['govtDelCnt'], y=df['delCnt'], color='blue')
plt.title("Govt Deliveries vs Total Deliveries")
plt.show()


# # This scatter plot shows if there is a relationship between government deliveries and the total number of deliveries.

# In[77]:


plt.figure(figsize=(8,5))
sns.scatterplot(x=df['pvtDelCnt'], y=df['delCnt'], color='red')
plt.title("Private Deliveries vs Total Deliveries")
plt.show()


# # This scatter plot shows if there is a relationship between government deliveries and the total number of deliveries.

# In[78]:


plt.figure(figsize=(8,5))
sns.boxplot(x='dataDate', y='delCnt', data=df)
plt.title("Boxplot of Deliveries Over Time")
plt.xticks(rotation=45)
plt.show()


# In[79]:


plt.figure(figsize=(8,5))
sns.lineplot(x=df.index, y=df['delCnt'], color='green')
plt.title("Line Plot of Total Deliveries Over Index")
plt.show()


# In[80]:


plt.figure(figsize=(8,5))
sns.ecdfplot(df['delCnt'])
plt.title("ECDF Plot of Total Deliveries")
plt.show()


# # The ECDF plot helps us understand the cumulative distribution of deliveries, showing the proportion of deliveries below a certain number.

# In[81]:


plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()


# # This heatmap highlights any missing data in the dataset. It helps to quickly spot columns that need cleaning or filling.

# In[82]:


plt.figure(figsize=(8,5))
sns.boxenplot(x=df['delCnt'])
plt.title("Boxen Plot of Total Deliveries")
plt.show()


# # A boxen plot is a modified boxplot that is useful for large datasets. It shows data distribution in more detail compared to a regular boxplot.

# In[ ]:




