#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


# Step 1: Load the Data
df = pd.read_csv('weather.csv')


# In[3]:


# Step 2: Data Exploration
print(df.head())
print(df.info())
print(df.describe())


# In[4]:


# Step 3: Data Visualization
sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.show()


# In[16]:


# Step 4: Data Analysis (analyze each term)
# Example: Calculate average MaxTemp by month
#df['DateObserved'] = pd.to_datetime(df['DateObserved'])
#df['Month'] = df['DateObserved'].dt.month
#monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()
df['Month'] = df['MaxTemp'].apply(lambda x: x % 12 + 1)

# Calculate the monthly average maximum temperature
monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()

# If you want to reset the index for a cleaner output
monthly_avg_max_temp = monthly_avg_max_temp.reset_index()

# Print or use the result as needed
print(monthly_avg_max_temp)


# In[17]:


# Step 5: Data Visualization (Part 2)
plt.figure(figsize=(10, 5))
plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
plt.xlabel('Month')
plt.ylabel('Average Max Temperature')

plt.title('Monthly Average Max Temperature')
plt.grid(True)
plt.show()


# In[18]:


# Step 6: Advanced Analysis (e.g., predict Rainfall)
# Prepare the data for prediction
X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and calculate the Mean Squared Error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse}')


# In[23]:


# Step 7: Conclusions and Insights (analyze each term)
# Example: Identify the highest and lowest rainfall months
highest_rainfall_month = monthly_avg_max_temp.idxmax()
lowest_rainfall_month = monthly_avg_max_temp.idxmin()
print(f'Highest rainfall month: {highest_rainfall_month},\nLowest rainfall month: {lowest_rainfall_month}')


# In[ ]:




