import pandas as pd
import numpy as np
# this is needed to make the graph chart
import matplotlib.pyplot as plt
# panda is a powerfule date mainuplation 
from sklearn.linear_model import LinearRegression
# isa plotting librarby for python and its numerical mathmetics extensipn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score


# Load the data
data = pd.read_csv('HistoricalCropData.csv')

# Select independent and dependent variables
x = data[['Year', 'Farm Numbers', 'Harvested (1000 acres)']]  # Independent variables

y = data['Yield (bushels)']  # Dependent variable (Yield)
# Use 80 percetn of the data to to train and 20 data to test
#splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = .2, random_state=0)

# train the linear regression model

train = LinearRegression()
train.fit(x_train,y_train)

#Makr the predaction on the test set for the bushels yeield
y_pred = train.predict(x_test)

meanSqueareError = mean_squared_error(y_test,y_pred)
randomError = np.sqrt(meanSqueareError)
r2 = r2_score(y_test,y_pred)

# print the perfromance metric
print(f"Mean Squered Error: {meanSqueareError:.2f}")
print(f"Root Mean Squered Error: {randomError:.2f}")
print(f"R-squared: {r2:.2f}")

# printing out the mean squere error
"""""
plt.scatter(x_train,y_train, color ='g')
plt.plot(x_test,y_pred,color ='k')

plt.show()
"""

#Plotting reltionsjo[ of the line excustiong
plt.figure(figsize= (20,6))

#compute the avarge for ploting the years
av_year = x['Year'].mean()
# compute the avarge of the farm number
avg_farm_numbers = x['Farm Numbers'].mean()
# averge harvested acres
avg_harvested_acres =x['Harvested (1000 acres)'].mean()
#exted by 4 year that us tring to predicaeted

future_year = 4
last_year = max(x['Year'])
features = ['Year', 'Farm Numbers', 'Harvested (1000 acres)']


for i, feature in enumerate(features):
    
    plt.subplot(1, 3, i+1)  
    ## trained data
    plt.scatter(x_train[feature], y_train, color = 'g')
    # tested year 
    plt.scatter(x_test[feature], y_test, color = 'r')
    plt.xlabel(feature)
    plt.ylabel('Yield')

    # Preparing data for predictions
    # Create a range of values for the current feature
    all_data = np.linspace(min(x[feature]), max(x[feature]), 100).reshape(-1, 1)
    
    # Depending on which feature we're looking at, we'll adjust our prediction data accordingly
    if feature == 'Year':
        # Using average values for other features to make predictions
        farm_array = np.full(all_data.shape, avg_farm_numbers)
        acres_array = np.full(all_data.shape, avg_harvested_acres)
        prediction_data = np.hstack([all_data, farm_array, acres_array])
    elif feature == 'Farm Numbers':
        # For Farm Numbers feature, adjust prediction data
        year_array = np.full(all_data.shape, av_year)
        acres_array = np.full(all_data.shape, avg_harvested_acres)
        prediction_data = np.hstack([year_array, all_data, acres_array])
    elif feature == 'Harvested (1000 acres)':
        # For Harvested feature, adjust prediction data
        year_array = np.full(all_data.shape, av_year)
        farm_array = np.full(all_data.shape, avg_farm_numbers)
        prediction_data = np.hstack([year_array, farm_array, all_data])
    
    # Using the model to make predictions and then plot
    predicted_yield = train.predict(prediction_data)
    plt.plot(all_data, predicted_yield, color='blue')

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()
    

