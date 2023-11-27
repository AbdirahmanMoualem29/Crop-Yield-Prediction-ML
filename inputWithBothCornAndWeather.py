import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Assuming 'corn_data.csv' and 'weather_data.csv' are in the current working directory.
# Load the datasets
corn_data = pd.read_csv('HistoricalCropData.csv')
weather_data = pd.read_csv("John_Glenn_Airport_Modified_Corn.csv", parse_dates=['DATE'], index_col="DATE")


# Calculate the daily average temperature and aggregate metrics
weather_data['avg_daily_temp'] = (weather_data['tmax'] + weather_data['tmin']) / 2

# Since 'DATE' is the index, use the index to access the year
weather_data['year'] = weather_data.index.year

# Now aggregate using the 'year' column you just created
annual_weather_metrics = weather_data.groupby('year').agg({
    'prcp': 'sum',
    'snow': 'sum',
    'snwd': 'sum',
    'avg_daily_temp': 'mean'
}).reset_index()

# Merge the corn data with the annual weather metrics data based on the year
combined_data = pd.merge(corn_data, annual_weather_metrics, left_on='Year', right_on='year', how='inner')

# Prepare data for regression
features = ['Year', 'Farm Numbers', 'Harvested (1000 acres)', 'avg_daily_temp', 'prcp', 'snow', 'snwd']
X = combined_data[features]
y = combined_data['Yield (bushels)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate the Linear Regression model
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Create and train the Random Forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate the Random Forest model
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Plotting the results to show improvement
plt.figure(figsize=(14, 7))

# Linear Regression plot
plt.subplot(1, 2, 1)
plt.scatter(X_test['Year'], y_test, color='blue', label='Actual Yield')
plt.scatter(X_test['Year'], y_pred_lr, color='red', label='Predicted Yield (LR)')
plt.title('Linear Regression Model')
plt.xlabel('Year')
plt.ylabel('Corn Yield (bushels)')
plt.legend()

# Random Forest plot
plt.subplot(1, 2, 2)
plt.scatter(X_test['Year'], y_test, color='blue', label='Actual Yield')
plt.scatter(X_test['Year'], y_pred_rf, color='green', label='Predicted Yield (RF)')
plt.title('Random Forest Model')
plt.xlabel('Year')
plt.ylabel('Corn Yield (bushels)')
plt.legend()

plt.tight_layout()
plt.show()

# Print the results
print("Linear Regression Mean Squared Error:", mse_lr)
print("Linear Regression R-squared:", r2_lr)
print("Random Forest Mean Squared Error:", mse_rf)
print("Random Forest R-squared:", r2_rf)


# Features from corn data only
features_corn = ['Year', 'Farm Numbers', 'Harvested (1000 acres)']
X_corn = corn_data[features_corn]
y_corn = corn_data['Yield (bushels)']

# Split the corn data
X_train_corn, X_test_corn, y_train_corn, y_test_corn = train_test_split(X_corn, y_corn, test_size=0.2, random_state=42)

# Linear Regression model
lr_model_corn = LinearRegression()
lr_model_corn.fit(X_train_corn, y_train_corn)

# Predict and evaluate
y_pred_corn = lr_model_corn.predict(X_test_corn)
mse_corn = mean_squared_error(y_test_corn, y_pred_corn)
r2_corn = r2_score(y_test_corn, y_pred_corn)


print("Without Weather Data - Linear Regression Mean Squared Error:", mse_corn)
print("Without Weather Data - Linear Regression R-squared:", r2_corn)
print("With Weather Data - Linear Regression Mean Squared Error:", mse_lr)
print("With Weather Data - Linear Regression R-squared:", r2_lr)
print("With Weather Data - Random Forest Mean Squared Error:", mse_rf)
print("With Weather Data - Random Forest R-squared:", r2_rf)


plt.figure(figsize=(14, 7))

# Without weather data
plt.subplot(1, 2, 1)
plt.scatter(X_test_corn['Year'], y_test_corn, color='blue', label='Actual Yield')
plt.scatter(X_test_corn['Year'], y_pred_corn, color='red', label='Predicted Yield (No Weather Data)')
plt.title('Model Without Weather Data')
plt.xlabel('Year')
plt.ylabel('Corn Yield (bushels)')
plt.legend()

# With weather data
plt.subplot(1, 2, 2)
plt.scatter(X_test['Year'], y_test, color='blue', label='Actual Yield')
plt.scatter(X_test['Year'], y_pred_rf, color='green', label='Predicted Yield (With Weather Data)')
plt.title('Model With Weather Data')
plt.xlabel('Year')
plt.ylabel('Corn Yield (bushels)')
plt.legend()

plt.tight_layout()
plt.show()


