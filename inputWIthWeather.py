import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the corn yield data and weather data
cornData = pd.read_csv('HistoricalCropData.csv')
weather = pd.read_csv("John_Glenn_Airport_Modified_Corn.csv", parse_dates=['DATE'], index_col="DATE")

# Drop missing values from the weather dataset
weather = weather.dropna()

# Filter out extreme values that might be outliers
weather = weather[(weather['tmax'] <= 100) & (weather['tmax'] >= -30)]

# Convert the 'station' column to numeric codes for use in model
weather['station'] = weather['station'].astype('category').cat.codes

# Standardize the weather data for better model performance
scaler = StandardScaler()
weather_scaled = scaler.fit_transform(weather[['prcp', 'snow', 'snwd', 'tmax', 'tmin']])

# Preparing the independent (X) and dependent (y) variables for regression
X = pd.DataFrame(weather_scaled, columns=['prcp', 'snow', 'snwd', 'tmax', 'tmin'])
y = weather['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display model coefficients and metrics
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

# Set up a matplotlib figure for displaying different plots
fig, axs = plt.subplots(1, 4, figsize=(10, 3))

# Plotting Actual vs Predicted Values
axs[0].scatter(y_test, y_pred, alpha=0.5)
axs[0].plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)
axs[0].set_xlabel('Actual')
axs[0].set_ylabel('Predicted')
axs[0].set_title('Actual vs. Predicted Values')

# Plotting Residuals
residuals = y_test - y_pred
axs[1].scatter(y_test, residuals)
axs[1].axhline(y=0, color='red', linestyle='--')
axs[1].set_xlabel('Actual Values')
axs[1].set_ylabel('Residuals')
axs[1].set_title('Residual Plot')

# Plotting Distribution of Residuals
sns.histplot(residuals, kde=True, ax=axs[2])
axs[2].set_xlabel('Residuals')
axs[2].set_ylabel('Frequency')
axs[2].set_title('Distribution of Residuals')

# Analyzing Temperature Trend Over Years
# Aggregate data by year to get the average temperature
annual_temp = weather['tmax'].resample('Y').mean()

# Prepare data for trend analysis
trend_df = pd.DataFrame({'year': annual_temp.index.year, 'avg_temp': annual_temp.values})

# Fit the linear regression model for trend analysis
model_trend = LinearRegression()
model_trend.fit(trend_df[['year']], trend_df['avg_temp'])

# Predict the temperatures for trend
trend_df['predicted'] = model_trend.predict(trend_df[['year']])

# Plotting Annual Average Temperature Trend
axs[3].scatter(trend_df['year'], trend_df['avg_temp'], label='Actual')
axs[3].plot(trend_df['year'], trend_df['predicted'], color='red', label='Trend')
axs[3].set_xlabel('Year')
axs[3].set_ylabel('Average Temperature')
axs[3].set_title('Annual Average Temperature Trend')
axs[3].legend()

# Adjusting layout
plt.tight_layout()
plt.show()

# Displaying trend slope
print("Temperature trend per year:", model_trend.coef_[0])

# Calculating and plotting annual average temperatures
weather['avg_daily_temp'] = (weather['tmax'] + weather['tmin']) / 2
annual_avg_temp = weather['avg_daily_temp'].resample('A').mean()

# Preparing data for regression
annual_avg_temp_df = annual_avg_temp.reset_index()
annual_avg_temp_df['year'] = annual_avg_temp_df['DATE'].dt.year
X = annual_avg_temp_df[['year']]
y = annual_avg_temp_df['avg_daily_temp']

# Training linear regression model
model = LinearRegression()
model.fit(X, y)

# Making predictions
annual_avg_temp_df['predicted_temp'] = model.predict(X)

# Calculating metrics and plotting results
mse = mean_squared_error(y, annual_avg_temp_df['predicted_temp'])
r2 = r2_score(y, annual_avg_temp_df['predicted_temp'])

plt.figure(figsize=(10, 5))
plt.scatter(X, y, label='Actual Average Temperature')
plt.plot(X, annual_avg_temp_df['predicted_temp'], color='red', label='Trend Line')
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.title('Annual Average Temperature Trend')
plt.legend()
plt.show()

# Displaying slope of temperature trend and evaluation metrics
print("Temperature trend per year:", model.coef_[0])
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# A positive coefficient here indicates a rising temperature trend over the years.
