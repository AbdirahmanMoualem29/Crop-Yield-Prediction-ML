import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers

# Load datasets
corn_data = pd.read_csv('HistoricalCropData.csv')
weather_data = pd.read_csv("John_Glenn_Airport_Modified_Corn.csv", parse_dates=['DATE'], index_col="DATE")

# Preprocess data
weather_data['avg_daily_temp'] = (weather_data['tmax'] + weather_data['tmin']) / 2
weather_data['year'] = weather_data.index.year
annual_weather_metrics = weather_data.groupby('year').agg({
    'prcp': 'sum',
    'snow': 'sum',
    'snwd': 'sum',
    'avg_daily_temp': 'mean'
}).reset_index()

combined_data = pd.merge(corn_data, annual_weather_metrics, left_on='Year', right_on='year', how='inner')
features = ['Year', 'Farm Numbers', 'Harvested (1000 acres)', 'avg_daily_temp', 'prcp', 'snow', 'snwd']
X = combined_data[features]
y = combined_data['Yield (bushels)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32)

# Evaluate the model
mse_nn = model.evaluate(X_test_scaled, y_test)
print("Neural Network Mean Squared Error:", mse_nn)

# Plot training history
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
