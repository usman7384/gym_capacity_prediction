import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from math import sin, cos, pi
from sklearn.model_selection import train_test_split
from datetime import timedelta
import numpy as np



# Load your dataset
df = pd.read_csv('data/gym_occupancy_data.csv')

# Convert timestamp from string to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
# drop time_of_day column
df = df.drop(['time_of_day'], axis=1)

# Function to round timestamps to the each 15h minute of the hour
def round_to_nearest_quarter_hour(timestamp):
    minutes = (timestamp.minute // 15) * 15
    return timestamp.replace(minute=minutes, second=0)


# Apply rounding to each timestamp
df['timestamp'] = df['timestamp'].apply(round_to_nearest_quarter_hour)

# remove duplicated timestamps by keeping the first occurrence
df = df.drop_duplicates(subset='timestamp', keep='first')


# Now set the adjusted timestamp as index
df.set_index('timestamp', inplace=True)

# Proceed with generating the full datetime range for reindexing
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15T')
df = df.reindex(full_range)

# Reset index to get 'timestamp' column back if needed
df.reset_index(inplace=True)
df.rename(columns={'index': 'timestamp'}, inplace=True)

# Impute numerical features
df['temperature'] = df['temperature'].interpolate(method='linear')
# Fill missing occupancy_percentage values with the mean of the previous and next values
df['occupancy_percentage'] = df['occupancy_percentage'].replace(0, np.nan)
df['occupancy_percentage'] = df['occupancy_percentage'].fillna((df['occupancy_percentage'].shift() + df['occupancy_percentage'].shift(-1)) / 2)

# fill missing weather_condition values with the last valid observation
df['weather_condition'] = df['weather_condition'].fillna(method='ffill')
df['season'] = df['season'].fillna(method='ffill')

# Extracting time features
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0 is Monday, 6 is Sunday

# Fill missing occupancy_percentage values based on the mean of occupancy percentage for the given time of the day
df['occupancy_percentage'] = df.groupby('hour')['occupancy_percentage'].transform(lambda x: x.fillna(x.mean()))
# fill remaining nan values with the 0
df['occupancy_percentage'] = df['occupancy_percentage'].fillna(0)

# Cyclical encoding for hours and days of the week
df['hour_sin'] = df['hour'].apply(lambda x: sin(2 * pi * x / 24))
df['hour_cos'] = df['hour'].apply(lambda x: cos(2 * pi * x / 24))
df['day_of_week_sin'] = df['day_of_week'].apply(lambda x: sin(2 * pi * x / 7))
df['day_of_week_cos'] = df['day_of_week'].apply(lambda x: cos(2 * pi * x / 7))

# One-hot encoding for non-ordinal categorical variables
df = pd.get_dummies(df, columns=['season', 'weather_condition'])

# Optionally normalize/standardize numerical features
# scaler = MinMaxScaler()
# numerical_features = ['temperature', 'occupancy_percentage', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']
# df[numerical_features] = scaler.fit_transform(df[numerical_features])



#Dropping original columns that have been encoded or are not needed
df = df.drop(['hour', 'day', 'month', 'day_of_week'], axis=1)

# if missing values print fix the code above
print(df.isnull().sum())

# save processed data to a new file
processed_file_path = 'data/processed_gym_occupancy_data.csv'
df.to_csv(processed_file_path, index=False)




# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)  # Notice shuffle=False for time series

# # Optionally, further split X_train to create a validation set
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2





