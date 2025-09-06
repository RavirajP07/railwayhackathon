import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("synthetic_trains.csv")

# Encode categorical features
le_type = LabelEncoder()
le_weather = LabelEncoder()
df['train_type_enc'] = le_type.fit_transform(df['train_type'])
df['weather_enc'] = le_weather.fit_transform(df['weather'])

# Select features and target
X = df[['train_type_enc', 'time_of_day', 'day_of_week',
        'weather_enc', 'congestion', 'historical_delay']]
y = df['actual_delay']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))

# Save model (optional)
import joblib
joblib.dump(model, "delay_predictor.pkl")