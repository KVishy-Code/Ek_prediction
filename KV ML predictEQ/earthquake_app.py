from flask import Flask, request, jsonify, render_template
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import math

app = Flask(__name__)

# df = pd.read_csv("../../../../Downloads/expanded_earthquake_prediction_data.csv")
df = pd.read_csv("../../../../Downloads/synthetic_earthquake_dataset_48251.csv")

curated_df = pd.read_csv("../../../../Downloads/curated_lat_long_country.csv")

numerical_cols = ['Latitude', 'Longitude', 'Past_EQ_Count', 'Avg_Magnitude', 'Avg_Depth_km', 'Seismic_Activity_Index', 'Soil_Stability', 'Population_Density', 'Years_Since_Last_EQ']
categorical_cols = ['Tectonic_Region']
target_col = 'Modeled_Major_EQ_Prob_20Y'

le = LabelEncoder()
df['Tectonic_Region_encoded'] = le.fit_transform(df['Tectonic_Region'])

X = df[numerical_cols + ['Tectonic_Region_encoded']]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6741)

# Use regularization parameters to prevent overfitting
model = DecisionTreeRegressor(
    max_depth=10,           # Limit tree depth
    min_samples_split=10,   # Minimum samples to split a node
    min_samples_leaf=5,     # Minimum samples in a leaf node
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model performance to check for overfitting
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
print(f"Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
print(f"Overfitting indicator (Train R2 - Test R2): {train_r2 - test_r2:.4f}")

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    radius = 6371
    return radius * c

def predict_earthquake_probability(latitude, longitude):
    # Find the closest data point in the earthquake dataset
    distances = df.apply(lambda row: haversine_distance(latitude, longitude, row['Latitude'], row['Longitude']), axis=1)
    closest_idx = distances.idxmin()
    closest_row = df.loc[closest_idx]

    # Prepare input data for prediction
    input_data = {
        'Latitude': latitude,
        'Longitude': longitude,
        'Past_EQ_Count': closest_row['Past_EQ_Count'],
        'Avg_Magnitude': closest_row['Avg_Magnitude'],
        'Avg_Depth_km': closest_row['Avg_Depth_km'],
        'Seismic_Activity_Index': closest_row['Seismic_Activity_Index'],
        'Soil_Stability': closest_row['Soil_Stability'],
        'Population_Density': closest_row['Population_Density'],
        'Years_Since_Last_EQ': closest_row['Years_Since_Last_EQ'],
        'Tectonic_Region_encoded': closest_row['Tectonic_Region_encoded']
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    # Find the closest country
    distances_country = curated_df.apply(lambda row: haversine_distance(latitude, longitude, row['latitude'], row['longitude']), axis=1)
    closest_country_idx = distances_country.idxmin()
    closest_country = curated_df.loc[closest_country_idx, 'country']

    return prediction, closest_country

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scoli')
def scoli():
    return render_template('scoli.html')

@app.route('/manual_input')
def manual_input():
    return render_template('manual_input.html')

@app.route('/predict_earthquake', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        lat = data['latitude']
        lon = data['longitude']
        print(f"Received lat: {lat}, lon: {lon}")
        prob, country = predict_earthquake_probability(lat, lon)
        print(f"Predicted prob: {prob}, country: {country}")
        return jsonify({'probability': prob, 'country': country})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
