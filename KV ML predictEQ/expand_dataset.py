import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import math

df = pd.read_csv("../../../../Downloads/earthquake_prediction_data.csv")

le = LabelEncoder()
df['Tectonic_Region_encoded'] = le.fit_transform(df['Tectonic_Region'])

def is_near_ring_of_fire(lat, lon):
    lon_360 = (lon + 360) % 360

    if lat >= -60 and lat <= 60 and lon >= -130 and lon <= -60:
        return True

    if lat >= 50 and lat <= 70 and lon >= -180 and lon <= -130:
        return True

    if lat >= -10 and lat <= 60 and lon_360 >= 130 and lon_360 <= 180:
        return True

    if lat >= -50 and lat <= -30 and lon_360 >= 160 and lon_360 <= 180:
        return True

    if lat >= -30 and lat <= 30 and lon_360 >= 120 and lon_360 <= 160:
        return True

    return False

def generate_synthetic_data(original_df, num_samples=800):
    synthetic_data = []

    numerical_cols = ['Latitude', 'Longitude', 'Past_EQ_Count', 'Avg_Magnitude', 'Avg_Depth_km',
                     'Seismic_Activity_Index', 'Soil_Stability', 'Population_Density', 'Years_Since_Last_EQ']
    target_col = 'Prob_Major_EQ_Next5Yrs'
    categorical_col = 'Tectonic_Region'

    stats = {}
    for col in numerical_cols + [target_col]:
        stats[col] = {
            'mean': original_df[col].mean(),
            'std': original_df[col].std(),
            'min': original_df[col].min(),
            'max': original_df[col].max(),
            'median': original_df[col].median()
        }

    tectonic_counts = original_df[categorical_col].value_counts()
    tectonic_probs = tectonic_counts / tectonic_counts.sum()

    for _ in range(num_samples):
        # Generate tectonic region first
        tectonic_region = np.random.choice(tectonic_counts.index, p=tectonic_probs)

        # Generate numerical features with some correlation considerations
        row = {}

        # Latitude and Longitude - global distribution without bias
        row['Latitude'] = np.random.normal(stats['Latitude']['mean'], stats['Latitude']['std'])
        row['Longitude'] = np.random.normal(stats['Longitude']['mean'], stats['Longitude']['std'])
        # Clip to realistic global ranges
        row['Latitude'] = np.clip(row['Latitude'], -90, 90)
        row['Longitude'] = np.clip(row['Longitude'], -180, 180)

        # Past_EQ_Count - depends on tectonic region
        if tectonic_region == 'Subduction Zone':
            row['Past_EQ_Count'] = np.random.normal(70, 20)
        elif tectonic_region == 'Transform Fault':
            row['Past_EQ_Count'] = np.random.normal(50, 15)
        elif tectonic_region == 'Divergent Boundary':
            row['Past_EQ_Count'] = np.random.normal(40, 12)
        else:  # Intraplate
            row['Past_EQ_Count'] = np.random.normal(25, 10)

        # Ensure reasonable bounds
        row['Past_EQ_Count'] = np.clip(row['Past_EQ_Count'], 0, 100)

        # Avg_Magnitude - correlated with Past_EQ_Count and tectonic region
        base_magnitude = 4.0 + (row['Past_EQ_Count'] / 100) * 3.0
        if tectonic_region == 'Subduction Zone':
            base_magnitude += 1.0
        row['Avg_Magnitude'] = np.random.normal(base_magnitude, 1.2)
        row['Avg_Magnitude'] = np.clip(row['Avg_Magnitude'], 2.0, 8.0)

        # Avg_Depth_km - varies by tectonic region
        if tectonic_region == 'Subduction Zone':
            row['Avg_Depth_km'] = np.random.normal(400, 150)
        elif tectonic_region == 'Transform Fault':
            row['Avg_Depth_km'] = np.random.normal(200, 100)
        elif tectonic_region == 'Divergent Boundary':
            row['Avg_Depth_km'] = np.random.normal(100, 50)
        else:  # Intraplate
            row['Avg_Depth_km'] = np.random.normal(300, 120)

        row['Avg_Depth_km'] = np.clip(row['Avg_Depth_km'], 5, 700)

        # Seismic_Activity_Index - correlated with Past_EQ_Count
        row['Seismic_Activity_Index'] = 0.2 + (row['Past_EQ_Count'] / 100) * 0.7 + np.random.normal(0, 0.1)
        row['Seismic_Activity_Index'] = np.clip(row['Seismic_Activity_Index'], 0.1, 0.9)

        # Soil_Stability - somewhat random but with regional patterns
        row['Soil_Stability'] = np.random.randint(1, 11)

        # Population_Density - varies globally
        row['Population_Density'] = np.random.normal(stats['Population_Density']['mean'],
                                                   stats['Population_Density']['std'])
        row['Population_Density'] = np.clip(row['Population_Density'], 0, 10000)

        # Years_Since_Last_EQ - inversely correlated with Seismic_Activity_Index
        base_years = 50 - (row['Seismic_Activity_Index'] * 40)
        row['Years_Since_Last_EQ'] = np.random.normal(base_years, 10)
        row['Years_Since_Last_EQ'] = np.clip(row['Years_Since_Last_EQ'], 0, 50)

        # Tectonic_Region
        row['Tectonic_Region'] = tectonic_region

        # Prob_Major_EQ_Next5Yrs - complex relationship with other factors
        risk_score = (row['Past_EQ_Count'] / 100 * 0.3 +
                     row['Seismic_Activity_Index'] * 0.4 +
                     (50 - row['Years_Since_Last_EQ']) / 50 * 0.2 +
                     (row['Avg_Magnitude'] - 2) / 6 * 0.1)

        # Base risk by tectonic region
        if tectonic_region == 'Intraplate':
            risk_score += 0.01  # Very low base risk
        elif tectonic_region == 'Divergent Boundary':
            risk_score += 0.05
        elif tectonic_region == 'Transform Fault':
            risk_score += 0.1
        elif tectonic_region == 'Subduction Zone':
            risk_score += 0.2

        row['Prob_Major_EQ_Next5Yrs'] = np.clip(risk_score + np.random.normal(0, 0.1), 0.01, 0.9)

        synthetic_data.append(row)

    return pd.DataFrame(synthetic_data)

synthetic_df = generate_synthetic_data(df, 800)

expanded_df = pd.concat([df, synthetic_df], ignore_index=True)

if 'Tectonic_Region_encoded' in expanded_df.columns:
    expanded_df = expanded_df.drop('Tectonic_Region_encoded', axis=1)

expanded_df.to_csv("../../../../Downloads/expanded_earthquake_prediction_data.csv", index=False)

print(f"Original dataset: {len(df)} rows")
print(f"Synthetic data: {len(synthetic_df)} rows")
print(f"Expanded dataset: {len(expanded_df)} rows")
print("Expanded dataset saved to ../../../../Downloads/expanded_earthquake_prediction_data.csv")

print("\nTectonic Region Distribution in Expanded Dataset:")
print(expanded_df['Tectonic_Region'].value_counts(normalize=True))
print("\nProbability Distribution Summary:")
print(expanded_df['Prob_Major_EQ_Next5Yrs'].describe())
