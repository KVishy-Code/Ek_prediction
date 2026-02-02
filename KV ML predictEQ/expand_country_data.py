import pandas as pd
import random

# Read the existing CSV
df = pd.read_csv("../../../../Downloads/curated_lat_long_country.csv")

# Get unique countries
countries = df['country'].unique()

# Expand data for each country to reach at least 250 points
expanded_data = []

# First, add all existing data
for _, row in df.iterrows():
    expanded_data.append({
        'latitude': row['latitude'],
        'longitude': row['longitude'],
        'country': row['country']
    })

# For each country, add more data points
for country in countries:
    current_count = len(df[df['country'] == country])
    needed = max(0, 250 - current_count)

    print(f"Expanding {country}: {current_count} -> {current_count + needed}")

    # Get existing coordinates for this country
    existing_coords = df[df['country'] == country][['latitude', 'longitude']].values

    # Calculate approximate bounds from existing data
    if len(existing_coords) > 0:
        lat_min = existing_coords[:, 0].min() - 5
        lat_max = existing_coords[:, 0].max() + 5
        lon_min = existing_coords[:, 1].min() - 5
        lon_max = existing_coords[:, 1].max() + 5

        # Generate additional points within extended bounds
        for i in range(needed):
            # Use existing point as base and add variation
            base_idx = random.randint(0, len(existing_coords) - 1)
            base_lat, base_lon = existing_coords[base_idx]

            # Add larger random variation to spread points across country
            lat_variation = random.uniform(-3, 3)
            lon_variation = random.uniform(-3, 3)

            lat = base_lat + lat_variation
            lon = base_lon + lon_variation

            # Keep within reasonable bounds
            lat = max(lat_min, min(lat_max, lat))
            lon = max(lon_min, min(lon_max, lon))

            expanded_data.append({
                'latitude': lat,
                'longitude': lon,
                'country': country
            })

# Create new dataframe
expanded_df = pd.DataFrame(expanded_data)

# Save to the same file (expanding it)
expanded_df.to_csv("../../../../Downloads/curated_lat_long_country.csv", index=False)

print(f"Original dataset: {len(df)} rows")
print(f"Expanded dataset: {len(expanded_df)} rows")
print("\nFinal counts per country:")
print(expanded_df['country'].value_counts())
