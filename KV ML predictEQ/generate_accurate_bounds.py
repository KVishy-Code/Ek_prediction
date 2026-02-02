import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import requests
import zipfile
import io

# Download Natural Earth data
url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip"
response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
zip_file.extractall("temp_natural_earth")
world = gpd.read_file("temp_natural_earth/ne_110m_admin_0_countries.shp")

# Create a dictionary to store accurate bounds for each country
country_bounds = {}

for idx, row in world.iterrows():
    country_name = row['name']
    geometry = row['geometry']

    # Get the bounding box
    bounds = geometry.bounds  # (min_lon, min_lat, max_lon, max_lat)

    # Store as (center_lat, center_lon, lat_range, lon_range)
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    lat_range = bounds[3] - bounds[1]
    lon_range = bounds[2] - bounds[0]

    country_bounds[country_name] = {
        'center_lat': center_lat,
        'center_lon': center_lon,
        'lat_min': bounds[1],
        'lat_max': bounds[3],
        'lon_min': bounds[0],
        'lon_max': bounds[2],
        'lat_range': lat_range,
        'lon_range': lon_range
    }

# Print some examples
print("Accurate country bounds from Natural Earth:")
for country in ['United States', 'Russia', 'Japan', 'Singapore', 'France']:
    if country in country_bounds:
        bounds = country_bounds[country]
        print(f"{country}:")
        print(f"  Center: ({bounds['center_lat']:.2f}, {bounds['center_lon']:.2f})")
        print(f"  Lat range: {bounds['lat_min']:.2f} to {bounds['lat_max']:.2f} ({bounds['lat_range']:.2f}°)")
        print(f"  Lon range: {bounds['lon_min']:.2f} to {bounds['lon_max']:.2f} ({bounds['lon_range']:.2f}°)")
        print()

# Save to a CSV for reference
bounds_df = pd.DataFrame.from_dict(country_bounds, orient='index')
bounds_df.to_csv("country_bounds_natural_earth.csv")
print("Saved accurate bounds to country_bounds_natural_earth.csv")
