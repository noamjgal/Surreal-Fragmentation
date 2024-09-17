import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import trackintel as ti
from trackintel.preprocessing.positionfixes import generate_staypoints
from trackintel.preprocessing.staypoints import generate_locations
from trackintel.analysis.tracking_quality import temporal_tracking_quality

# Load the data
print("Loading data...")
df = pd.read_excel('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/gpsappS_9.1_excel.xlsx', sheet_name='gpsappS_8')

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df['long'], df['lat']),
    crs="EPSG:4326"
)

# Create positionfixes
pfs = ti.Positionfixes(gdf.rename(columns={
    'user': 'user_id',
    'Timestamp': 'tracked_at'
}))

# Ensure datetime format
pfs['tracked_at'] = pd.to_datetime(pfs['tracked_at'])

# Generate staypoints
print("Generating staypoints...")
pfs, sp = generate_staypoints(pfs, method='sliding', dist_threshold=100, time_threshold=5 * 60)

# Generate triplegs
print("Generating triplegs...")
tpls = pfs.generate_triplegs(staypoints=sp)

# Generate locations
print("Generating locations...")
sp, locs = generate_locations(sp, method='dbscan', epsilon=100, num_samples=3)

# Predict transport modes
print("Predicting transport modes...")
tpls['mode'] = tpls.predict_transport_mode()

# Calculate temporal tracking quality
print("Calculating temporal tracking quality...")
quality = temporal_tracking_quality(tpls, granularity='all')

# Analyze trips
print("Analyzing trips...")
trips = tpls.generate_trips(sp)
trips['duration'] = (trips['finished_at'] - trips['started_at']).dt.total_seconds() / 60  # in minutes
trips['distance'] = trips.length / 1000  # in km

# Calculate modal split
print("Calculating modal split...")
modal_split = tpls.modal_split(mode='mode')

# Visualization
print("Creating visualizations...")

# Plot modal split
plt.figure(figsize=(10, 6))
modal_split.plot(kind='bar')
plt.title('Modal Split')
plt.xlabel('Transport Mode')
plt.ylabel('Proportion')
plt.tight_layout()
plt.savefig('modal_split.png')
plt.close()

# Plot trip distance distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=trips, x='distance', bins=30, kde=True)
plt.title('Trip Distance Distribution')
plt.xlabel('Distance (km)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('trip_distance_distribution.png')
plt.close()

# Plot trip duration distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=trips, x='duration', bins=30, kde=True)
plt.title('Trip Duration Distribution')
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('trip_duration_distribution.png')
plt.close()

# Plot tracking quality distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=quality, x='quality', bins=30, kde=True)
plt.title('Tracking Quality Distribution')
plt.xlabel('Tracking Quality')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('tracking_quality_distribution.png')
plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print(f"Total number of trips: {len(trips)}")
print(f"Average trip distance: {trips['distance'].mean():.2f} km")
print(f"Average trip duration: {trips['duration'].mean():.2f} minutes")
print(f"Average tracking quality: {quality['quality'].mean():.2f}")
print(f"\nModal Split:\n{modal_split}")

print("Analysis complete. Results and visualizations have been saved.")