import numpy as np
import pandas as pd
import folium
import heapq
import os
from scipy.interpolate import CubicSpline

### ----------------------- LOAD DATA -----------------------
DATA_PATH = 'analysed_data/terrain_data.jsonl'  # Ensure this JSONL has lat, lon, elevation, slope, ndvi, etc.
df = pd.read_json(DATA_PATH, lines=True)

### ----------------------- GRID CONVERSION -----------------------

df = df.dropna(subset=["elevation", "slope", "ndvi"])
df = df.sort_values(by=["lat", "lon"]).reset_index(drop=True)

# Map lat/lon to row/col
df["lat_idx"] = df["lat"].rank(method="dense").astype(int) - 1
df["lon_idx"] = df["lon"].rank(method="dense").astype(int) - 1

nrows = df["lat_idx"].max() + 1
ncols = df["lon_idx"].max() + 1

# Initialize terrain matrices
elevation_grid = np.full((nrows, ncols), np.nan)
slope_grid = np.full((nrows, ncols), np.nan)
ndvi_grid = np.full((nrows, ncols), np.nan)

for _, row in df.iterrows():
    r, c = int(row["lat_idx"]), int(row["lon_idx"])
    elevation_grid[r, c] = row["elevation"]
    slope_grid[r, c] = row["slope"]
    ndvi_grid[r, c] = row["ndvi"]

### ----------------------- NORMALIZE TERRAIN DATA -----------------------

def normalize_grid(grid):
    """Normalize grid values between 0 and 1, ignoring NaN values"""
    valid_mask = ~np.isnan(grid)
    result = np.copy(grid)
    if np.any(valid_mask):
        valid_values = grid[valid_mask]
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)
        if max_val > min_val:
            result[valid_mask] = (valid_values - min_val) / (max_val - min_val)
    return result

# Normalize terrain data
elevation_norm = normalize_grid(elevation_grid)
slope_norm = normalize_grid(slope_grid)
ndvi_norm = normalize_grid(ndvi_grid)

### ----------------------- TACTICAL POINT DETECTION -----------------------

def detect_hiding_spots(ndvi_norm, slope_norm):
    # Focus on HIGH vegetation (NDVI) for hiding spots
    # Dense vegetation provides better concealment
    hiding_mask = (ndvi_norm > 0.6) & (slope_norm < 0.8)
    return np.argwhere(hiding_mask)

def detect_choke_points(elevation_norm, slope_norm):
    """
    Detect strategic choke points based on terrain features.
    Choke points are narrow passages where terrain funnels movement.
    
    Strategic choke points have:
    1. Elevation gradients that indicate terrain funnel effects
    2. Moderate slope (still traversable)
    3. Located in passages between higher terrain
    """
    # Calculate gradient in both directions
    grad_y, grad_x = np.gradient(elevation_norm)
    
    # Calculate magnitude of gradient
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient magnitude
    grad_mag_norm = normalize_grid(gradient_magnitude)
    
    # Create mask for gradient changes - LOWERED THRESHOLD
    high_gradient_mask = grad_mag_norm > 0.45  # Reduced from 0.6
    
    # Calculate local "corridor" effect - areas with high gradients on multiple sides
    corridor_mask = np.zeros_like(high_gradient_mask, dtype=bool)
    
    # Examine 3×3 neighborhoods to find points with high gradients on multiple sides
    for i in range(1, high_gradient_mask.shape[0]-1):
        for j in range(1, high_gradient_mask.shape[1]-1):
            # Skip if central point has too high slope (impassable) - INCREASED ALLOWABLE SLOPE
            if slope_norm[i, j] > 0.8:  # Increased from 0.7
                continue
                
            # Get neighborhood
            neighborhood = high_gradient_mask[i-1:i+2, j-1:j+2]
            
            # Count high gradient neighbors
            gradient_neighbor_count = np.sum(neighborhood) - neighborhood[1, 1]
            
            # Identify as corridor if high gradients on multiple sides
            # REDUCED required neighbor count from 2 to 1
            if gradient_neighbor_count >= 1:  # Reduced from 2
                corridor_mask[i, j] = True
    
    # Further filter to identify true choke points - LOOSENED CRITERIA
    choke_mask = corridor_mask & (slope_norm < 0.8) & (grad_mag_norm > 0.4)  # Removed lower slope bound, reduced gradient threshold
    
    # Thin out choke points to avoid overcrowding
    # Use a more targeted approach with looser spacing
    thinned_mask = np.zeros_like(choke_mask)
    min_distance = 3  # Reduced from 4
    
    # Find all potential choke points
    choke_points = np.argwhere(choke_mask)
    
    # Start with highest gradient magnitude points
    choke_values = [(grad_mag_norm[r, c], r, c) for r, c in choke_points]
    choke_values.sort(reverse=True)  # Sort by gradient magnitude (highest first)
    
    # Keep track of selected points
    selected_points = []
    
    # Select points while maintaining minimum distance
    for _, r, c in choke_values:
        # Check if this point is far enough from already selected points
        too_close = False
        for sr, sc in selected_points:
            if np.sqrt((r - sr)**2 + (c - sc)**2) < min_distance:
                too_close = True
                break
        
        if not too_close:
            thinned_mask[r, c] = True
            selected_points.append((r, c))
    
    return np.argwhere(thinned_mask)

def detect_checkpoints(elevation_norm, slope_norm, ndvi_norm):
    # Surveillance points need:
    # 1. Medium to high elevation (for good visibility)
    # 2. Low to moderate slope (to be able to stand/position)
    # 3. Low to moderate vegetation (for better visibility)
    checkpoint_mask = (elevation_norm > 0.6) & (slope_norm < 0.4) & (ndvi_norm < 0.5)
    return np.argwhere(checkpoint_mask)

### ----------------------- A* PATHFINDING -----------------------

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def create_cost_grid(elevation_norm, slope_norm, ndvi_norm, path_type):
    """
    Create different cost grids for different path types:
    - easy: prefer low slope, avoid high elevation
    - balanced: moderate weights for all factors
    - tough: favor areas with high vegetation (potential hiding spots), prefer difficult terrain
    """
    # Initialize with NaN where original data was NaN
    cost_grid = np.full_like(elevation_norm, np.nan)
    valid_mask = ~np.isnan(elevation_norm)
    
    if path_type == "easy":
        # Easy path: strongly avoid slopes, moderately avoid elevation, don't care much about vegetation
        cost_grid[valid_mask] = (0.7 * slope_norm[valid_mask] + 
                               0.2 * elevation_norm[valid_mask] + 
                               0.1 * (1 - ndvi_norm[valid_mask]))  # Lower vegetation might mean clearer paths
    
    elif path_type == "balanced":
        # Balanced path: equal weights to all factors
        cost_grid[valid_mask] = (0.4 * slope_norm[valid_mask] + 
                               0.3 * elevation_norm[valid_mask] + 
                               0.3 * (1 - ndvi_norm[valid_mask]))
    
    elif path_type == "tough":
        # Tough path: favor high vegetation (hiding spots), less concerned with slopes
        # This is a path an enemy might take to remain concealed
        cost_grid[valid_mask] = (0.2 * slope_norm[valid_mask] + 
                               0.2 * elevation_norm[valid_mask] + 
                               0.6 * (1 - ndvi_norm[valid_mask]))  # Invert NDVI to make high vegetation preferable
                               
    # Normalize the cost grid again to ensure values are between 0 and 1
    cost_grid = normalize_grid(cost_grid)
    
    # Scale up to make the algorithm work better (A* works with positive costs)
    cost_grid = cost_grid * 10 + 1  # Add 1 to avoid zero costs
    
    # Make NaN areas impassable
    cost_grid[~valid_mask] = 1000
    
    return cost_grid

def astar(cost_grid, start, goal):
    nrows, ncols = cost_grid.shape
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx in [-1, 0, 1] for dy in [-1, 0, 1]
            if not (dx == 0 and dy == 0)
        ]

        for neighbor in neighbors:
            x, y = neighbor
            if 0 <= x < nrows and 0 <= y < ncols and not np.isnan(cost_grid[x, y]):
                # Calculate movement cost (diagonal moves cost more)
                movement_cost = 1.4 if abs(current[0] - x) + abs(current[1] - y) == 2 else 1.0
                neighbor_cost = cost_grid[x, y] * movement_cost
                
                tentative_g = g_score[current] + neighbor_cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    return []  # no path found

### ----------------------- FIND PATHS -----------------------

# Find valid start and end indices
min_lat_idx = df['lat_idx'].min()
min_lon_idx = df['lon_idx'].min()
max_lat_idx = df['lat_idx'].max()
max_lon_idx = df['lon_idx'].max()

start = (min_lat_idx, min_lon_idx)
end = (max_lat_idx, max_lon_idx)

# Calculate paths for each type
easy_cost_grid = create_cost_grid(elevation_norm, slope_norm, ndvi_norm, "easy")
balanced_cost_grid = create_cost_grid(elevation_norm, slope_norm, ndvi_norm, "balanced")
tough_cost_grid = create_cost_grid(elevation_norm, slope_norm, ndvi_norm, "tough")

easy_path = astar(easy_cost_grid, start, end)
balanced_path = astar(balanced_cost_grid, start, end)
tough_path = astar(tough_cost_grid, start, end)

# Get start/end coordinates safely
def get_lat_lon(idx_tuple):
    subset = df[(df.lat_idx == idx_tuple[0]) & (df.lon_idx == idx_tuple[1])][["lat", "lon"]]
    if not subset.empty:
        return subset.values[0]
    else:
        return (None, None)

start_lat, start_lon = get_lat_lon(start)
end_lat, end_lon = get_lat_lon(end)

### ----------------------- MAPPING -----------------------

m = folium.Map(location=[start_lat, start_lon], zoom_start=15)  # Adjusted zoom level

# Add start and end markers
if start_lat is not None and start_lon is not None:
    folium.Marker([start_lat, start_lon], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
if end_lat is not None and end_lon is not None:
    folium.Marker([end_lat, end_lon], tooltip="End", icon=folium.Icon(color="red")).add_to(m)

# Add tactical points to the map
# Hiding Spots
hiding = detect_hiding_spots(ndvi_norm, slope_norm)
hiding_count = 0
for r, c in hiding:
    subset = df[(df.lat_idx == r) & (df.lon_idx == c)][["lat", "lon"]]
    if not subset.empty:
        lat, lon = subset.values[0]
        tooltip = f"Hiding Spot\nLat: {lat:.5f}, Lon: {lon:.5f}"
        folium.CircleMarker(
            [lat, lon], radius=5, color="blue", fill=True, fill_opacity=0.5,
            tooltip=tooltip
        ).add_to(m)
        hiding_count += 1

# Choke Points
choke = detect_choke_points(elevation_norm, slope_norm)
choke_count = 0
for r, c in choke:
    subset = df[(df.lat_idx == r) & (df.lon_idx == c)][["lat", "lon"]]
    if not subset.empty:
        lat, lon = subset.values[0]
        tooltip = f"Choke Point\nLat: {lat:.5f}, Lon: {lon:.5f}"
        folium.CircleMarker(
            [lat, lon], radius=5, color="orange", fill=True, fill_opacity=0.5,
            tooltip=tooltip
        ).add_to(m)
        choke_count += 1

# Surveillance Points
checkpoints = detect_checkpoints(elevation_norm, slope_norm, ndvi_norm)
checkpoint_count = 0
for r, c in checkpoints:
    subset = df[(df.lat_idx == r) & (df.lon_idx == c)][["lat", "lon"]]
    if not subset.empty:
        lat, lon = subset.values[0]
        tooltip = f"Surveillance Point\nLat: {lat:.5f}, Lon: {lon:.5f}"
        folium.CircleMarker(
            [lat, lon], radius=5, color="purple", fill=True, fill_opacity=0.5,
            tooltip=tooltip
        ).add_to(m)
        checkpoint_count += 1


# Helper function to draw a path with smoothing
def draw_path(path, color, label):
    if not path:
        return
    
    # Get coordinates for path points
    path_coords = []
    for p in path:
        lat, lon = get_lat_lon(p)
        if lat is not None and lon is not None:
            path_coords.append([lat, lon])
    
    if len(path_coords) > 2:
        # Extract coordinates for spline interpolation
        lats = [p[0] for p in path_coords]
        lons = [p[1] for p in path_coords]
        
        # Create spline interpolation
        t = np.linspace(0, 1, len(path_coords))
        lat_spline = CubicSpline(t, lats)
        lon_spline = CubicSpline(t, lons)
        
        # Generate smooth path
        t_smooth = np.linspace(0, 1, 100)
        smooth_lats = lat_spline(t_smooth)
        smooth_lons = lon_spline(t_smooth)
        
        # Create smooth path coordinates
        smooth_path = [[lat, lon] for lat, lon in zip(smooth_lats, smooth_lons)]
        
        # Draw the smooth path
        folium.PolyLine(
            smooth_path, 
            color=color, 
            weight=3.5, 
            opacity=0.8,
            tooltip=label
        ).add_to(m)
    else:
        # If not enough points for spline, draw direct path
        folium.PolyLine(
            path_coords, 
            color=color, 
            weight=3.5,
            tooltip=label
        ).add_to(m)

# Draw all three paths
draw_path(easy_path, "green", "Easy Path")
draw_path(balanced_path, "blue", "Balanced Path")
draw_path(tough_path, "red", "Tough Path (Enemy Route)")

# Add legend to the map
legend_html = '''
<div style="position: fixed; 
     bottom: 50px; right: 50px; width: 180px; height: 210px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; padding: 10px;
     ">
     <p><strong>Path Types</strong></p>
     <p><span style="color:green; font-weight:bold;">●</span> Easy Path</p>
     <p><span style="color:blue; font-weight:bold;">●</span> Balanced Path</p>
     <p><span style="color:red; font-weight:bold;">●</span> Tough/Enemy Path</p>
     <hr>
     <p><span style="color:blue; font-weight:bold;">●</span> Hiding Spot</p>
     <p><span style="color:orange; font-weight:bold;">●</span> Choke Point</p>
     <p><span style="color:purple; font-weight:bold;">●</span> Surveillance Point</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname("outputs/map.html"), exist_ok=True)
m.save("outputs/map.html")

# Open the map automatically in the default web browser
import webbrowser
map_path = os.path.abspath("outputs/map.html")
webbrowser.open('file://' + map_path)

# Print summary information
print("\n✅ Terrain Analysis Complete!")
print("----------------------------")
print(f"Map generated at: outputs/map.html")
print(f"Tactical Points Detected:")
print(f"  - Hiding Spots: {hiding_count}")
print(f"  - Choke Points: {choke_count}")
print(f"  - Surveillance Points: {checkpoint_count}")
print(f"Paths Generated:")
print(f"  - Green: Easy path ({len(easy_path)} points)")
print(f"  - Blue: Balanced path ({len(balanced_path)} points)")
print(f"  - Red: Tough path ({len(tough_path)} points, potential enemy route)")
print("----------------------------")

### ----------------------- SAVE TACTICAL DATA -----------------------
import json

def get_coordinates_from_indices(indices):
    """Convert grid indices to lat/lon coordinates"""
    coordinates = []
    for r, c in indices:
        subset = df[(df.lat_idx == r) & (df.lon_idx == c)][["lat", "lon"]]
        if not subset.empty:
            coordinates.append(subset.values[0].tolist())
    return coordinates

tactical_data = {
    "paths": {
        "easy": get_coordinates_from_indices(easy_path),
        "balanced": get_coordinates_from_indices(balanced_path),
        "tough": get_coordinates_from_indices(tough_path)
    },
    "tactical_points": {
        "hiding_spots": get_coordinates_from_indices(hiding),
        "choke_points": get_coordinates_from_indices(choke),
        "surveillance_points": get_coordinates_from_indices(checkpoints)
    }
}

# Save to JSON file
os.makedirs("outputs", exist_ok=True)
with open("outputs/tactical_data.json", "w") as f:
    json.dump(tactical_data, f, indent=2)

print("✅ Tactical data saved to outputs/tactical_data.json")