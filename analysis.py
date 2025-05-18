import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import heapq
import os
from scipy.interpolate import CubicSpline
import random
import math
from folium.features import DivIcon

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
    # LOOSENED CRITERIA: Focus on higher vegetation (NDVI) for hiding spots
    # Reduced threshold from 0.6 to 0.5 for NDVI
    # Increased allowable slope from 0.8 to 0.9
    hiding_mask = (ndvi_norm > 0.5) & (slope_norm < 0.9)
    return np.argwhere(hiding_mask)


def detect_choke_points(elevation_norm, slope_norm):
    """
    Detect strategic choke points based on terrain features.
    LOOSENED CRITERIA for choke points
    """
    # Calculate gradient in both directions
    grad_y, grad_x = np.gradient(elevation_norm)

    # Calculate magnitude of gradient
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize gradient magnitude
    grad_mag_norm = normalize_grid(gradient_magnitude)

    # Create mask for gradient changes - FURTHER LOWERED THRESHOLD
    high_gradient_mask = grad_mag_norm > 0.4  # Reduced from 0.45

    # Calculate local "corridor" effect - areas with high gradients on multiple sides
    corridor_mask = np.zeros_like(high_gradient_mask, dtype=bool)

    # Examine 3×3 neighborhoods to find points with high gradients on multiple sides
    for i in range(1, high_gradient_mask.shape[0] - 1):
        for j in range(1, high_gradient_mask.shape[1] - 1):
            # Skip if central point has too high slope (impassable) - INCREASED ALLOWABLE SLOPE
            if slope_norm[i, j] > 0.9:  # Increased from 0.8
                continue

            # Get neighborhood
            neighborhood = high_gradient_mask[i - 1:i + 2, j - 1:j + 2]

            # Count high gradient neighbors
            gradient_neighbor_count = np.sum(neighborhood) - neighborhood[1, 1]

            # Identify as corridor if high gradients on multiple sides
            # No change to required neighbor count (already at minimum 1)
            if gradient_neighbor_count >= 1:
                corridor_mask[i, j] = True

    # Further filter to identify true choke points - FURTHER LOOSENED CRITERIA
    choke_mask = corridor_mask & (slope_norm < 0.9) & (grad_mag_norm > 0.35)  # Reduced gradient threshold even more

    # Thin out choke points to avoid overcrowding
    # Use a more targeted approach with looser spacing
    thinned_mask = np.zeros_like(choke_mask)
    min_distance = 2  # Reduced from 3 to get more points

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
            if np.sqrt((r - sr) ** 2 + (c - sc) ** 2) < min_distance:
                too_close = True
                break

        if not too_close:
            thinned_mask[r, c] = True
            selected_points.append((r, c))

    return np.argwhere(thinned_mask)


def detect_checkpoints(elevation_norm, slope_norm, ndvi_norm):
    # Surveillance points need:
    # LOOSENED CRITERIA:
    # 1. Lowered elevation threshold from 0.6 to 0.5
    # 2. Increased slope allowance from 0.4 to 0.5
    # 3. Increased vegetation allowance from 0.5 to 0.6
    checkpoint_mask = (elevation_norm > 0.5) & (slope_norm < 0.5) & (ndvi_norm < 0.6)
    return np.argwhere(checkpoint_mask)


### ----------------------- A* PATHFINDING -----------------------

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def create_cost_grid(elevation_norm, slope_norm, ndvi_norm, path_type, randomness=0.0):
    """
    Create different cost grids for different path types with optional randomness factor
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

    # Add randomness if specified
    if randomness > 0 and valid_mask.any():
        # Generate random values in the range [-randomness, randomness]
        random_values = np.random.uniform(-randomness, randomness, size=cost_grid.shape)
        # Apply random values only to valid cells
        cost_grid[valid_mask] += random_values[valid_mask]
        # Ensure no negative values
        cost_grid[valid_mask] = np.maximum(0.01, cost_grid[valid_mask])

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

# Generate multiple path start/end points for variety
def generate_path_endpoints(min_lat_idx, min_lon_idx, max_lat_idx, max_lon_idx, num_paths=10):  # Changed from 50 to 10
    """Generate a variety of start and end points for multiple paths"""
    endpoints = []

    # First add the original path (furthest corners)
    endpoints.append(((min_lat_idx, min_lon_idx), (max_lat_idx, max_lon_idx)))

    # Generate random start/end pairs with minimum distance requirement
    min_distance = (max_lat_idx - min_lat_idx + max_lon_idx - min_lon_idx) * 0.3  # At least 30% of diagonal

    while len(endpoints) < num_paths:
        # Generate random start and end points
        start = (random.randint(min_lat_idx, max_lat_idx),
                 random.randint(min_lon_idx, max_lon_idx))
        end = (random.randint(min_lat_idx, max_lat_idx),
               random.randint(min_lon_idx, max_lon_idx))

        # Calculate distance between start and end
        distance = math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

        # Only add if distance is sufficient
        if distance >= min_distance:
            endpoints.append((start, end))

    return endpoints


# Find valid start and end indices
min_lat_idx = df['lat_idx'].min()
min_lon_idx = df['lon_idx'].min()
max_lat_idx = df['lat_idx'].max()
max_lon_idx = df['lon_idx'].max()

# Generate 10 different path endpoints instead of 50
path_endpoints = generate_path_endpoints(min_lat_idx, min_lon_idx, max_lat_idx, max_lon_idx, num_paths=10)

# Store all paths
all_paths = {
    "easy": [],
    "balanced": [],
    "tough": []
}

# Calculate multiple paths for each type with slight randomness for variety
for i, (start, end) in enumerate(path_endpoints):
    path_id = f"path_{i + 1}"
    print(f"Calculating {path_id} ({i + 1}/10)...")  # Updated to reflect 10 paths

    # Add randomness factor that decreases with each type
    # More randomness for easy paths, less for tough paths
    easy_randomness = 0.2
    balanced_randomness = 0.15
    tough_randomness = 0.1

    # Create cost grids with randomness
    easy_cost_grid = create_cost_grid(elevation_norm, slope_norm, ndvi_norm, "easy", randomness=easy_randomness)
    balanced_cost_grid = create_cost_grid(elevation_norm, slope_norm, ndvi_norm, "balanced",
                                          randomness=balanced_randomness)
    tough_cost_grid = create_cost_grid(elevation_norm, slope_norm, ndvi_norm, "tough", randomness=tough_randomness)

    # Calculate paths
    easy_path = astar(easy_cost_grid, start, end)
    balanced_path = astar(balanced_cost_grid, start, end)
    tough_path = astar(tough_cost_grid, start, end)

    # Store paths with their respective IDs if they exist
    if easy_path:
        all_paths["easy"].append({"id": f"easy_{i + 1}", "path": easy_path, "start": start, "end": end})
    if balanced_path:
        all_paths["balanced"].append({"id": f"balanced_{i + 1}", "path": balanced_path, "start": start, "end": end})
    if tough_path:
        all_paths["tough"].append({"id": f"tough_{i + 1}", "path": tough_path, "start": start, "end": end})


# Get coordinates for a point
def get_lat_lon(idx_tuple):
    subset = df[(df.lat_idx == idx_tuple[0]) & (df.lon_idx == idx_tuple[1])][["lat", "lon"]]
    if not subset.empty:
        return subset.values[0]
    else:
        return (None, None)


### ----------------------- DETECT TACTICAL POINTS -----------------------

# Detect all tactical points
hiding_spots = detect_hiding_spots(ndvi_norm, slope_norm)
choke_points = detect_choke_points(elevation_norm, slope_norm)
surveillance_points = detect_checkpoints(elevation_norm, slope_norm, ndvi_norm)

# Convert grid indices to coordinates
hiding_coords = []
for r, c in hiding_spots:
    lat, lon = get_lat_lon((r, c))
    if lat is not None and lon is not None:
        hiding_coords.append({"lat": lat, "lon": lon, "type": "hiding", "grid_idx": (r, c)})

choke_coords = []
for r, c in choke_points:
    lat, lon = get_lat_lon((r, c))
    if lat is not None and lon is not None:
        choke_coords.append({"lat": lat, "lon": lon, "type": "choke", "grid_idx": (r, c)})

surveillance_coords = []
for r, c in surveillance_points:
    lat, lon = get_lat_lon((r, c))
    if lat is not None and lon is not None:
        surveillance_coords.append({"lat": lat, "lon": lon, "type": "surveillance", "grid_idx": (r, c)})

# All tactical points
all_tactical_points = hiding_coords + choke_coords + surveillance_coords

### ----------------------- MAPPING -----------------------

m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=13)

# Add tactical point layers (visible by default)
hiding_layer = folium.FeatureGroup(name="Hiding Spots", show=True)
choke_layer = folium.FeatureGroup(name="Choke Points", show=True)
surveillance_layer = folium.FeatureGroup(name="Surveillance Points", show=True)


# Define a function to calculate distance between two points using Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on Earth in kilometers."""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


# Store tactical points in their respective layers
for point in hiding_coords:
    folium.CircleMarker(
        [point["lat"], point["lon"]],
        radius=6,  # Smaller radius
        color="blue",  # Ring color
        fill=True,
        fill_color="blue",  # Same as border color
        fill_opacity=0.0,  # Transparent center
        weight=2.5,  # Ring thickness
        popup=f"Hiding Spot<br>Lat: {point['lat']:.5f}<br>Lon: {point['lon']:.5f}",
        class_name='tactical-point'
    ).add_to(hiding_layer)

for point in choke_coords:
    folium.CircleMarker(
        [point["lat"], point["lon"]],
        radius=6,
        color="orange",
        fill=True,
        fill_color="orange",
        fill_opacity=0.0,
        weight=2.5,
        popup=f"Choke Point<br>Lat: {point['lat']:.5f}<br>Lon: {point['lon']:.5f}",
        class_name='tactical-point'
    ).add_to(choke_layer)

for point in surveillance_coords:
    folium.CircleMarker(
        [point["lat"], point["lon"]],
        radius=6,
        color="purple",
        fill=True,
        fill_color="purple",
        fill_opacity=0.0,
        weight=2.5,
        popup=f"Surveillance Point<br>Lat: {point['lat']:.5f}<br>Lon: {point['lon']:.5f}",
        class_name='tactical-point'
    ).add_to(surveillance_layer)

# Add layers to map
hiding_layer.add_to(m)
choke_layer.add_to(m)
surveillance_layer.add_to(m)


# Helper function to convert path to coordinates
def path_to_coords(path):
    coords = []
    for p in path:
        lat, lon = get_lat_lon(p)
        if lat is not None and lon is not None:
            coords.append([lat, lon])
    return coords


# Helper function to draw a path with smoothing
def draw_path(path_dict, color, path_type, opacity=0.8):
    """Draw a path with given color, type and opacity"""
    path = path_dict["path"]
    path_id = path_dict["id"]

    if not path:
        return None

    # Get coordinates for path points
    path_coords = path_to_coords(path)

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
        polyline = folium.PolyLine(
            smooth_path,
            color=color,
            weight=3.5,
            opacity=opacity,
            tooltip=f"{path_type.capitalize()} Path {path_id}",
            name=path_id
        )

        return polyline
    else:
        # If not enough points for spline, draw direct path
        polyline = folium.PolyLine(
            path_coords,
            color=color,
            weight=3.5,
            opacity=opacity,
            tooltip=f"{path_type.capitalize()} Path {path_id}",
            name=path_id
        )

        return polyline


# Create path feature groups
easy_paths_group = folium.FeatureGroup(name="Easy Paths", show=True)
balanced_paths_group = folium.FeatureGroup(name="Balanced Paths", show=True)
tough_paths_group = folium.FeatureGroup(name="Tough/Enemy Paths", show=True)

# Draw paths with appropriate opacity
# Tough paths are fully opaque, others are translucent by default
for path_dict in all_paths["easy"]:
    path_layer = draw_path(path_dict, "green", "Easy", opacity=0.3)
    if path_layer:
        path_layer.add_to(easy_paths_group)

for path_dict in all_paths["balanced"]:
    path_layer = draw_path(path_dict, "blue", "Balanced", opacity=0.3)
    if path_layer:
        path_layer.add_to(balanced_paths_group)

for path_dict in all_paths["tough"]:
    path_layer = draw_path(path_dict, "red", "Tough", opacity=0.8)  # Fully opaque
    if path_layer:
        path_layer.add_to(tough_paths_group)

# Add path groups to map
easy_paths_group.add_to(m)
balanced_paths_group.add_to(m)
tough_paths_group.add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Generate custom JavaScript for interactivity
js_function = """
// Create map for storing path data
const pathData = {};

// Store all tactical points
const tacticalPoints = {};

// Function to calculate distance between two points in kilometers
function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Radius of the Earth in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
            Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
}

// Function to check if a point is within distance of a path
function isPointNearPath(point, pathCoords, maxDistanceKm) {
    for (const pathPoint of pathCoords) {
        const distance = calculateDistance(
            point.lat, point.lon, 
            pathPoint[0], pathPoint[1]
        );
        if (distance <= maxDistanceKm) {
            return true;
        }
    }
    return false;
}

// Function to reset all paths and points
function resetAllPaths() {
    // Reset all paths to translucent except tough paths
    Object.keys(pathData).forEach(pathId => {
        const path = pathData[pathId];
        if (pathId.startsWith('easy') || pathId.startsWith('balanced')) {
            path.setStyle({opacity: 0.3});
        } else if (pathId.startsWith('tough')) {
            path.setStyle({opacity: 0.8});
        }
    });

    // Hide all tactical points
    document.querySelectorAll('.tactical-point').forEach(marker => {
        marker.style.display = 'none';
    });
}

// Function to handle path click
function handlePathClick(e) {
    const pathId = e.target.options.name;
    const pathCoords = e.target.getLatLngs().map(ll => [ll.lat, ll.lng]);

    // Reset all paths first
    resetAllPaths();

    // Make clicked path fully opaque
    e.target.setStyle({opacity: 1.0});

    // Show only tactical points within 5km of this path
    Object.keys(tacticalPoints).forEach(pointId => {
        const point = tacticalPoints[pointId];
        const nearPath = isPointNearPath(point, pathCoords, 5);

        if (nearPath) {
            document.getElementById(pointId).style.display = 'block';
        } else {
            document.getElementById(pointId).style.display = 'none';
        }
    });
}

// Add click handlers to all paths
document.addEventListener('DOMContentLoaded', function() {
    // Registry for all path polylines
    const polylines = document.querySelectorAll('path.leaflet-interactive');

    polylines.forEach(function(polyline) {
        if (polyline._leaflet_id) {
            // Add click handler
            const leafletObj = window.map._layers[polyline._leaflet_id];
            if (leafletObj && leafletObj.options && leafletObj.options.name) {
                const pathId = leafletObj.options.name;
                pathData[pathId] = leafletObj;
                leafletObj.on('click', handlePathClick);
            }
        }
    });

    // Find all tactical point markers
    const markers = document.querySelectorAll('.leaflet-marker-icon:not(.leaflet-div-icon)');
    markers.forEach(function(marker) {
        if (marker._leaflet_id) {
            const markerId = 'tactical-point-' + marker._leaflet_id;
            marker.id = markerId;
            marker.classList.add('tactical-point');
            marker.style.display = 'none';  // Hide initially

            // Try to get coordinates
            const leafletObj = window.map._layers[marker._leaflet_id];
            if (leafletObj && leafletObj._latlng) {
                tacticalPoints[markerId] = {
                    lat: leafletObj._latlng.lat,
                    lon: leafletObj._latlng.lng
                };
            }
        }
    });
});
"""

# Add the JavaScript to the map
m.get_root().html.add_child(folium.Element(f"""
<script>
window.map = {m.get_name()};
{js_function}
</script>
"""))

# Add legend to the map
legend_html = '''
<div style="position: fixed; 
     bottom: 50px; right: 50px; width: 200px; height: 230px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; padding: 10px;
     ">
     <p><strong>Path Types</strong></p>
     <p><span style="color:green; font-weight:bold;">●</span> Easy Path (Click to select)</p>
     <p><span style="color:blue; font-weight:bold;">●</span> Balanced Path (Click to select)</p>
     <p><span style="color:red; font-weight:bold;">●</span> Tough/Enemy Path (Click to select)</p>
     <hr>
     <p><span style="color:blue; font-weight:bold;">●</span> Hiding Spot</p>
     <p><span style="color:orange; font-weight:bold;">●</span> Choke Point</p>
     <p><span style="color:purple; font-weight:bold;">●</span> Surveillance Point</p>
     <hr>
     <p><small><i>Click any path to show tactical points within 5km</i></small></p>
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
print(f"  - Hiding Spots: {len(hiding_coords)}")
print(f"  - Choke Points: {len(choke_coords)}")
print(f"  - Surveillance Points: {len(surveillance_coords)}")
print(f"Paths Generated:")
print(f"  - Green: Easy paths ({len(all_paths['easy'])} paths)")
print(f"  - Blue: Balanced paths ({len(all_paths['balanced'])} paths)")
print(f"  - Red: Tough paths ({len(all_paths['tough'])} paths, potential enemy routes)")
print("  - Total: {0} paths".format(len(all_paths['easy']) + len(all_paths['balanced']) + len(all_paths['tough'])))
print("----------------------------")