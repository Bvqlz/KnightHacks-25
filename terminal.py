import geopandas as gpd
import dill as pickle
import numpy as np
import pandas as pd
from shapely import wkt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from pathlib import Path
import shutil
import math
import matplotlib.pyplot as plt

# --- Load polygon helper ---
def load_polygon(polygon_path="polygon_lon_lat.wkt"):
    """Load polygon boundary from WKT file, returns None if not found."""
    try:
        with open(polygon_path) as f:
            polygon = wkt.loads(f.read())
        return polygon
    except FileNotFoundError:
        print(f"⚠ Polygon file '{polygon_path}' not found")
        return None
    except Exception as e:
        print(f"⚠ Failed to load polygon: {e}")
        return None

# --- Load data ---
print("=== Loading Data ===")
distance_matrix = np.load('distance_matrix.npy')
asset_indexes = np.load('asset_indexes.npy')
photo_indexes = np.load('photo_indexes.npy')
points_lat_long = np.load('points_lat_long.npy')
predecessors = np.load('predecessors.npy')
waypoint_indexes = np.load('waypoint_indexes.npy')
max_distance_per_trip = 37725

print(f"distance_matrix: {distance_matrix.shape}")
print(f"asset_indexes: {asset_indexes}")
print(f"photo_indexes: {photo_indexes}")
print(f"points_lat_long: {points_lat_long.shape}")
print(f"predecessors: {predecessors.shape}")
print(f"waypoint_indexes: {waypoint_indexes}")

# Load polygon once at startup
POLYGON = load_polygon()

# --- Understand the data structure ---
num_navigable = distance_matrix.shape[0]
num_total_points = points_lat_long.shape[0]
num_assets = asset_indexes[1] - asset_indexes[0] + 1
num_photos = photo_indexes[1] - photo_indexes[0] + 1

print(f"\n=== Data Structure ===")
print(f"Total points in coordinate list: {num_total_points}")
print(f"Navigable waypoints (in distance matrix): {num_navigable}")
print(f"Photo waypoints: {num_photos} (indices {photo_indexes[0]} to {photo_indexes[1]})")
print(f"Asset points: {num_assets} (indices {asset_indexes[0]} to {asset_indexes[1]})")
print(f"Note: Assets are NOT in distance matrix - they're reference points only")

def build_routing_model(data, num_vehicles):
    """Builds and returns an OR-Tools RoutingModel for a given number of vehicles."""
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]),
        num_vehicles,
        data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data["distance_matrix"][from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,
        data["max_distance"],
        True,
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    return manager, routing, transit_callback_index

def estimate_min_vehicles(distance_matrix, max_distance, depot=0):
    """Quick lower-bound estimate of minimum vehicles required."""
    N = distance_matrix.shape[0]
    dm = distance_matrix.copy().astype(np.float64)
    big = dm.max() * 10 + 1.0
    np.fill_diagonal(dm, big)
    min_out = np.min(dm, axis=1)
    total_min_travel = np.sum(min_out)
    approx_total_needed = total_min_travel * 1.5
    est = max(1, math.ceil(approx_total_needed / float(max_distance)))
    return est

def reconstruct_path(start, end, predecessors):
    """Reconstruct actual waypoint sequence from start to end using predecessor matrix.
    
    Args:
        start: Starting waypoint index
        end: Ending waypoint index
        predecessors: Predecessor matrix from Dijkstra
        
    Returns:
        List of waypoint indices forming the complete path
    """
    if start == end:
        return [start]
    
    # Check if path exists (convention: -9999 means no path/predecessor)
    if predecessors[start][end] == -9999 or predecessors[start][end] < 0:
        return [start, end]  # Direct connection or fallback
    
    # Reconstruct path backwards from end to start
    path = []
    current = end
    while current != start:
        path.append(current)
        prev = predecessors[start][current]
        if prev < 0 or prev == current:  # Safety check for cycles
            break
        current = int(prev)
    path.append(start)
    
    return list(reversed(path))

def expand_route_with_paths(route, predecessors, index_map):
    """Expand a high-level route into detailed waypoint sequence.
    
    Args:
        route: List of LOCAL waypoint indices [depot, wp1, wp2, ..., depot]
        predecessors: Predecessor matrix (uses GLOBAL indices)
        index_map: Dictionary mapping local indices to global indices
        
    Returns:
        List of all GLOBAL waypoint indices including intermediate nodes
    """
    full_path = []
    
    for i in range(len(route) - 1):
        # Convert local indices to global for predecessor lookup
        start_global = index_map[route[i]]
        end_global = index_map[route[i+1]]
        
        # Get the full path segment in global indices
        segment = reconstruct_path(start_global, end_global, predecessors)
        
        # Add segment, avoiding duplicate waypoints at junctions
        if i == 0:
            full_path.extend(segment)
        else:
            full_path.extend(segment[1:])  # Skip first (already added)
    
    return full_path

def plot_polygon_on_ax(ax, polygon=None):
    """Helper to plot polygon boundary on an existing axis."""
    if polygon is not None:
        gdf_poly = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        gdf_poly.boundary.plot(ax=ax, color='blue', linewidth=2, label='Boundary', zorder=1)

# --- Visualize the problem ---
try:
    navigable_coords = points_lat_long[:num_navigable]
    asset_coords = points_lat_long[asset_indexes[0]:asset_indexes[1]+1]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot polygon boundary
    plot_polygon_on_ax(ax, POLYGON)

    # Plot navigable waypoints and assets
    ax.scatter(navigable_coords[:, 0], navigable_coords[:, 1],
               c='red', s=10, alpha=0.5, label=f'Navigable waypoints ({num_navigable})')
    ax.scatter(asset_coords[:, 0], asset_coords[:, 1],
               c='green', s=50, marker='s', alpha=0.7, label=f'Assets ({num_assets})')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Drone Inspection Area')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Visualization error: {e}")

# --- Caching utilities ---
def cache_save(key, result, cache_dir="cache"):
    """Save result to cache with atomic write to prevent corruption."""
    Path(cache_dir).mkdir(exist_ok=True)
    cache_file = Path(cache_dir) / f"{key}.pkl"
    temp_file = Path(cache_dir) / f"{key}.pkl.tmp"

    try:
        with open(temp_file, "wb") as f:
            pickle.dump(result, f)
        shutil.move(str(temp_file), str(cache_file))
        print(f"✓ Solution cached to: {cache_file}")
    except Exception as e:
        print(f"✗ Cache save failed: {e}")
        if temp_file.exists():
            temp_file.unlink()

def cache_load(key, cache_dir="cache"):
    """Load cached result with error handling."""
    cache_file = Path(cache_dir) / f"{key}.pkl"
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "rb") as f:
            result = pickle.load(f)
        print(f"✓ Loaded cached solution from: {cache_file}")
        return result
    except (EOFError, Exception) as e:
        print(f"✗ Corrupted cache detected ({e}). Deleting and re-solving...")
        try:
            cache_file.unlink()
        except:
            pass
        return None

def clear_cache(cache_dir="cache"):
    """Delete all cached solutions."""
    cache_path = Path(cache_dir)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"✓ Cache cleared: {cache_dir}")

def create_data_model(subset_size=None):
    """Stores data for the problem restricted to depot + photo waypoints.
    
    Args:
        subset_size: If provided, use only first subset_size photo waypoints (for testing)
    """
    data = {}

    # Generate all photo node indices from the range
    # Check if photo_indexes[1] is within valid bounds
    max_valid_index = distance_matrix.shape[0] - 1
    end_index = min(photo_indexes[1], max_valid_index)
    
    # If photo_indexes[1] equals the array size, it's likely exclusive end
    if photo_indexes[1] == distance_matrix.shape[0]:
        photo_nodes = np.arange(photo_indexes[0], photo_indexes[1])
    else:
        # Otherwise it's inclusive end
        photo_nodes = np.arange(photo_indexes[0], end_index + 1)
    
    # Apply subset if requested
    if subset_size is not None and subset_size < len(photo_nodes):
        photo_nodes = photo_nodes[:subset_size]
        print(f"Using subset of {subset_size} photo waypoints for testing")

    # Include depot (index 0) + photo waypoints
    route_nodes = np.concatenate(([0], photo_nodes))

    # Restrict distance matrix to these nodes
    sub_distance_matrix = distance_matrix[np.ix_(route_nodes, route_nodes)].astype(np.int32)
    data["distance_matrix"] = sub_distance_matrix
    data["num_waypoints"] = len(route_nodes)

    # Depot is still first node
    data["depot"] = 0
    data["max_distance"] = int(max_distance_per_trip)

    # Save points coordinates for plotting (subset coordinates)
    data["points_lat_long"] = points_lat_long[route_nodes]

    # Map local indices back to original indices
    data["index_map"] = {local: global_ for local, global_ in enumerate(route_nodes)}

    return data

def export_mission_data(all_routes, data, output_dir="output"):
    """Export complete mission data in JSON format.
    
    Returns 4 JSON files:
    1. photo_points.json - All photo waypoint locations
    2. asset_points.json - All asset locations
    3. polygon_boundary.json - Flight boundary as coordinate array
    4. mission_paths.json - All drone routes with full waypoint sequences
    
    Args:
        all_routes: List of (route, distance) tuples from solver
        data: Data model containing index_map
        output_dir: Directory to save JSON files
    """
    import json
    from pathlib import Path
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # ============================================
    # 1. PHOTO POINTS JSON
    # ============================================
    photo_points = []
    photo_start = photo_indexes[0]
    photo_end = photo_indexes[1]
    
    # Handle exclusive vs inclusive end index
    if photo_end == distance_matrix.shape[0]:
        photo_range = range(photo_start, photo_end)
    else:
        photo_range = range(photo_start, photo_end + 1)
    
    for idx in photo_range:
        if idx < len(points_lat_long):
            lon, lat = points_lat_long[idx]
            photo_points.append({
                "waypoint_index": int(idx),
                "longitude": float(lon),
                "latitude": float(lat),
                "type": "photo"
            })
    
    with open(f"{output_dir}/photo_points.json", 'w') as f:
        json.dump({
            "type": "photo_waypoints",
            "count": len(photo_points),
            "points": photo_points
        }, f, indent=2)
    
    print(f"✓ Photo points exported: {len(photo_points)} points")
    
    # ============================================
    # 2. ASSET POINTS JSON
    # ============================================
    asset_points = []
    asset_start = asset_indexes[0]
    asset_end = asset_indexes[1]
    
    for idx in range(asset_start, min(asset_end + 1, len(points_lat_long))):
        lon, lat = points_lat_long[idx]
        asset_points.append({
            "asset_index": int(idx),
            "longitude": float(lon),
            "latitude": float(lat),
            "type": "asset"
        })
    
    with open(f"{output_dir}/asset_points.json", 'w') as f:
        json.dump({
            "type": "electrical_assets",
            "count": len(asset_points),
            "points": asset_points
        }, f, indent=2)
    
    print(f"✓ Asset points exported: {len(asset_points)} points")
    
    # ============================================
    # 3. POLYGON BOUNDARY JSON
    # ============================================
    polygon_coords = []
    
    if POLYGON is not None:
        # Extract coordinates from the polygon
        # Handle both Polygon and MultiPolygon
        from shapely.geometry import Polygon, MultiPolygon
        
        if isinstance(POLYGON, Polygon):
            # Single polygon - get exterior coordinates
            coords = list(POLYGON.exterior.coords)
            polygon_coords = [[float(lon), float(lat)] for lon, lat in coords]
        elif isinstance(POLYGON, MultiPolygon):
            # Multiple polygons - include all parts
            for poly in POLYGON.geoms:
                coords = list(poly.exterior.coords)
                polygon_coords.append([[float(lon), float(lat)] for lon, lat in coords])
    
    with open(f"{output_dir}/polygon_boundary.json", 'w') as f:
        json.dump({
            "type": "flight_boundary",
            "coordinate_system": "EPSG:4326",
            "coordinates": polygon_coords,
            "note": "Coordinates in [longitude, latitude] format"
        }, f, indent=2)
    
    print(f"✓ Polygon boundary exported")
    
    # ============================================
    # 4. MISSION PATHS JSON
    # ============================================
    missions = []
    
    for vehicle_id, (route, total_dist) in enumerate(all_routes):
        # Expand route to include all intermediate waypoints
        expanded = expand_route_with_paths(route, predecessors, data["index_map"])
        coords = points_lat_long[expanded]
        
        # Build waypoint sequence
        waypoints = []
        cumulative_dist = 0
        
        for seq in range(len(expanded)):
            wp_idx = expanded[seq]
            lon, lat = coords[seq]
            
            # Calculate segment distance
            if seq < len(expanded) - 1:
                wp_current = expanded[seq]
                wp_next = expanded[seq + 1]
                
                # Check if both waypoints are in the navigable range
                if wp_current < distance_matrix.shape[0] and wp_next < distance_matrix.shape[0]:
                    segment_dist = distance_matrix[wp_current][wp_next]
                else:
                    # For non-navigable waypoints (like assets), estimate from coordinates
                    coord_current = points_lat_long[wp_current]
                    coord_next = points_lat_long[wp_next]
                    # Simple Euclidean distance (you could use haversine for better accuracy)
                    dx = (coord_next[0] - coord_current[0]) * 364000  # rough lon to feet
                    dy = (coord_next[1] - coord_current[1]) * 364000  # rough lat to feet
                    segment_dist = int(np.sqrt(dx**2 + dy**2))
            else:
                segment_dist = 0

            cumulative_dist += segment_dist            
            # Determine waypoint type
            if wp_idx == 0:
                wp_type = "depot"
            elif photo_indexes[0] <= wp_idx <= photo_indexes[1]:
                wp_type = "photo"
            elif asset_indexes[0] <= wp_idx <= asset_indexes[1]:
                wp_type = "asset"
            else:
                wp_type = "intermediate"
            
            waypoints.append({
                "sequence": seq,
                "waypoint_index": int(wp_idx),
                "longitude": float(lon),
                "latitude": float(lat),
                "waypoint_type": wp_type,
                "segment_distance_ft": float(segment_dist),
                "cumulative_distance_ft": float(cumulative_dist)
            })
        
        # Calculate statistics
        photo_count = sum(1 for wp in waypoints if wp["waypoint_type"] == "photo")
        
        missions.append({
            "mission_id": vehicle_id + 1,
            "total_distance_ft": float(total_dist),
            "total_distance_km": float(total_dist) / 3280.84,
            "total_distance_miles": float(total_dist) / 5280.0,
            "num_waypoints": len(waypoints),
            "num_photo_points": photo_count,
            "depot_index": 0,
            "waypoints": waypoints
        })
    
    # Summary statistics
    summary = {
        "total_missions": len(missions),
        "total_distance_ft": sum(m["total_distance_ft"] for m in missions),
        "total_distance_km": sum(m["total_distance_km"] for m in missions),
        "total_distance_miles": sum(m["total_distance_miles"] for m in missions),  
        "total_waypoints": sum(m["num_waypoints"] for m in missions),
        "total_photo_points_visited": sum(m["num_photo_points"] for m in missions),
        "max_distance_per_mission_ft": max_distance_per_trip,
        "depot_coordinates": {
            "longitude": float(points_lat_long[0][0]),
            "latitude": float(points_lat_long[0][1])
        }
    }
    
    with open(f"{output_dir}/mission_paths.json", 'w') as f:
        json.dump({
            "type": "drone_mission_paths",
            "summary": summary,
            "missions": missions
        }, f, indent=2)
    
    print(f"✓ Mission paths exported: {len(missions)} missions")
    
    # ============================================
    # SUMMARY
    # ============================================
    print(f"\n{'='*50}")
    print(f"EXPORT COMPLETE - All files saved to '{output_dir}/'")
    print(f"{'='*50}")
    print(f"  1. photo_points.json      - {len(photo_points)} photo waypoints")
    print(f"  2. asset_points.json      - {len(asset_points)} electrical assets")
    print(f"  3. polygon_boundary.json  - Flight boundary coordinates")
    print(f"  4. mission_paths.json     - {len(missions)} drone missions")
    print(f"{'='*50}")
    print(f"Total distance: {summary['total_distance_km']:.2f} km ({summary['total_distance_miles']:.2f} miles)")
    print(f"Total waypoints: {summary['total_waypoints']}")
    print(f"{'='*50}\n")
    
    return {
        "photo_points": photo_points,
        "asset_points": asset_points,
        "polygon_coords": polygon_coords,
        "missions": missions,
        "summary": summary
    }

def main(use_subset=False, subset_size=500, max_vehicles=60, per_attempt_time_s=30):
    """Iteratively increases the number of vehicles until a feasible solution is found."""
    
    # Create data model with optional subset
    if use_subset:
        data = create_data_model(subset_size=subset_size)
        cache_key = f"solution_subset_{subset_size}"
    else:
        data = create_data_model(subset_size=None)
        cache_key = "solution_full"
    
    # Try to load cached solution
    cached_result = cache_load(cache_key)
    if cached_result is not None:
        print("Using cached solution!")
        all_routes = cached_result
        
        # Visualize cached solution
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            plot_polygon_on_ax(ax, POLYGON)

            # Background waypoints and assets
            navigable_coords = points_lat_long[:num_navigable]
            asset_coords = points_lat_long[asset_indexes[0]:asset_indexes[1]+1]
            ax.scatter(navigable_coords[:, 0], navigable_coords[:, 1],
                      c='lightgray', s=5, alpha=0.3, label='All waypoints', zorder=1)
            ax.scatter(asset_coords[:, 0], asset_coords[:, 1],
                      c='green', s=60, marker='s', alpha=0.7, label='Assets', zorder=2)

            # Plot each vehicle route with different color
            colors = plt.cm.get_cmap('tab10', len(all_routes))
            for i, (route, dist) in enumerate(all_routes):
                # Expand route with intermediate waypoints using predecessors
                expanded_route = expand_route_with_paths(route, predecessors, data["index_map"])
                
                # Get coordinates using global indices
                coords = points_lat_long[expanded_route]
                
                ax.plot(coords[:, 0], coords[:, 1], '-', linewidth=1.5,
                        color=colors(i), alpha=0.8, label=f'Route {i+1} ({dist/1000:.1f} km)')
                ax.scatter(coords[0, 0], coords[0, 1],
                          color=colors(i), marker='*', s=100, zorder=4)
                ax.scatter(coords[-1, 0], coords[-1, 1],
                          color=colors(i), marker='X', s=80, zorder=4)

            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('All Drone Missions Overview (Cached) - With Path Decoding')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
        export_data = export_mission_data(all_routes, data, output_dir="output")

        return all_routes
        
    est_min = estimate_min_vehicles(data["distance_matrix"], data["max_distance"])
    print(f"Estimated minimum vehicles required: {est_min}")

    for num_vehicles in range(est_min, max_vehicles + 1):
        print(f"\nAttempting solve with {num_vehicles} vehicle(s)...")
        data["num_vehicles"] = num_vehicles

        manager, routing, _ = build_routing_model(data, num_vehicles)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = per_attempt_time_s
        search_parameters.log_search = False

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            print(f" Feasible solution found with {num_vehicles} vehicles.")
            all_routes = []

            for vehicle_id in range(num_vehicles):
                index = routing.Start(vehicle_id)
                route = []
                route_distance = 0

                while not routing.IsEnd(index):
                    node = manager.IndexToNode(index)
                    route.append(node)
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

                route.append(manager.IndexToNode(index))
                all_routes.append((route, route_distance))
                print(f"  Route #{vehicle_id + 1}: {len(route)} waypoints, {route_distance:.1f} m")

            # Cache the solution
            cache_save(cache_key, all_routes)

            # --- Visualization for all routes ---
            try:
                fig, ax = plt.subplots(figsize=(12, 10))
                plot_polygon_on_ax(ax, POLYGON)

                # Background waypoints and assets
                navigable_coords = points_lat_long[:num_navigable]
                asset_coords = points_lat_long[asset_indexes[0]:asset_indexes[1]+1]
                ax.scatter(navigable_coords[:, 0], navigable_coords[:, 1],
                          c='lightgray', s=5, alpha=0.3, label='All waypoints', zorder=1)
                ax.scatter(asset_coords[:, 0], asset_coords[:, 1],
                          c='green', s=60, marker='s', alpha=0.7, label='Assets', zorder=2)

                # Plot each vehicle route with different color
                colors = plt.cm.get_cmap('tab10', len(all_routes))
                for i, (route, dist) in enumerate(all_routes):
                    # Expand route with intermediate waypoints using predecessors
                    expanded_route = expand_route_with_paths(route, predecessors, data["index_map"])
                    
                    # Get coordinates using global indices
                    coords = points_lat_long[expanded_route]
                    
                    ax.plot(coords[:, 0], coords[:, 1], '-', linewidth=1.5,
                            color=colors(i), alpha=0.8, label=f'Route {i+1} ({dist/1000:.1f} km)')
                    ax.scatter(coords[0, 0], coords[0, 1],
                              color=colors(i), marker='*', s=100, zorder=4)
                    ax.scatter(coords[-1, 0], coords[-1, 1],
                              color=colors(i), marker='X', s=80, zorder=4)

                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title('All Drone Missions Overview - With Path Decoding')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Visualization failed: {e}")
                import traceback
                traceback.print_exc()
            export_data = export_mission_data(all_routes, data, output_dir="output")
            return all_routes
        else:
            print(f"\n No solution found with {num_vehicles} vehicles. Retrying...")

    print(f"\n No feasible solution found up to {max_vehicles} vehicles.")
    return None

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Test with 500 waypoints (fast, ~1 min)")
    print("2. Full optimization with all waypoints (slow, ~5-10 min)")
    print("3. Clear cache and re-solve")

    TEST_MODE = False
    CLEAR_CACHE = False

    if CLEAR_CACHE:
        clear_cache()

    if TEST_MODE:
        print("\n Running in TEST MODE (500 waypoints)\n")
        route = main(use_subset=True, subset_size=500)
    else:
        print("\n Running FULL OPTIMIZATION (all waypoints)\n")
        route = main(use_subset=False)