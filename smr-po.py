#!/usr/bin/env python3
"""
SMR Path Optimization Tool - Single Page Application
=====================================================
- Draw multiple zones on the same map
- Name each zone
- Select start point from dropdown
- Calculate optimized route and display on map
- Save all zones and routes to a single JSON file
"""

import json
import re
import os
from math import radians, sin, cos, sqrt, atan2
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import urllib.parse

# Configuration
DATA_FILE = 'product_sense_public_shops_with_area.json'
WORKING_DIR = '/home/sajadulakash/Desktop/SMR PO'
OUTPUT_FILE = 'zones_routes.json'
PORT = 9541

# ============================================================================
# Data Loading
# ============================================================================

def parse_coordinate(coord_str):
    """Parse coordinate like '23.8692469° N' to float"""
    if coord_str is None:
        return None
    cleaned = re.sub(r'[°NSEW\s]', '', str(coord_str))
    try:
        return float(cleaned)
    except ValueError:
        return None

def load_stops(filepath):
    """Load stops from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stops = []
    for item in data:
        lat = parse_coordinate(item.get('lat'))
        lon = parse_coordinate(item.get('long'))
        if lat and lon:
            stops.append({
                'id': item.get('id', ''),
                'name': item.get('name', 'Unknown'),
                'address': item.get('address', ''),
                'area': item.get('area', ''),
                'lat': lat,
                'lon': lon
            })
    return stops

# ============================================================================
# Route Optimization
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# ============================================================================
# OSRM API for Road Routing (Fast Online Service)
# ============================================================================

import urllib.request

OSRM_SERVER = "https://router.project-osrm.org"
OSRM_TIMEOUT = 60  # seconds

def point_in_polygon(lat, lon, polygon):
    """Check if a point is inside a polygon using ray casting algorithm.
    polygon is a list of [lat, lon] coordinates.
    """
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        if ((polygon[i][0] > lat) != (polygon[j][0] > lat) and
            lon < (polygon[j][1] - polygon[i][1]) * (lat - polygon[i][0]) / 
                  (polygon[j][0] - polygon[i][0]) + polygon[i][1]):
            inside = not inside
        j = i
    
    return inside

def get_osrm_route_segment(from_stop, to_stop):
    """Get road route for a single segment between two stops."""
    try:
        coords_str = f"{from_stop['lon']},{from_stop['lat']};{to_stop['lon']},{to_stop['lat']}"
        url = f"{OSRM_SERVER}/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'RouteOptimizer/1.0'})
        with urllib.request.urlopen(req, timeout=OSRM_TIMEOUT) as response:
            data = json.loads(response.read().decode())
        
        if data.get('code') == 'Ok':
            route = data['routes'][0]
            distance_m = route['distance']
            geometry = route['geometry']['coordinates']
            road_coords = [[coord[1], coord[0]] for coord in geometry]
            return distance_m, road_coords
    except Exception as e:
        pass
    
    return None, None

def clip_route_to_polygon(route_coords, polygon):
    """Clip route coordinates to stay inside the polygon.
    For segments that go outside, use straight line (which stays more inside).
    """
    if not route_coords or not polygon:
        return route_coords
    
    clipped = []
    last_inside_point = None
    
    for coord in route_coords:
        is_inside = point_in_polygon(coord[0], coord[1], polygon)
        
        if is_inside:
            # If we were outside and now inside, add entry point
            if last_inside_point is not None and len(clipped) > 0:
                # We're back inside
                pass
            clipped.append(coord)
            last_inside_point = coord
        else:
            # Point is outside - we'll skip it but remember we went outside
            # The route will have gaps that will be filled with straight lines
            pass
    
    return clipped if clipped else route_coords

def get_route_inside_zone(optimized_stops, polygon):
    """Get route geometry that stays inside the zone.
    
    Strategy:
    1. For each segment, get OSRM route
    2. Check if route goes outside polygon
    3. If outside, use straight line between stops (stays inside zone)
    4. Calculate total distance
    """
    if len(optimized_stops) < 2:
        return 0, None
    
    all_coords = []
    total_distance = 0
    
    for i in range(len(optimized_stops) - 1):
        from_stop = optimized_stops[i]
        to_stop = optimized_stops[i + 1]
        
        # Get OSRM route for this segment
        seg_dist, seg_coords = get_osrm_route_segment(from_stop, to_stop)
        
        if seg_coords:
            # Check how much of the route is outside the polygon
            outside_count = 0
            for coord in seg_coords:
                if not point_in_polygon(coord[0], coord[1], polygon):
                    outside_count += 1
            
            outside_ratio = outside_count / len(seg_coords) if seg_coords else 0
            
            if outside_ratio < 0.3:  # Less than 30% outside - use OSRM route
                all_coords.extend(seg_coords)
                total_distance += seg_dist if seg_dist else 0
            else:
                # Too much outside - use straight line (stays inside zone)
                # For straight line, calculate haversine distance
                straight_dist = haversine_distance(
                    from_stop['lat'], from_stop['lon'],
                    to_stop['lat'], to_stop['lon']
                ) * 1000  # Convert to meters
                
                # Create straight line path
                all_coords.append([from_stop['lat'], from_stop['lon']])
                all_coords.append([to_stop['lat'], to_stop['lon']])
                
                # Use estimated road distance (straight * 1.4 factor)
                total_distance += straight_dist * 1.4
        else:
            # No OSRM route, use straight line
            straight_dist = haversine_distance(
                from_stop['lat'], from_stop['lon'],
                to_stop['lat'], to_stop['lon']
            ) * 1000
            
            all_coords.append([from_stop['lat'], from_stop['lon']])
            all_coords.append([to_stop['lat'], to_stop['lon']])
            total_distance += straight_dist * 1.4
    
    return total_distance / 1000, all_coords  # Return km

def get_osrm_route(stops):
    """Get road route geometry from OSRM for a sequence of stops.
    Returns: (total_distance_km, road_geometry) or (None, None) on failure
    """
    if len(stops) < 2:
        return None, None
    
    try:
        # OSRM expects lon,lat format
        coords_str = ";".join([f"{s['lon']},{s['lat']}" for s in stops])
        url = f"{OSRM_SERVER}/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'RouteOptimizer/1.0'})
        with urllib.request.urlopen(req, timeout=OSRM_TIMEOUT) as response:
            data = json.loads(response.read().decode())
        
        if data.get('code') == 'Ok':
            route = data['routes'][0]
            distance_km = route['distance'] / 1000
            geometry = route['geometry']['coordinates']  # [[lon, lat], ...]
            # Convert to [[lat, lon], ...] for Leaflet
            road_coords = [[coord[1], coord[0]] for coord in geometry]
            return distance_km, road_coords
    except Exception as e:
        print(f"   ⚠️ OSRM route error: {e}")
    
    return None, None

def get_osrm_distance_matrix(stops, batch_size=50):
    """Get real road distance matrix from OSRM Table API.
    OSRM automatically snaps points to nearest road.
    Returns matrix in meters (integers) for OR-Tools.
    """
    n = len(stops)
    matrix = [[0] * n for _ in range(n)]
    
    if n <= 1:
        return matrix
    
    try:
        # For small number of stops, get full matrix in one call
        if n <= batch_size:
            coords_str = ";".join([f"{s['lon']},{s['lat']}" for s in stops])
            url = f"{OSRM_SERVER}/table/v1/driving/{coords_str}?annotations=distance"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'RouteOptimizer/1.0'})
            with urllib.request.urlopen(req, timeout=OSRM_TIMEOUT) as response:
                data = json.loads(response.read().decode())
            
            if data.get('code') == 'Ok':
                distances = data['distances']  # Already in meters
                for i in range(n):
                    for j in range(n):
                        if distances[i][j] is not None:
                            matrix[i][j] = int(distances[i][j])
                        else:
                            # No road connection, use large penalty
                            matrix[i][j] = 999999999
                return matrix
        else:
            # For larger sets, batch the requests
            # First get all coordinates
            coords_str = ";".join([f"{s['lon']},{s['lat']}" for s in stops])
            
            # Get distances in batches (sources in batches, all destinations)
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                sources = ";".join([str(i) for i in range(batch_start, batch_end)])
                
                url = f"{OSRM_SERVER}/table/v1/driving/{coords_str}?annotations=distance&sources={sources}"
                
                req = urllib.request.Request(url, headers={'User-Agent': 'RouteOptimizer/1.0'})
                with urllib.request.urlopen(req, timeout=OSRM_TIMEOUT) as response:
                    data = json.loads(response.read().decode())
                
                if data.get('code') == 'Ok':
                    distances = data['distances']
                    for i, src_idx in enumerate(range(batch_start, batch_end)):
                        for j in range(n):
                            if distances[i][j] is not None:
                                matrix[src_idx][j] = int(distances[i][j])
                            else:
                                matrix[src_idx][j] = 999999999
            
            return matrix
            
    except Exception as e:
        print(f"   ⚠️ OSRM table error: {e}, falling back to haversine")
    
    # Fallback to haversine if OSRM fails
    return build_haversine_matrix(stops)

def build_haversine_matrix(stops):
    """Build distance matrix using haversine (straight-line) distances.
    Fallback when OSRM is unavailable.
    """
    n = len(stops)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_km = haversine_distance(
                    stops[i]['lat'], stops[i]['lon'],
                    stops[j]['lat'], stops[j]['lon']
                )
                matrix[i][j] = int(dist_km * 1000)  # meters as integer
    return matrix

def solve_tsp_ortools(stops, start_idx=0, matrix=None):
    """Solve TSP using Google OR-Tools - same algorithm as Google Maps.
    Uses Guided Local Search metaheuristic for near-optimal solutions.
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    
    n = len(stops)
    if n <= 1:
        return list(range(n)), 0
    
    if matrix is None:
        matrix = build_distance_matrix(stops)
    
    # Create routing index manager
    manager = pywrapcp.RoutingIndexManager(n, 1, start_idx)
    
    # Create routing model
    routing = pywrapcp.RoutingModel(manager)
    
    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Search parameters - use Guided Local Search (Google's preferred method)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 5  # Max 5 seconds for optimization
    search_parameters.log_search = False
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        # Extract route
        route = []
        index = routing.Start(0)
        total_distance = 0
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            total_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        
        return route, total_distance / 1000  # Convert back to km
    else:
        # Fallback to simple nearest neighbor if OR-Tools fails
        print("   ⚠️ OR-Tools failed, using fallback algorithm")
        return fallback_nearest_neighbor(stops, start_idx, matrix)

def fallback_nearest_neighbor(stops, start_idx=0, matrix=None):
    """Simple nearest neighbor as fallback"""
    n = len(stops)
    if n == 0:
        return [], 0
    
    if matrix is None:
        matrix = build_distance_matrix(stops)
    
    visited = [False] * n
    route = [start_idx]
    visited[start_idx] = True
    total_dist = 0
    current = start_idx
    
    for _ in range(n - 1):
        nearest = None
        nearest_dist = float('inf')
        for j in range(n):
            if not visited[j] and matrix[current][j] < nearest_dist:
                nearest = j
                nearest_dist = matrix[current][j]
        if nearest is not None:
            route.append(nearest)
            visited[nearest] = True
            total_dist += nearest_dist
            current = nearest
    
    return route, total_dist / 1000  # Convert to km

def optimize_route(stops, start_idx=0):
    """Optimize route through stops using real road distances from OSRM.
    
    Features:
    - Uses actual road distances (not straight-line) for optimization
    - OSRM automatically snaps stops to nearest road
    - OR-Tools finds minimum total distance route
    - Returns optimized order with road geometry for display
    """
    if len(stops) <= 1:
        return stops, 0, None
    
    print(f"\nOptimizing route for {len(stops)} stops using REAL ROAD distances...")
    
    # Get real road distance matrix from OSRM
    # This automatically handles:
    # - Snapping points to nearest road
    # - Computing actual driving distances
    # - Considering one-way streets, road networks, etc.
    print(f"   Getting road distance matrix from OSRM...")
    matrix = get_osrm_distance_matrix(stops)
    
    # Check if we got valid road distances
    sample_dist = matrix[0][1] if len(stops) > 1 else 0
    if sample_dist == 999999999:
        print(f"   OSRM unavailable, falling back to haversine distances")
        matrix = build_haversine_matrix(stops)
    else:
        print(f"   Road distance matrix ready ({len(stops)}x{len(stops)})")
    
    # Run Google OR-Tools optimization with road distances
    print(f"   Running OR-Tools solver (Guided Local Search)...")
    route_idx, total_dist = solve_tsp_ortools(stops, start_idx, matrix)
    
    optimized = [stops[i] for i in route_idx]
    
    # Get actual road geometry from OSRM for visualization
    road_geometry = None
    print(f"   Getting road path geometry for display...")
    road_dist, road_geometry = get_osrm_route(optimized)
    
    if road_geometry:
        print(f"   Road path: {len(road_geometry)} points, {round(road_dist, 2)} km")
        total_dist = road_dist  # Use the accurate road distance
    else:
        print(f"   Could not get road geometry for display")
    
    print(f"   Optimization complete: {round(total_dist, 2)} km total road distance")
    
    return optimized, total_dist, road_geometry

def optimize_route_in_zone(stops, start_idx=0, polygon=None):
    """Optimize route that stays INSIDE the zone polygon.
    
    Features:
    - Uses actual road distances for optimization
    - Route visualization stays inside zone boundaries
    - If road goes outside zone, uses straight line path instead
    - Backtracking is allowed to stay inside
    """
    if len(stops) <= 1:
        return stops, 0, None
    
    print(f"\nOptimizing route for {len(stops)} stops (ZONE-CONSTRAINED)...")
    
    # Get road distance matrix from OSRM for optimization
    print(f"   Getting road distance matrix from OSRM...")
    matrix = get_osrm_distance_matrix(stops)
    
    # Check if we got valid road distances
    sample_dist = matrix[0][1] if len(stops) > 1 else 0
    if sample_dist == 999999999:
        print(f"   OSRM unavailable, falling back to haversine distances")
        matrix = build_haversine_matrix(stops)
    else:
        print(f"   Road distance matrix ready ({len(stops)}x{len(stops)})")
    
    # Run Google OR-Tools optimization
    print(f"   Running OR-Tools solver (Guided Local Search)...")
    route_idx, total_dist = solve_tsp_ortools(stops, start_idx, matrix)
    
    optimized = [stops[i] for i in route_idx]
    
    # Get route geometry that stays INSIDE the zone
    road_geometry = None
    if polygon:
        print(f"   Getting zone-constrained road path...")
        road_dist, road_geometry = get_route_inside_zone(optimized, polygon)
        
        if road_geometry:
            print(f"   Zone-constrained path: {len(road_geometry)} points, {round(road_dist, 2)} km")
            total_dist = road_dist
        else:
            print(f"   Could not get zone-constrained geometry, using OSRM...")
            road_dist, road_geometry = get_osrm_route(optimized)
            if road_geometry:
                total_dist = road_dist
    else:
        # No polygon provided, use regular OSRM route
        road_dist, road_geometry = get_osrm_route(optimized)
        if road_geometry:
            total_dist = road_dist
    
    print(f"   Optimization complete: {round(total_dist, 2)} km total distance")
    
    return optimized, total_dist, road_geometry

# HTTP Server with API
# ============================================================================

class RequestHandler(BaseHTTPRequestHandler):
    stops = []
    zones_data = {'zones': []}
    
    def log_message(self, format, *args):
        pass  # Suppress logging
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html = generate_main_page(RequestHandler.stops)
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/api/zones':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(RequestHandler.zones_data).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/api/optimize':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            # Get stops and start index
            stops = data['stops']
            start_idx = data.get('start_idx', 0)
            zone_name = data.get('zone_name', 'Unnamed Zone')
            polygon = data.get('polygon', [])
            
            # Optimize route with zone constraint (route stays inside polygon)
            optimized, distance, road_geometry = optimize_route_in_zone(stops, start_idx, polygon)
            
            # Create zone data
            zone_data = {
                'name': zone_name,
                'polygon': polygon,
                'total_stops': len(optimized),
                'total_distance_km': round(distance, 2),
                'route': optimized,
                'road_geometry': road_geometry  # Actual road path for visualization
            }
            
            # Add to zones and save
            RequestHandler.zones_data['zones'].append(zone_data)
            save_zones_to_file(RequestHandler.zones_data)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'success': True,
                'zone': zone_data,
                'zone_index': len(RequestHandler.zones_data['zones']) - 1
            }
            self.wfile.write(json.dumps(response).encode())
        
        elif self.path == '/api/delete-zone':
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            zone_index = post_data.get('zone_index', -1)
            
            if 0 <= zone_index < len(RequestHandler.zones_data['zones']):
                deleted_zone = RequestHandler.zones_data['zones'].pop(zone_index)
                save_zones_to_file(RequestHandler.zones_data)
                print(f"🗑️ Deleted zone: {deleted_zone['name']}")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': True}).encode())
            else:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'error': 'Invalid zone index'}).encode())
        
        elif self.path == '/api/clear':
            RequestHandler.zones_data = {'zones': []}
            save_zones_to_file(RequestHandler.zones_data)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True}).encode())
        
        else:
            self.send_response(404)
            self.end_headers()

def save_zones_to_file(zones_data):
    """Save zones data to JSON file"""
    filepath = os.path.join(WORKING_DIR, OUTPUT_FILE)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(zones_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved {len(zones_data['zones'])} zones to {OUTPUT_FILE}")

def load_zones_from_file():
    """Load existing zones from file"""
    filepath = os.path.join(WORKING_DIR, OUTPUT_FILE)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'zones': []}

# ============================================================================
# Generate Main Page HTML
# ============================================================================

def generate_main_page(stops):
    """Generate the single-page application HTML"""
    
    lats = [s['lat'] for s in stops]
    lons = [s['lon'] for s in stops]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    stops_js = json.dumps([{
        'id': s['id'],
        'name': s['name'],
        'address': s['address'],
        'lat': s['lat'],
        'lon': s['lon']
    } for s in stops])
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMR PO - Route Optimizer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #map {{ height: 100vh; width: calc(100% - 380px); float: left; }}
        #sidebar {{
            width: 380px;
            height: 100vh;
            float: right;
            background: #f5f5f5;
            overflow-y: auto;
            padding: 15px;
        }}
        h2 {{ margin: 0 0 15px 0; color: #333; font-size: 18px; }}
        h3 {{ margin: 15px 0 10px 0; color: #555; font-size: 14px; }}
        .panel {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .step {{
            padding: 10px;
            margin: 8px 0;
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            font-size: 13px;
        }}
        .step.active {{ background: #fff3e0; border-left-color: #ff9800; }}
        .step.done {{ background: #e8f5e9; border-left-color: #4CAF50; }}
        input, select {{
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }}
        button {{
            width: 100%;
            padding: 12px;
            margin: 5px 0;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
        }}
        .btn-primary {{ background: #4CAF50; color: white; }}
        .btn-primary:hover {{ background: #45a049; }}
        .btn-primary:disabled {{ background: #ccc; cursor: not-allowed; }}
        .btn-danger {{ background: #f44336; color: white; }}
        .btn-danger:hover {{ background: #d32f2f; }}
        .btn-secondary {{ background: #607D8B; color: white; }}
        .zone-item {{
            padding: 10px;
            margin: 5px 0;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 12px;
        }}
        .zone-item .name {{ 
            font-weight: bold; 
            color: #333; 
            display: flex;
            align-items: center;
        }}
        .zone-item .name:hover {{ 
            color: #2196f3;
        }}
        .zone-item .stats {{ 
            color: #666; 
            margin-top: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .delete-btn {{
            background: none;
            border: none;
            cursor: pointer;
            font-size: 14px;
            padding: 2px 6px;
            border-radius: 3px;
            opacity: 0.6;
        }}
        .delete-btn:hover {{
            background: #ffebee;
            opacity: 1;
        }}
        .zone-color {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        #status {{
            padding: 10px;
            background: #e8f5e9;
            border-radius: 5px;
            font-size: 13px;
            margin-bottom: 10px;
        }}
        .hidden {{ display: none; }}
        #loading {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 30px 50px;
            border-radius: 10px;
            z-index: 9999;
            display: none;
        }}
    </style>
</head>
<body>
    <div id="loading">⏳ Optimizing route...</div>
    <div id="map"></div>
    <div id="sidebar">
        <h2>🗺️ SMR Route Optimizer</h2>
        
        <div class="panel">
            <div id="status">
                📍 Total stops: {len(stops)}<br>
                🎯 Zones created: <span id="zoneCount">0</span>
            </div>
            
            <div class="step active" id="step1">
                <strong>Step 1:</strong> Draw a polygon zone on map
            </div>
            <div class="step" id="step2">
                <strong>Step 2:</strong> Name the zone & select start point
            </div>
            <div class="step" id="step3">
                <strong>Step 3:</strong> Calculate & view route
            </div>
        </div>
        
        <div class="panel hidden" id="zoneSetup">
            <h3>📝 Zone Setup</h3>
            <label>Zone Name:</label>
            <input type="text" id="zoneName" placeholder="Enter zone name...">
            
            <label>Start Point:</label>
            <select id="startDropdown">
                <option value="">-- Draw zone first --</option>
            </select>
            
            <p style="font-size: 12px; color: #666;">
                🎯 Stops in zone: <span id="selectedCount">0</span>
            </p>
            
            <button class="btn-primary" id="calcBtn" disabled onclick="calculateRoute()">
                🚗 Calculate Optimized Route
            </button>
            
            <button class="btn-secondary" onclick="cancelZone()">
                ❌ Cancel
            </button>
        </div>
        
        <div class="panel">
            <h3>📋 Saved Zones</h3>
            <div id="zoneList">
                <p style="color: #999; font-size: 13px;">No zones yet. Draw a polygon to create one.</p>
            </div>
            <button class="btn-danger" onclick="clearAllZones()" style="margin-top: 10px;">
                🗑️ Clear All Zones
            </button>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
    <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
    
    <script>
        const allStops = {stops_js};
        const zoneColors = ['#e91e63', '#9c27b0', '#673ab7', '#3f51b5', '#2196f3', '#00bcd4', '#009688', '#4caf50', '#ff9800', '#ff5722'];
        
        let selectedStops = [];
        let currentPolygon = null;
        let zones = [];
        let routeLayers = [];
        let currentStep = 1;
        
        // Initialize map with higher max zoom
        const map = L.map('map', {{
            maxZoom: 22
        }}).setView([{center_lat}, {center_lon}], 12);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap',
            maxZoom: 22,
            maxNativeZoom: 19
        }}).addTo(map);
        
        // Marker cluster for all stops
        const markers = L.markerClusterGroup({{ disableClusteringAtZoom: 16 }});
        allStops.forEach(stop => {{
            const marker = L.marker([stop.lat, stop.lon])
                .bindPopup(`<b>${{stop.name}}</b><br>${{stop.address}}`);
            markers.addLayer(marker);
        }});
        map.addLayer(markers);
        
        // Drawing layer
        const drawnItems = new L.FeatureGroup();
        map.addLayer(drawnItems);
        
        // Drawing controls
        const drawControl = new L.Control.Draw({{
            draw: {{
                polygon: {{ shapeOptions: {{ color: '#ff9800' }} }},
                rectangle: {{ shapeOptions: {{ color: '#ff9800' }} }},
                circle: false,
                circlemarker: false,
                marker: false,
                polyline: false
            }},
            edit: false
        }});
        map.addControl(drawControl);
        
        // Handle polygon drawing
        map.on('draw:created', function(e) {{
            // Clear previous temp polygon
            drawnItems.clearLayers();
            currentPolygon = e.layer;
            drawnItems.addLayer(currentPolygon);
            
            // Filter stops inside polygon
            selectedStops = allStops.filter(stop => {{
                return isPointInPolygon(L.latLng(stop.lat, stop.lon), currentPolygon);
            }});
            
            if (selectedStops.length > 0) {{
                document.getElementById('selectedCount').textContent = selectedStops.length;
                populateDropdown();
                document.getElementById('zoneSetup').classList.remove('hidden');
                document.getElementById('zoneName').value = '';
                document.getElementById('zoneName').focus();
                setStep(2);
            }} else {{
                alert('⚠️ No stops found in this area. Try drawing a larger zone.');
                drawnItems.clearLayers();
                currentPolygon = null;
            }}
        }});
        
        function isPointInPolygon(point, polygon) {{
            const polyPoints = polygon.getLatLngs()[0];
            let inside = false;
            for (let i = 0, j = polyPoints.length - 1; i < polyPoints.length; j = i++) {{
                const xi = polyPoints[i].lat, yi = polyPoints[i].lng;
                const xj = polyPoints[j].lat, yj = polyPoints[j].lng;
                if (((yi > point.lng) !== (yj > point.lng)) &&
                    (point.lat < (xj - xi) * (point.lng - yi) / (yj - yi) + xi)) {{
                    inside = !inside;
                }}
            }}
            return inside;
        }}
        
        function populateDropdown() {{
            const dropdown = document.getElementById('startDropdown');
            dropdown.innerHTML = '<option value="">-- Select start point --</option>';
            selectedStops.forEach((stop, i) => {{
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = stop.name;
                dropdown.appendChild(opt);
            }});
            dropdown.onchange = function() {{
                document.getElementById('calcBtn').disabled = (this.value === '');
                if (this.value !== '') setStep(3);
            }};
        }}
        
        function setStep(step) {{
            currentStep = step;
            for (let i = 1; i <= 3; i++) {{
                const el = document.getElementById('step' + i);
                el.classList.remove('active', 'done');
                if (i < step) el.classList.add('done');
                if (i === step) el.classList.add('active');
            }}
        }}
        
        async function calculateRoute() {{
            const zoneName = document.getElementById('zoneName').value.trim() || 'Zone ' + (zones.length + 1);
            const startIdx = parseInt(document.getElementById('startDropdown').value);
            
            if (isNaN(startIdx)) {{
                alert('Please select a start point');
                return;
            }}
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            
            try {{
                const response = await fetch('/api/optimize', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        zone_name: zoneName,
                        stops: selectedStops,
                        start_idx: startIdx,
                        polygon: currentPolygon.getLatLngs()[0].map(p => [p.lat, p.lng])
                    }})
                }});
                
                const result = await response.json();
                
                if (result.success) {{
                    // Add zone to list
                    zones.push(result.zone);
                    
                    // Draw route on map
                    drawRoute(result.zone, result.zone_index);
                    
                    // Update UI
                    updateZoneList();
                    
                    // Clear temp polygon and hide setup
                    drawnItems.clearLayers();
                    currentPolygon = null;
                    selectedStops = [];
                    document.getElementById('zoneSetup').classList.add('hidden');
                    setStep(1);
                    
                    alert(`✅ Zone "${{zoneName}}" created!\\n📍 ${{result.zone.total_stops}} stops\\n📏 ${{result.zone.total_distance_km}} km`);
                }}
            }} catch (err) {{
                alert('Error: ' + err.message);
            }} finally {{
                document.getElementById('loading').style.display = 'none';
            }}
        }}
        
        // Global array to store arrow markers for cleanup
        let arrowMarkers = [];
        
        function addArrowsToRoute(coords, color) {{
            // Add arrow markers along the route to show direction
            // Use longer lookahead for consistent direction calculation
            const arrowInterval = 15; // Add arrow every N points
            const lookAhead = 10; // Points to look ahead for direction (smoother)
            
            for (let i = arrowInterval; i < coords.length - lookAhead; i += arrowInterval) {{
                // Use points further apart for more consistent direction
                const p1 = coords[i];
                const p2 = coords[Math.min(i + lookAhead, coords.length - 1)];
                
                // coords are [lat, lon] - lat is Y (vertical), lon is X (horizontal)
                const lat1 = p1[0], lon1 = p1[1];
                const lat2 = p2[0], lon2 = p2[1];
                
                // Skip if points are too close (would give unstable direction)
                const dist = Math.sqrt(Math.pow(lat2 - lat1, 2) + Math.pow(lon2 - lon1, 2));
                if (dist < 0.0001) continue;
                
                // Calculate bearing/angle from point 1 to point 2
                const deltaLat = lat2 - lat1;
                const deltaLon = lon2 - lon1;
                
                // atan2(deltaLon, deltaLat) gives angle from North (0°)
                const angleRad = Math.atan2(deltaLon, deltaLat);
                const angleDeg = angleRad * 180 / Math.PI;
                
                // Arrow symbol ▶ points right by default, rotate to point in travel direction
                const rotation = 90 - angleDeg;
                
                // Create arrow marker
                const arrowIcon = L.divIcon({{
                    className: 'route-arrow',
                    html: `<div style="
                        color: ${{color}};
                        font-size: 12px;
                        font-weight: bold;
                        transform: rotate(${{rotation}}deg);
                        text-shadow: 1px 1px 1px white, -1px -1px 1px white, 1px -1px 1px white, -1px 1px 1px white;
                        line-height: 1;
                    ">▶</div>`,
                    iconSize: [12, 12],
                    iconAnchor: [6, 6]
                }});
                
                const arrowMarker = L.marker([p1[0], p1[1]], {{ icon: arrowIcon }}).addTo(map);
                arrowMarkers.push(arrowMarker);
            }}
        }}
        
        function clearArrowMarkers() {{
            arrowMarkers.forEach(m => map.removeLayer(m));
            arrowMarkers = [];
        }}
        
        function drawRoute(zone, index) {{
            const color = zoneColors[index % zoneColors.length];
            const route = zone.route;
            const roadGeometry = zone.road_geometry;  // Actual road path from OSMnx
            const startColor = '#00C853';  // Green for start
            const endColor = '#FF1744';    // Red for end
            
            // Draw polygon
            const polygonLayer = L.polygon(zone.polygon, {{
                color: color,
                fillColor: color,
                fillOpacity: 0.1,
                weight: 2
            }}).addTo(map);
            
            // Calculate polygon center for zone label
            const bounds = polygonLayer.getBounds();
            const center = bounds.getCenter();
            
            // Add zone name label on map
            const labelIcon = L.divIcon({{
                className: 'zone-label',
                html: `<div style="
                    background: ${{color}};
                    color: white;
                    padding: 5px 10px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                    white-space: nowrap;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                    border: 2px solid white;
                ">${{zone.name}}</div>`,
                iconSize: [100, 30],
                iconAnchor: [50, 15]
            }});
            const labelMarker = L.marker([center.lat, center.lng], {{ icon: labelIcon }}).addTo(map);
            
            // Draw route line - use road geometry if available, otherwise straight lines
            let routeCoords;
            if (roadGeometry && roadGeometry.length > 0) {{
                routeCoords = roadGeometry;  // Use actual road path
                console.log('Drawing road path:', roadGeometry.length, 'points');
            }} else {{
                routeCoords = route.map(s => [s.lat, s.lon]);  // Fallback to straight lines
                console.log('Drawing straight lines (no road geometry)');
            }}
            
            const routeLine = L.polyline(routeCoords, {{
                color: color,
                weight: 3,
                opacity: 0.9
            }}).addTo(map);
            
            // Add numbered markers with special colors for start and end
            const markerGroup = L.layerGroup();
            const lastIdx = route.length - 1;
            
            route.forEach((stop, i) => {{
                let markerColor = color;
                let markerSize = 20;
                let fontSize = 10;
                let label = i + 1;
                
                // Start point - green, larger
                if (i === 0) {{
                    markerColor = startColor;
                    markerSize = 28;
                    fontSize = 12;
                    label = '▶';
                }}
                // End point - red, larger
                else if (i === lastIdx) {{
                    markerColor = endColor;
                    markerSize = 28;
                    fontSize = 12;
                    label = '◼';
                }}
                
                const icon = L.divIcon({{
                    className: 'route-marker',
                    html: `<div style="
                        background: ${{markerColor}};
                        color: white;
                        width: ${{markerSize}}px;
                        height: ${{markerSize}}px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: ${{fontSize}}px;
                        font-weight: bold;
                        border: 2px solid white;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.4);
                    ">${{label}}</div>`,
                    iconSize: [markerSize, markerSize],
                    iconAnchor: [markerSize/2, markerSize/2]
                }});
                
                let popupText = `<b>#${{i+1}}: ${{stop.name}}</b><br>${{stop.address}}`;
                if (i === 0) popupText = `<b>🟢 START: ${{stop.name}}</b><br>${{stop.address}}`;
                if (i === lastIdx) popupText = `<b>🔴 END: ${{stop.name}}</b><br>${{stop.address}}`;
                
                L.marker([stop.lat, stop.lon], {{ icon }})
                    .bindPopup(popupText)
                    .addTo(markerGroup);
            }});
            markerGroup.addTo(map);
            
            // Store layers for later removal
            routeLayers.push({{ polygon: polygonLayer, line: routeLine, markers: markerGroup, label: labelMarker }});
        }}
        
        function updateZoneList() {{
            const container = document.getElementById('zoneList');
            document.getElementById('zoneCount').textContent = zones.length;
            
            if (zones.length === 0) {{
                container.innerHTML = '<p style="color: #999; font-size: 13px;">No zones yet.</p>';
                return;
            }}
            
            container.innerHTML = zones.map((zone, i) => `
                <div class="zone-item">
                    <div class="name" onclick="focusZone(${{i}})" style="cursor: pointer;" title="Click to view zone on map">
                        <span class="zone-color" style="background: ${{zoneColors[i % zoneColors.length]}}"></span>
                        ${{zone.name}}
                    </div>
                    <div class="stats">
                        📍 ${{zone.total_stops}} stops | 📏 ${{zone.total_distance_km}} km
                        <button onclick="deleteZone(${{i}})" class="delete-btn" title="Delete zone">🗑️</button>
                    </div>
                </div>
            `).join('');
        }}
        
        function focusZone(index) {{
            // Pan and zoom the map to show the selected zone
            if (routeLayers[index] && routeLayers[index].polygon) {{
                const bounds = routeLayers[index].polygon.getBounds();
                map.fitBounds(bounds, {{ padding: [50, 50] }});
            }}
        }}
        
        async function deleteZone(index) {{
            const zoneName = zones[index].name;
            if (!confirm(`Are you sure you want to delete zone "${{zoneName}}"?`)) return;
            
            try {{
                const response = await fetch('/api/delete-zone', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ zone_index: index }})
                }});
                
                const result = await response.json();
                
                if (result.success) {{
                    // Remove layers from map
                    const layer = routeLayers[index];
                    if (layer) {{
                        map.removeLayer(layer.polygon);
                        map.removeLayer(layer.line);
                        map.removeLayer(layer.markers);
                        if (layer.label) map.removeLayer(layer.label);
                    }}
                    
                    // Remove from arrays
                    routeLayers.splice(index, 1);
                    zones.splice(index, 1);
                    
                    // Re-draw all zones to update colors
                    redrawAllZones();
                    updateZoneList();
                }}
            }} catch (err) {{
                alert('Error: ' + err.message);
            }}
        }}
        
        function redrawAllZones() {{
            // Remove all route layers
            routeLayers.forEach(layer => {{
                map.removeLayer(layer.polygon);
                map.removeLayer(layer.line);
                map.removeLayer(layer.markers);
                if (layer.label) map.removeLayer(layer.label);
            }});
            routeLayers = [];
            
            // Clear arrow markers
            clearArrowMarkers();
            
            // Re-draw with correct colors
            zones.forEach((zone, i) => {{
                drawRoute(zone, i);
            }});
        }}
        
        function cancelZone() {{
            drawnItems.clearLayers();
            currentPolygon = null;
            selectedStops = [];
            document.getElementById('zoneSetup').classList.add('hidden');
            setStep(1);
        }}
        
        async function clearAllZones() {{
            if (!confirm('Are you sure you want to delete all zones?')) return;
            
            try {{
                await fetch('/api/clear', {{ method: 'POST' }});
                
                // Remove all route layers from map
                routeLayers.forEach(layer => {{
                    map.removeLayer(layer.polygon);
                    map.removeLayer(layer.line);
                    map.removeLayer(layer.markers);
                    if (layer.label) map.removeLayer(layer.label);
                }});
                routeLayers = [];
                zones = [];
                
                // Clear arrow markers
                clearArrowMarkers();
                
                updateZoneList();
                alert('✅ All zones cleared');
            }} catch (err) {{
                alert('Error: ' + err.message);
            }}
        }}
        
        // Load existing zones on page load
        async function loadExistingZones() {{
            try {{
                const response = await fetch('/api/zones');
                const data = await response.json();
                zones = data.zones || [];
                
                zones.forEach((zone, i) => {{
                    drawRoute(zone, i);
                }});
                
                updateZoneList();
            }} catch (err) {{
                console.log('No existing zones');
            }}
        }}
        
        loadExistingZones();
    </script>
</body>
</html>'''
    
    return html

# ============================================================================
# Main
# ============================================================================

def main():
    os.chdir(WORKING_DIR)
    
    print("=" * 60)
    print("🗺️  SMR Path Optimization Tool")
    print("=" * 60)
    
    # Load stops
    print(f"\n📂 Loading data from {DATA_FILE}...")
    stops = load_stops(DATA_FILE)
    print(f"   ✅ Loaded {len(stops)} stops")
    
    # Load existing zones
    RequestHandler.stops = stops
    RequestHandler.zones_data = load_zones_from_file()
    print(f"   📋 Loaded {len(RequestHandler.zones_data['zones'])} existing zones")
    
    # Get local IP address (first non-localhost IP)
    import socket
    import subprocess
    try:
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        local_ip = result.stdout.strip().split()[0]
    except:
        local_ip = '127.0.0.1'
    
    # Start server on all interfaces (0.0.0.0)
    print(f"\n🌐 Starting server...")
    print(f"   Local:   http://localhost:{PORT}")
    print(f"   Network: http://{local_ip}:{PORT}")
    server = HTTPServer(('0.0.0.0', PORT), RequestHandler)
    
    # Open browser
    import webbrowser
    webbrowser.open(f'http://localhost:{PORT}')
    
    print("\n" + "=" * 60)
    print("INSTRUCTIONS:")
    print("=" * 60)
    print("1. Draw a polygon zone on the map")
    print("2. Enter a name for the zone")
    print("3. Select a start point from dropdown")
    print("4. Click 'Calculate Optimized Route'")
    print("5. Repeat to add more zones!")
    print("")
    print(f"📱 Access from other devices: http://{local_ip}:{PORT}")
    print(f"📁 All zones saved to: {OUTPUT_FILE}")
    print("=" * 60)
    print("\n✅ Server running. Press Ctrl+C to stop.\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")

if __name__ == '__main__':
    main()
