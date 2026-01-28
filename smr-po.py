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

def build_distance_matrix(stops):
    """Build distance matrix for all stops"""
    n = len(stops)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = haversine_distance(
                    stops[i]['lat'], stops[i]['lon'],
                    stops[j]['lat'], stops[j]['lon']
                )
    return matrix

def nearest_neighbor_route(stops, start_idx=0):
    """Find route using nearest neighbor algorithm"""
    n = len(stops)
    if n == 0:
        return [], 0
    
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
    
    return route, total_dist

def two_opt_improve(stops, route, max_iterations=30):
    """Improve route using 2-opt algorithm"""
    matrix = build_distance_matrix(stops)
    n = len(route)
    
    def route_length(r):
        return sum(matrix[r[i]][r[i+1]] for i in range(len(r)-1))
    
    best_distance = route_length(route)
    iterations = 0
    
    improved = True
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(1, min(n - 2, 100)):
            for j in range(i + 2, min(n, i + 50)):
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_dist = route_length(new_route)
                if new_dist < best_distance:
                    route = new_route
                    best_distance = new_dist
                    improved = True
                    break
            if improved:
                break
    
    return route, best_distance

def optimize_route(stops, start_idx=0):
    """Optimize route through stops"""
    if len(stops) <= 1:
        return stops, 0
    
    route_idx, _ = nearest_neighbor_route(stops, start_idx)
    route_idx, total_dist = two_opt_improve(stops, route_idx, max_iterations=30)
    
    optimized = [stops[i] for i in route_idx]
    return optimized, total_dist

# ============================================================================
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
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = generate_main_page(RequestHandler.stops)
            self.wfile.write(html.encode())
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
            
            # Optimize route
            optimized, distance = optimize_route(stops, start_idx)
            
            # Create zone data
            zone_data = {
                'name': zone_name,
                'polygon': polygon,
                'total_stops': len(optimized),
                'total_distance_km': round(distance, 2),
                'route': optimized
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
        .zone-item .name {{ font-weight: bold; color: #333; }}
        .zone-item .stats {{ color: #666; margin-top: 5px; }}
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
        
        // Initialize map
        const map = L.map('map').setView([{center_lat}, {center_lon}], 12);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap'
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
        
        function drawRoute(zone, index) {{
            const color = zoneColors[index % zoneColors.length];
            const route = zone.route;
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
            
            // Draw route line
            const routeCoords = route.map(s => [s.lat, s.lon]);
            const routeLine = L.polyline(routeCoords, {{
                color: color,
                weight: 3,
                opacity: 0.8
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
                    <div class="name">
                        <span class="zone-color" style="background: ${{zoneColors[i % zoneColors.length]}}"></span>
                        ${{zone.name}}
                    </div>
                    <div class="stats">
                        📍 ${{zone.total_stops}} stops | 📏 ${{zone.total_distance_km}} km
                    </div>
                </div>
            `).join('');
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
