const bootstrap = window.SMR_BOOTSTRAP;
const allStops = bootstrap.stops;
const zoneColors = ['#e91e63', '#9c27b0', '#673ab7', '#3f51b5', '#2196f3', '#00bcd4', '#009688', '#4caf50', '#ff9800', '#ff5722'];

let selectedStops = [];
let currentPolygon = null;
let zones = [];
let routeLayers = [];
let currentStep = 1;

// Initialize map with higher max zoom
const map = L.map('map', {
    maxZoom: 22
}).setView(bootstrap.center, 12);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap',
    maxZoom: 22,
    maxNativeZoom: 19
}).addTo(map);

// Marker cluster for all stops
const markers = L.markerClusterGroup({ disableClusteringAtZoom: 16 });
allStops.forEach(stop => {
    const marker = L.marker([stop.lat, stop.lon])
        .bindPopup(`<b>${stop.name}</b><br>${stop.address}`);
    markers.addLayer(marker);
});
map.addLayer(markers);

// Drawing layer
const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

// Drawing controls
const drawControl = new L.Control.Draw({
    draw: {
        polygon: { shapeOptions: { color: '#ff9800' } },
        rectangle: { shapeOptions: { color: '#ff9800' } },
        circle: false,
        circlemarker: false,
        marker: false,
        polyline: false
    },
    edit: false
});
map.addControl(drawControl);

// Handle polygon drawing
map.on('draw:created', function(e) {
    // Clear previous temp polygon
    drawnItems.clearLayers();
    currentPolygon = e.layer;
    drawnItems.addLayer(currentPolygon);

    // Filter stops inside polygon
    selectedStops = allStops.filter(stop => {
        return isPointInPolygon(L.latLng(stop.lat, stop.lon), currentPolygon);
    });

    if (selectedStops.length > 0) {
        document.getElementById('selectedCount').textContent = selectedStops.length;
        populateDropdown();
        document.getElementById('zoneSetup').classList.remove('hidden');
        document.getElementById('zoneName').value = '';

        // Calculate and show valid zone range
        updateZoneRangeInfo(selectedStops.length);

        document.getElementById('zoneName').focus();
        setStep(2);
    } else {
        alert('⚠️ No stops found in this area. Try drawing a larger zone.');
        drawnItems.clearLayers();
        currentPolygon = null;
    }
});

function isPointInPolygon(point, polygon) {
    const polyPoints = polygon.getLatLngs()[0];
    let inside = false;
    for (let i = 0, j = polyPoints.length - 1; i < polyPoints.length; j = i++) {
        const xi = polyPoints[i].lat, yi = polyPoints[i].lng;
        const xj = polyPoints[j].lat, yj = polyPoints[j].lng;
        if (((yi > point.lng) !== (yj > point.lng)) &&
            (point.lat < (xj - xi) * (point.lng - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

function populateDropdown() {
    const dropdown = document.getElementById('startDropdown');
    dropdown.innerHTML = '<option value="">-- Select start point --</option>';
    selectedStops.forEach((stop, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = stop.name;
        dropdown.appendChild(opt);
    });
    dropdown.onchange = function() {
        document.getElementById('calcBtn').disabled = (this.value === '');
        if (this.value !== '') setStep(3);
    };
}

function setStep(step) {
    currentStep = step;
    for (let i = 1; i <= 3; i++) {
        const el = document.getElementById('step' + i);
        el.classList.remove('active', 'done');
        if (i < step) el.classList.add('done');
        if (i === step) el.classList.add('active');
    }
}

async function calculateRoute() {
    const zoneName = document.getElementById('zoneName').value.trim() || 'Zone ' + (zones.length + 1);
    const startIdx = parseInt(document.getElementById('startDropdown').value);

    if (isNaN(startIdx)) {
        alert('Please select a start point');
        return;
    }

    // Show loading
    document.getElementById('loading').style.display = 'block';

    try {
        const response = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                zone_name: zoneName,
                stops: selectedStops,
                start_idx: startIdx,
                polygon: currentPolygon.getLatLngs()[0].map(p => [p.lat, p.lng])
            })
        });

        const result = await response.json();

        if (result.success) {
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

            alert(`✅ Zone "${zoneName}" created!\\n📍 ${result.zone.total_stops} stops\\n📏 ${result.zone.total_distance_km} km`);
        }
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

// Global array to store arrow markers for cleanup
let arrowMarkers = [];

function addArrowsToRoute(coords, color) {
    // Add arrow markers along the route to show direction
    // Use longer lookahead for consistent direction calculation
    const arrowInterval = 15; // Add arrow every N points
    const lookAhead = 10; // Points to look ahead for direction (smoother)

    for (let i = arrowInterval; i < coords.length - lookAhead; i += arrowInterval) {
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
        const arrowIcon = L.divIcon({
            className: 'route-arrow',
            html: `<div style="
                color: ${color};
                font-size: 12px;
                font-weight: bold;
                transform: rotate(${rotation}deg);
                text-shadow: 1px 1px 1px white, -1px -1px 1px white, 1px -1px 1px white, -1px 1px 1px white;
                line-height: 1;
            ">▶</div>`,
            iconSize: [12, 12],
            iconAnchor: [6, 6]
        });

        const arrowMarker = L.marker([p1[0], p1[1]], { icon: arrowIcon }).addTo(map);
        arrowMarkers.push(arrowMarker);
    }
}

function clearArrowMarkers() {
    arrowMarkers.forEach(m => map.removeLayer(m));
    arrowMarkers = [];
}

function drawRoute(zone, index) {
    const color = zoneColors[index % zoneColors.length];
    const route = zone.route;
    const roadGeometry = zone.road_geometry;  // Actual road path from OSMnx
    const startColor = '#00C853';  // Green for start
    const endColor = '#FF1744';    // Red for end

    // Draw polygon
    const polygonLayer = L.polygon(zone.polygon, {
        color: color,
        fillColor: color,
        fillOpacity: 0.1,
        weight: 2
    }).addTo(map);

    // Calculate polygon center for zone label
    const bounds = polygonLayer.getBounds();
    const center = bounds.getCenter();

    // Add zone name label on map
    const labelIcon = L.divIcon({
        className: 'zone-label',
        html: `<div style="
            background: ${color};
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            white-space: nowrap;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            border: 2px solid white;
        ">${zone.name}</div>`,
        iconSize: [100, 30],
        iconAnchor: [50, 15]
    });
    const labelMarker = L.marker([center.lat, center.lng], { icon: labelIcon }).addTo(map);

    // Draw route line - use road geometry if available, otherwise straight lines
    let routeCoords;
    if (roadGeometry && roadGeometry.length > 0) {
        routeCoords = roadGeometry;  // Use actual road path
        console.log('Drawing road path:', roadGeometry.length, 'points');
    } else {
        routeCoords = route.map(s => [s.lat, s.lon]);  // Fallback to straight lines
        console.log('Drawing straight lines (no road geometry)');
    }

    const routeLine = L.polyline(routeCoords, {
        color: color,
        weight: 3,
        opacity: 0.9
    }).addTo(map);

    // Add numbered markers with special colors for start and end
    const markerGroup = L.layerGroup();
    const lastIdx = route.length - 1;

    route.forEach((stop, i) => {
        let markerColor = color;
        let markerSize = 20;
        let fontSize = 10;
        let label = i + 1;

        // Start point - green, larger
        if (i === 0) {
            markerColor = startColor;
            markerSize = 28;
            fontSize = 12;
            label = '▶';
        }
        // End point - red, larger
        else if (i === lastIdx) {
            markerColor = endColor;
            markerSize = 28;
            fontSize = 12;
            label = '◼';
        }

        const icon = L.divIcon({
            className: 'route-marker',
            html: `<div style="
                background: ${markerColor};
                color: white;
                width: ${markerSize}px;
                height: ${markerSize}px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: ${fontSize}px;
                font-weight: bold;
                border: 2px solid white;
                box-shadow: 0 2px 5px rgba(0,0,0,0.4);
            ">${label}</div>`,
            iconSize: [markerSize, markerSize],
            iconAnchor: [markerSize/2, markerSize/2]
        });

        let popupText = `<b>#${i+1}: ${stop.name}</b><br>${stop.address}`;
        if (i === 0) popupText = `<b>🟢 START: ${stop.name}</b><br>${stop.address}`;
        if (i === lastIdx) popupText = `<b>🔴 END: ${stop.name}</b><br>${stop.address}`;

        L.marker([stop.lat, stop.lon], { icon })
            .bindPopup(popupText)
            .addTo(markerGroup);
    });
    markerGroup.addTo(map);

    // Store layers for later removal
    routeLayers.push({ polygon: polygonLayer, line: routeLine, markers: markerGroup, label: labelMarker });
}

function updateZoneList() {
    const container = document.getElementById('zoneList');
    document.getElementById('zoneCount').textContent = zones.length;

    if (zones.length === 0) {
        container.innerHTML = '<p style="color: #999; font-size: 13px;">No zones yet.</p>';
        return;
    }

    container.innerHTML = zones.map((zone, i) => `
        <div class="zone-item" id="zone-item-${i}">
            <div class="name" id="zone-name-${i}" onclick="focusZone(${i})" style="cursor: pointer;" title="Click to view zone on map">
                <span class="zone-color" style="background: ${zoneColors[i % zoneColors.length]}"></span>
                <span id="zone-name-text-${i}">${zone.name}</span>
            </div>
            <div class="stats">
                📍 ${zone.total_stops} stops | 📏 ${zone.total_distance_km} km
                <div class="zone-actions">
                    <button onclick="event.stopPropagation(); startRenameZone(${i})" class="rename-btn" title="Rename zone">✏️</button>
                    <button onclick="event.stopPropagation(); deleteZone(${i})" class="delete-btn" title="Delete zone">🗑️</button>
                </div>
            </div>
        </div>
    `).join('');
}

// Track if we're currently saving (to prevent blur cancel)
let isSaving = false;

function focusZone(index) {
    // Pan and zoom the map to show the selected zone
    if (routeLayers[index] && routeLayers[index].polygon) {
        const bounds = routeLayers[index].polygon.getBounds();
        map.fitBounds(bounds, { padding: [50, 50] });
    }
}

function startRenameZone(index) {
    event.stopPropagation();  // Prevent focusZone from triggering

    const nameContainer = document.getElementById(`zone-name-${index}`);
    const currentName = zones[index].name;

    // Replace with input field
    nameContainer.innerHTML = `
        <input type="text" class="zone-name-input" id="rename-input-${index}" value="${currentName}">
        <button id="save-btn-${index}" style="background:#4CAF50;color:white;padding:3px 8px;border:none;border-radius:3px;cursor:pointer;font-size:11px;width:auto;margin-left:5px;">\u2713</button>
    `;

    const input = document.getElementById(`rename-input-${index}`);
    const saveBtn = document.getElementById(`save-btn-${index}`);

    // Focus and select the input
    input.focus();
    input.select();

    // Handle Enter key
    input.onkeypress = function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            saveRenameZone(index, currentName);
        }
    };

    // Handle Escape key to cancel
    input.onkeydown = function(e) {
        if (e.key === 'Escape') {
            cancelRenameZone(index, currentName);
        }
    };

    // Save button click
    saveBtn.onmousedown = function(e) {
        e.preventDefault();  // Prevent blur from firing
        isSaving = true;
    };
    saveBtn.onclick = function(e) {
        e.stopPropagation();
        saveRenameZone(index, currentName);
    };

    // Handle blur (click outside)
    input.onblur = function(e) {
        setTimeout(() => {
            if (!isSaving) {
                cancelRenameZone(index, currentName);
            }
            isSaving = false;
        }, 150);
    };
}

async function saveRenameZone(index, originalName) {
    const input = document.getElementById(`rename-input-${index}`);
    if (!input) return;

    const newName = input.value.trim();
    if (!newName) {
        alert('Zone name cannot be empty');
        cancelRenameZone(index, originalName);
        return;
    }

    if (newName === originalName) {
        cancelRenameZone(index, originalName);
        return;
    }

    try {
        const response = await fetch('/api/rename-zone', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ zone_index: index, new_name: newName })
        });

        const result = await response.json();

        if (result.success) {
            // Update local zone data
            zones[index].name = newName;

            // Update sidebar list
            updateZoneList();

            // Update map labels by redrawing
            redrawAllZones();

            console.log('Zone renamed successfully:', newName);
        } else {
            alert('Failed to rename zone: ' + (result.error || 'Unknown error'));
            cancelRenameZone(index, originalName);
        }
    } catch (err) {
        alert('Error: ' + err.message);
        cancelRenameZone(index, originalName);
    }

    isSaving = false;
}

function cancelRenameZone(index, originalName) {
    const nameContainer = document.getElementById(`zone-name-${index}`);
    if (nameContainer) {
        const color = zoneColors[index % zoneColors.length];
        nameContainer.innerHTML = `
            <span class="zone-color" style="background: ${color}"></span>
            <span id="zone-name-text-${index}">${originalName}</span>
        `;
    }
}

async function deleteZone(index) {
    const zoneName = zones[index].name;
    if (!confirm(`Are you sure you want to delete zone "${zoneName}"?`)) return;

    try {
        const response = await fetch('/api/delete-zone', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ zone_index: index })
        });

        const result = await response.json();

        if (result.success) {
            // Remove layers from map
            const layer = routeLayers[index];
            if (layer) {
                map.removeLayer(layer.polygon);
                map.removeLayer(layer.line);
                map.removeLayer(layer.markers);
                if (layer.label) map.removeLayer(layer.label);
            }

            // Remove from arrays
            routeLayers.splice(index, 1);
            zones.splice(index, 1);

            // Re-draw all zones to update colors
            redrawAllZones();
            updateZoneList();
        }
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

function redrawAllZones() {
    // Remove all route layers
    routeLayers.forEach(layer => {
        map.removeLayer(layer.polygon);
        map.removeLayer(layer.line);
        map.removeLayer(layer.markers);
        if (layer.label) map.removeLayer(layer.label);
    });
    routeLayers = [];

    // Clear arrow markers
    clearArrowMarkers();

    // Re-draw with correct colors
    zones.forEach((zone, i) => {
        drawRoute(zone, i);
    });
}

function cancelZone() {
    drawnItems.clearLayers();
    currentPolygon = null;
    selectedStops = [];
    document.getElementById('zoneSetup').classList.add('hidden');
    setStep(1);
}

async function clearAllZones() {
    if (!confirm('Are you sure you want to delete all zones?')) return;

    try {
        await fetch('/api/clear', { method: 'POST' });

        // Remove all route layers from map
        routeLayers.forEach(layer => {
            map.removeLayer(layer.polygon);
            map.removeLayer(layer.line);
            map.removeLayer(layer.markers);
            if (layer.label) map.removeLayer(layer.label);
        });
        routeLayers = [];
        zones = [];

        // Clear arrow markers
        clearArrowMarkers();

        updateZoneList();
        alert('✅ All zones cleared');
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

function updateZoneRangeInfo(totalStops) {
    const infoDiv = document.getElementById('zoneRangeInfo');
    const targetInput = document.getElementById('targetStops');
    const targetStops = parseInt(targetInput.value) || 100;

    if (totalStops < 2) {
        infoDiv.innerHTML = `⚠️ Need at least <b>2</b> stops. You have <b>${totalStops}</b>.`;
        infoDiv.style.background = '#ffebee';
        infoDiv.style.color = '#c62828';
        document.getElementById('numZones').disabled = true;
        return;
    }

    // Suggest zones based on target stops per zone
    const suggested = Math.max(1, Math.round(totalStops / targetStops));
    const maxZones = Math.min(50, totalStops);

    infoDiv.innerHTML = `📊 <b>${totalStops}</b> stops → <b>${suggested}</b> zones (~${targetStops} each)<br>
        <span style="font-size:10px;">Balanced for fair SMR workload</span>`;
    infoDiv.style.background = '#e3f2fd';
    infoDiv.style.color = '#1565c0';
    document.getElementById('numZones').disabled = false;
    document.getElementById('numZones').min = 1;
    document.getElementById('numZones').max = maxZones;
    document.getElementById('numZones').value = '';
    document.getElementById('numZones').placeholder = `Auto (${suggested})`;
}

// Update zone count when target changes
document.getElementById('targetStops').addEventListener('change', function() {
    updateZoneRangeInfo(selectedStops.length);
});

async function autoCreateZones() {
    const totalStops = selectedStops.length;

    if (totalStops < 2) {
        alert(`⚠️ Not enough stops!\n\nYou need at least 2 stops.\nCurrently selected: ${totalStops} stops`);
        return;
    }

    const targetStops = parseInt(document.getElementById('targetStops').value) || 100;
    let numZones = parseInt(document.getElementById('numZones').value);
    const maxZones = Math.min(50, totalStops);

    // Auto-calculate if not provided
    if (!numZones || isNaN(numZones)) {
        numZones = Math.max(1, Math.round(totalStops / targetStops));
    }

    if (numZones < 1 || numZones > maxZones) {
        alert(`⚠️ Invalid zone count!\n\nPlease enter a value between 1 and ${maxZones}.`);
        return;
    }

    const avgPerZone = Math.round(totalStops / numZones);
    if (!confirm(`⚖️ Create ${numZones} DISTANCE-BALANCED zones?\n\n📊 Configuration:\n• Total stops: ${totalStops}\n• Zones: ${numZones}\n• ~${avgPerZone} stops per zone\n\n✨ All zones will have SIMILAR TRAVEL DISTANCE\n(e.g., Zone 1: 16km, Zone 2: 17km, Zone 3: 18km)\n\nThis ensures fair workload for SMR assignment.`)) {
        return;
    }

    // Show loading
    document.getElementById('loading').textContent = `⚖️ Creating ${numZones} distance-balanced zones...\nOptimizing for equal travel distance.`;
    document.getElementById('loading').style.display = 'block';

    try {
        const response = await fetch('/api/auto-create-zones', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                stops: selectedStops,
                num_zones: numZones,
                target_stops: targetStops
            })
        });

        const result = await response.json();

        if (result.success) {
            // Add zones to list
            result.zones.forEach(zone => {
                zones.push(zone);
            });

            // Draw all new zones
            redrawAllZones();
            updateZoneList();

            // Clear temp polygon and hide setup
            drawnItems.clearLayers();
            currentPolygon = null;
            selectedStops = [];
            document.getElementById('zoneSetup').classList.add('hidden');
            setStep(1);

            // Show summary
            let summary = `✅ Created ${result.zones_created} zones automatically!\n\n`;
            result.zones.forEach((z, i) => {
                summary += `${z.name}: ${z.total_stops} stops, ${z.total_distance_km} km\n`;
            });
            alert(summary);
        } else {
            alert('❌ Error: ' + result.error);
        }
    } catch (err) {
        alert('❌ Error: ' + err.message);
    } finally {
        document.getElementById('loading').textContent = '⏳ Optimizing route...';
        document.getElementById('loading').style.display = 'none';
    }
}

// Load existing zones on page load
async function loadExistingZones() {
    try {
        const response = await fetch('/api/zones');
        const data = await response.json();
        zones = data.zones || [];

        zones.forEach((zone, i) => {
            drawRoute(zone, i);
        });

        updateZoneList();
    } catch (err) {
        console.log('No existing zones');
    }
}

loadExistingZones();
