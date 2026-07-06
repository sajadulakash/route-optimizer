const bootstrap = window.SMR_BOOTSTRAP;
const allStops = bootstrap.stops;
const zoneColors = ['#e91e63', '#9c27b0', '#673ab7', '#3f51b5', '#2196f3', '#00bcd4', '#009688', '#4caf50', '#ff9800', '#ff5722'];
const MIN_STOP_RENDER_ZOOM = 17;
const MAX_VISIBLE_STOPS = 600;


let map;
let stopLayer;
let infoWindow;
let HtmlMapMarker;
let selectedStops = [];
let currentSelectionOverlay = null;
let zones = [];
let routeLayers = [];
let currentStep = 1;
let drawingMode = null;
let drawingPoints = [];
let drawingPreview = null;
let drawingVertexMarkers = [];
let isSaving = false;
let stopRenderTimer = null;
let planningRequestTimer = null;
let pendingPlanningRequests = [];


function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function defineHtmlMapMarker() {
    HtmlMapMarker = class extends google.maps.OverlayView {
        constructor(mapInstance, position, html, className, onClick) {
            super();
            this.position = position instanceof google.maps.LatLng
                ? position
                : new google.maps.LatLng(position);
            this.html = html;
            this.className = className;
            this.onClick = onClick;
            this.div = null;
            this.setMap(mapInstance);
        }

        onAdd() {
            this.div = document.createElement('div');
            this.div.className = this.className;
            this.div.innerHTML = this.html;
            if (this.onClick) {
                this.div.addEventListener('click', this.onClick);
            }
            this.getPanes().overlayMouseTarget.appendChild(this.div);
        }

        draw() {
            if (!this.div) return;
            const point = this.getProjection().fromLatLngToDivPixel(this.position);
            if (!point) return;
            this.div.style.left = `${point.x}px`;
            this.div.style.top = `${point.y}px`;
        }

        onRemove() {
            if (this.div) {
                this.div.remove();
                this.div = null;
            }
        }
    };
}

window.initMap = function initMap() {
    defineHtmlMapMarker();

    map = new google.maps.Map(document.getElementById('map'), {
        center: { lat: bootstrap.center[0], lng: bootstrap.center[1] },
        zoom: 12,
        mapTypeControl: true,
        streetViewControl: false,
        fullscreenControl: true,
        gestureHandling: 'greedy'
    });

    infoWindow = new google.maps.InfoWindow();
    addStopsToMap();
    map.addListener('click', handleDrawingClick);
    map.addListener('idle', scheduleVisibleStopRender);
    loadExistingZones();
    loadPlanningRequests();
    planningRequestTimer = window.setInterval(loadPlanningRequests, 4000);
    setDrawingStatus('Zoom in or draw an area. Visible shops load by viewport.');
    scheduleVisibleStopRender();
};

window.gm_authFailure = function gmAuthFailure() {
    document.getElementById('map').innerHTML = '<div class="map-error">Google Maps could not authenticate. Check GOOGLE_MAPS_BROWSER_API_KEY and its HTTP referrer restrictions.</div>';
};

function addStopsToMap() {
    stopLayer = new google.maps.Data({ map });
    stopLayer.setStyle({
        icon: {
            path: google.maps.SymbolPath.CIRCLE,
            fillColor: '#1976d2',
            fillOpacity: 0.78,
            strokeColor: '#ffffff',
            strokeWeight: 1,
            scale: 4
        }
    });

    stopLayer.addListener('click', event => {
        const name = escapeHtml(event.feature.getProperty('name'));
        const address = escapeHtml(event.feature.getProperty('address'));
        infoWindow.setContent(`<b>${name}</b><br>${address}`);
        infoWindow.setPosition(event.latLng);
        infoWindow.open({ map });
    });
}

function clearVisibleStops() {
    if (!stopLayer) return;
    stopLayer.forEach(feature => stopLayer.remove(feature));
}

function scheduleVisibleStopRender() {
    window.clearTimeout(stopRenderTimer);
    stopRenderTimer = window.setTimeout(renderVisibleStops, 120);
}

function renderVisibleStops() {
    if (!map || !stopLayer) return;

    const bounds = map.getBounds();
    if (!bounds) return;

    clearVisibleStops();

    if (map.getZoom() < MIN_STOP_RENDER_ZOOM) {
        if (!drawingMode && !currentSelectionOverlay) {
            setDrawingStatus(`Zoom in to level ${MIN_STOP_RENDER_ZOOM}+ to display nearby shops. Drawing still selects from all ${allStops.length} shops.`);
        }
        return;
    }

    const southWest = bounds.getSouthWest();
    const northEast = bounds.getNorthEast();
    const south = southWest.lat();
    const north = northEast.lat();
    const west = southWest.lng();
    const east = northEast.lng();
    const crossesAntimeridian = west > east;
    let inViewCount = 0;
    let renderedCount = 0;

    for (const stop of allStops) {
        const inLat = stop.lat >= south && stop.lat <= north;
        const inLng = crossesAntimeridian
            ? stop.lon >= west || stop.lon <= east
            : stop.lon >= west && stop.lon <= east;
        if (!inLat || !inLng) continue;

        inViewCount += 1;
        if (renderedCount >= MAX_VISIBLE_STOPS) continue;

        stopLayer.add({
            geometry: new google.maps.Data.Point(new google.maps.LatLng(stop.lat, stop.lon)),
            properties: {
                id: stop.id,
                name: stop.name,
                address: stop.address
            }
        });
        renderedCount += 1;
    }

    if (!drawingMode && !currentSelectionOverlay) {
        const capped = inViewCount > renderedCount ? ` Showing first ${renderedCount}. Zoom in for more detail.` : '';
        setDrawingStatus(`${renderedCount} of ${inViewCount} shops visible in this viewport.${capped}`);
    }
}

function setDrawingStatus(message) {
    document.getElementById('drawingStatus').textContent = message;
}

function setDrawingButtons() {
    document.querySelectorAll('[data-drawing-mode]').forEach(button => {
        button.classList.toggle('active', button.dataset.drawingMode === drawingMode);
    });
    document.getElementById('finishDrawingBtn').disabled = drawingMode !== 'polygon' || drawingPoints.length < 3;
}

function startDrawing(mode) {
    clearCurrentSelection();
    clearDrawingDraft();
    drawingMode = mode;
    map.setOptions({ draggableCursor: 'crosshair' });
    setDrawingButtons();

    if (mode === 'polygon') {
        setDrawingStatus('Click at least 3 points, then click Finish Polygon.');
    } else {
        setDrawingStatus('Click two opposite corners to create a rectangle.');
    }
}

function handleDrawingClick(event) {
    if (!drawingMode) return;

    const point = { lat: event.latLng.lat(), lng: event.latLng.lng() };
    drawingPoints.push(point);

    if (drawingMode === 'rectangle' && drawingPoints.length === 2) {
        const first = drawingPoints[0];
        const second = drawingPoints[1];
        completeSelection([
            { lat: first.lat, lng: first.lng },
            { lat: first.lat, lng: second.lng },
            { lat: second.lat, lng: second.lng },
            { lat: second.lat, lng: first.lng }
        ]);
        return;
    }

    updateDrawingPreview();
    setDrawingButtons();
}

function updateDrawingPreview() {
    if (drawingPreview) drawingPreview.setMap(null);
    drawingPreview = new google.maps.Polygon({
        map,
        paths: drawingPoints,
        strokeColor: '#ff9800',
        strokeOpacity: 1,
        strokeWeight: 2,
        fillColor: '#ff9800',
        fillOpacity: drawingPoints.length >= 3 ? 0.12 : 0,
        clickable: false
    });

    drawingVertexMarkers.forEach(marker => marker.setMap(null));
    drawingVertexMarkers = drawingPoints.map(point => new HtmlMapMarker(
        map,
        point,
        '<span></span>',
        'drawing-vertex'
    ));
}

function finishDrawing() {
    if (drawingMode !== 'polygon' || drawingPoints.length < 3) return;
    completeSelection([...drawingPoints]);
}

function completeSelection(path) {
    clearDrawingDraft();
    currentSelectionOverlay = new google.maps.Polygon({
        map,
        paths: path,
        strokeColor: '#ff9800',
        strokeOpacity: 1,
        strokeWeight: 2,
        fillColor: '#ff9800',
        fillOpacity: 0.14,
        clickable: false
    });

    drawingMode = null;
    map.setOptions({ draggableCursor: null });
    setDrawingButtons();

    selectedStops = allStops.filter(stop => {
        const location = new google.maps.LatLng(stop.lat, stop.lon);
        return google.maps.geometry.poly.containsLocation(location, currentSelectionOverlay) ||
            google.maps.geometry.poly.isLocationOnEdge(location, currentSelectionOverlay, 1e-9);
    });

    if (selectedStops.length === 0) {
        alert('No stops found in this area. Try drawing a larger zone.');
        clearCurrentSelection();
        setDrawingStatus('No stops selected. Draw another area.');
        return;
    }

    document.getElementById('selectedCount').textContent = selectedStops.length;
    populateDropdown();
    document.getElementById('zoneSetup').classList.remove('hidden');
    document.getElementById('zoneName').value = '';
    updateZoneRangeInfo(selectedStops.length);
    document.getElementById('zoneName').focus();
    setStep(2);
    setDrawingStatus(`${selectedStops.length} stops selected.`);
}

function clearDrawingDraft() {
    if (drawingPreview) {
        drawingPreview.setMap(null);
        drawingPreview = null;
    }
    drawingVertexMarkers.forEach(marker => marker.setMap(null));
    drawingVertexMarkers = [];
    drawingPoints = [];
}

function clearCurrentSelection() {
    if (currentSelectionOverlay) {
        currentSelectionOverlay.setMap(null);
        currentSelectionOverlay = null;
    }
}

function cancelDrawing() {
    drawingMode = null;
    clearDrawingDraft();
    clearCurrentSelection();
    selectedStops = [];
    map.setOptions({ draggableCursor: null });
    setDrawingButtons();
    setDrawingStatus('Drawing cancelled.');
}

function getCurrentPolygonCoordinates() {
    if (!currentSelectionOverlay) return [];
    return currentSelectionOverlay.getPath().getArray().map(point => [point.lat(), point.lng()]);
}

function populateDropdown() {
    const dropdown = document.getElementById('startDropdown');
    dropdown.innerHTML = '<option value="">-- Select start point --</option>';
    selectedStops.forEach((stop, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = stop.name;
        dropdown.appendChild(option);
    });
    dropdown.onchange = function onStartChange() {
        document.getElementById('calcBtn').disabled = this.value === '';
        if (this.value !== '') setStep(3);
    };
}

function setStep(step) {
    currentStep = step;
    for (let index = 1; index <= 3; index += 1) {
        const element = document.getElementById(`step${index}`);
        element.classList.remove('active', 'done');
        if (index < step) element.classList.add('done');
        if (index === step) element.classList.add('active');
    }
}

async function calculateRoute() {
    const zoneName = document.getElementById('zoneName').value.trim() || `Zone ${zones.length + 1}`;
    const startIdx = parseInt(document.getElementById('startDropdown').value, 10);

    if (Number.isNaN(startIdx)) {
        alert('Please select a start point');
        return;
    }

    document.getElementById('loading').style.display = 'block';

    try {
        const response = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                zone_name: zoneName,
                stops: selectedStops,
                start_idx: startIdx,
                polygon: getCurrentPolygonCoordinates()
            })
        });
        const result = await response.json();

        if (result.success) {
            zones.push(result.zone);
            drawRoute(result.zone, result.zone_index);
            updateZoneList();
            clearCurrentSelection();
            selectedStops = [];
            document.getElementById('zoneSetup').classList.add('hidden');
            setStep(1);
            setDrawingStatus('Zone created. Draw another area when ready.');
            alert(`Zone "${zoneName}" created!\n${result.zone.total_stops} stops\n${result.zone.total_distance_km} km`);
        } else {
            alert(`Error: ${result.error || 'Route optimization failed'}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function markerHtml(label, color, size, fontSize) {
    return `<div class="route-marker-dot" style="background:${color};width:${size}px;height:${size}px;font-size:${fontSize}px">${escapeHtml(label)}</div>`;
}

function drawRoute(zone, index) {
    const color = zoneColors[index % zoneColors.length];
    const route = zone.route || [];
    const polygonPath = (zone.polygon || []).map(point => ({ lat: point[0], lng: point[1] }));
    const routePath = ((zone.road_geometry && zone.road_geometry.length > 0)
        ? zone.road_geometry.map(point => ({ lat: point[0], lng: point[1] }))
        : route.map(stop => ({ lat: stop.lat, lng: stop.lon })));

    const polygon = new google.maps.Polygon({
        map,
        paths: polygonPath,
        strokeColor: color,
        strokeOpacity: 1,
        strokeWeight: 2,
        fillColor: color,
        fillOpacity: 0.1
    });

    const line = new google.maps.Polyline({
        map,
        path: routePath,
        strokeColor: color,
        strokeOpacity: 0.9,
        strokeWeight: 4,
        icons: [{
            icon: {
                path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
                fillColor: color,
                fillOpacity: 1,
                strokeColor: '#ffffff',
                strokeWeight: 1,
                scale: 3
            },
            offset: '40px',
            repeat: '100px'
        }]
    });

    const bounds = new google.maps.LatLngBounds();
    polygonPath.forEach(point => bounds.extend(point));
    if (polygonPath.length === 0) routePath.forEach(point => bounds.extend(point));
    const center = bounds.isEmpty() ? map.getCenter() : bounds.getCenter();

    const label = new HtmlMapMarker(
        map,
        center,
        `<div class="zone-label-content" style="background:${color}">${escapeHtml(zone.name)}</div>`,
        'zone-label-overlay'
    );

    const markers = route.map((stop, routeIndex) => {
        const isStart = routeIndex === 0;
        const isEnd = routeIndex === route.length - 1;
        const markerColor = isStart ? '#00c853' : (isEnd ? '#ff1744' : color);
        const markerSize = isStart || isEnd ? 28 : 20;
        const markerLabel = isStart ? 'S' : (isEnd ? 'E' : routeIndex + 1);
        const popupTitle = isStart ? 'START' : (isEnd ? 'END' : `#${routeIndex + 1}`);
        const position = { lat: stop.lat, lng: stop.lon };

        return new HtmlMapMarker(
            map,
            position,
            markerHtml(markerLabel, markerColor, markerSize, isStart || isEnd ? 11 : 10),
            'route-marker-overlay',
            () => {
                infoWindow.setContent(`<b>${popupTitle}: ${escapeHtml(stop.name)}</b><br>${escapeHtml(stop.address)}`);
                infoWindow.setPosition(position);
                infoWindow.open({ map });
            }
        );
    });

    routeLayers.push({ polygon, line, markers, label, bounds });
}

function removeRouteLayer(layer) {
    if (!layer) return;
    layer.polygon.setMap(null);
    layer.line.setMap(null);
    layer.markers.forEach(marker => marker.setMap(null));
    layer.label.setMap(null);
}

function updateZoneList() {
    const container = document.getElementById('zoneList');
    document.getElementById('zoneCount').textContent = zones.length;

    if (zones.length === 0) {
        container.innerHTML = '<p class="empty-state">No zones yet.</p>';
        return;
    }

    container.innerHTML = zones.map((zone, index) => `
        <div class="zone-item" id="zone-item-${index}">
            <div class="name" id="zone-name-${index}" onclick="focusZone(${index})" title="Click to view zone on map">
                <span class="zone-color" style="background:${zoneColors[index % zoneColors.length]}"></span>
                <span id="zone-name-text-${index}">${escapeHtml(zone.name)}</span>
            </div>
            <div class="stats">
                ${zone.total_stops} stops | ${zone.total_distance_km} km
                <div class="zone-actions">
                    <button onclick="event.stopPropagation();startRenameZone(event,${index})" class="rename-btn" title="Rename zone">Rename</button>
                    <button onclick="event.stopPropagation();deleteZone(${index})" class="delete-btn" title="Delete zone">Delete</button>
                </div>
            </div>
        </div>
    `).join('');
}

function focusZone(index) {
    const layer = routeLayers[index];
    if (layer && !layer.bounds.isEmpty()) map.fitBounds(layer.bounds, 50);
}

function startRenameZone(clickEvent, index) {
    clickEvent.stopPropagation();
    const nameContainer = document.getElementById(`zone-name-${index}`);
    const currentName = zones[index].name;

    nameContainer.innerHTML = `
        <input type="text" class="zone-name-input" id="rename-input-${index}" value="${escapeHtml(currentName)}">
        <button id="save-btn-${index}" class="save-rename-btn">Save</button>
    `;

    const input = document.getElementById(`rename-input-${index}`);
    const saveButton = document.getElementById(`save-btn-${index}`);
    input.focus();
    input.select();

    input.onkeypress = event => {
        if (event.key === 'Enter') {
            event.preventDefault();
            saveRenameZone(index, currentName);
        }
    };
    input.onkeydown = event => {
        if (event.key === 'Escape') cancelRenameZone(index, currentName);
    };
    saveButton.onmousedown = event => {
        event.preventDefault();
        isSaving = true;
    };
    saveButton.onclick = event => {
        event.stopPropagation();
        saveRenameZone(index, currentName);
    };
    input.onblur = () => {
        setTimeout(() => {
            if (!isSaving) cancelRenameZone(index, currentName);
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
            zones[index].name = newName;
            updateZoneList();
            redrawAllZones();
        } else {
            alert(`Failed to rename zone: ${result.error || 'Unknown error'}`);
            cancelRenameZone(index, originalName);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
        cancelRenameZone(index, originalName);
    }
    isSaving = false;
}

function cancelRenameZone(index, originalName) {
    const nameContainer = document.getElementById(`zone-name-${index}`);
    if (!nameContainer) return;
    nameContainer.innerHTML = `
        <span class="zone-color" style="background:${zoneColors[index % zoneColors.length]}"></span>
        <span id="zone-name-text-${index}">${escapeHtml(originalName)}</span>
    `;
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
            removeRouteLayer(routeLayers[index]);
            routeLayers.splice(index, 1);
            zones.splice(index, 1);
            redrawAllZones();
            updateZoneList();
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

function redrawAllZones() {
    routeLayers.forEach(removeRouteLayer);
    routeLayers = [];
    zones.forEach((zone, index) => drawRoute(zone, index));
}

function cancelZone() {
    cancelDrawing();
    document.getElementById('zoneSetup').classList.add('hidden');
    setStep(1);
}

async function clearAllZones() {
    if (!confirm('Are you sure you want to delete all zones?')) return;

    try {
        const response = await fetch('/api/clear', { method: 'POST' });
        const result = await response.json();
        if (!result.success) throw new Error(result.error || 'Unable to clear zones');

        routeLayers.forEach(removeRouteLayer);
        routeLayers = [];
        zones = [];
        updateZoneList();
        alert('All zones cleared');
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

function updateZoneRangeInfo(totalStops) {
    const infoDiv = document.getElementById('zoneRangeInfo');
    const targetStops = parseInt(document.getElementById('targetStops').value, 10) || 100;

    if (totalStops < 2) {
        infoDiv.innerHTML = `Need at least <b>2</b> stops. You have <b>${totalStops}</b>.`;
        infoDiv.className = 'zone-range-info error';
        document.getElementById('numZones').disabled = true;
        return;
    }

    const suggested = Math.max(1, Math.round(totalStops / targetStops));
    const maxZones = Math.min(50, totalStops);
    infoDiv.innerHTML = `<b>${totalStops}</b> stops -> <b>${suggested}</b> zones (~${targetStops} each)<br><span>Balanced for fair SMR workload</span>`;
    infoDiv.className = 'zone-range-info';
    document.getElementById('numZones').disabled = false;
    document.getElementById('numZones').min = 1;
    document.getElementById('numZones').max = maxZones;
    document.getElementById('numZones').value = '';
    document.getElementById('numZones').placeholder = `Auto (${suggested})`;
}

document.getElementById('targetStops').addEventListener('change', () => {
    updateZoneRangeInfo(selectedStops.length);
});

async function autoCreateZones() {
    const totalStops = selectedStops.length;
    if (totalStops < 2) {
        alert(`You need at least 2 stops. Currently selected: ${totalStops}`);
        return;
    }

    const targetStops = parseInt(document.getElementById('targetStops').value, 10) || 100;
    let numZones = parseInt(document.getElementById('numZones').value, 10);
    const maxZones = Math.min(50, totalStops);

    if (!numZones || Number.isNaN(numZones)) numZones = Math.max(1, Math.round(totalStops / targetStops));
    if (numZones < 1 || numZones > maxZones) {
        alert(`Please enter a zone count between 1 and ${maxZones}.`);
        return;
    }

    const averagePerZone = Math.round(totalStops / numZones);
    if (!confirm(`Create ${numZones} distance-balanced zones?\n\nTotal stops: ${totalStops}\nAbout ${averagePerZone} stops per zone`)) return;

    const loading = document.getElementById('loading');
    loading.textContent = `Creating ${numZones} distance-balanced zones...`;
    loading.style.display = 'block';

    try {
        const response = await fetch('/api/auto-create-zones', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ stops: selectedStops, num_zones: numZones, target_stops: targetStops })
        });
        const result = await response.json();

        if (result.success) {
            result.zones.forEach(zone => zones.push(zone));
            redrawAllZones();
            updateZoneList();
            clearCurrentSelection();
            selectedStops = [];
            document.getElementById('zoneSetup').classList.add('hidden');
            setStep(1);
            setDrawingStatus('Zones created. Draw another area when ready.');

            const summary = result.zones
                .map(zone => `${zone.name}: ${zone.total_stops} stops, ${zone.total_distance_km} km`)
                .join('\n');
            alert(`Created ${result.zones_created} zones.\n\n${summary}`);
        } else {
            alert(`Error: ${result.error}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        loading.textContent = 'Optimizing route...';
        loading.style.display = 'none';
    }
}

function planningRequestLabel(request) {
    const location = [request.area_name, request.area_code].filter(Boolean).join(' / ');
    return location || 'Selected area';
}

function renderPlanningRequests() {
    const container = document.getElementById('planningRequestList');
    if (!container) return;
    if (!pendingPlanningRequests.length) {
        container.innerHTML = '<p class="empty-state">No pending requests.</p>';
        return;
    }
    container.innerHTML = pendingPlanningRequests.map(request => `
        <div class="planning-request-item">
            <div class="planning-request-title">${escapeHtml(planningRequestLabel(request))}</div>
            <div class="planning-request-meta">Target: ${escapeHtml(request.target_stops || 100)} stops per zone</div>
            <div class="planning-request-actions">
                <button class="request-preview-btn" onclick="previewPlanningRequest('${request.id}')">Preview</button>
                <button class="request-accept-btn" onclick="acceptPlanningRequest('${request.id}')">Accept</button>
                <button class="request-reject-btn" onclick="rejectPlanningRequest('${request.id}')">Reject</button>
            </div>
        </div>
    `).join('');
}

async function loadPlanningRequests() {
    try {
        const response = await fetch('/api/planning-requests?status=pending');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        pendingPlanningRequests = data.requests || [];
        renderPlanningRequests();
    } catch (error) {
        console.error('Unable to load planning requests:', error);
    }
}

function showPlanningRequestArea(request) {
    const polygon = request.preview_polygon || [];
    if (polygon.length < 3) throw new Error('The request has no valid polygon preview');
    const path = polygon.map(point => ({ lat: point[0], lng: point[1] }));

    // Draw the requested area for map context ONLY. The shops to optimize are
    // exactly the ones Market Intelligence sent with the request
    // (request.shop_count) — NOT the backend's local dataset. So we must not run
    // completeSelection() here, which would re-select local stops inside the
    // polygon and show a misleading count.
    clearDrawingDraft();
    clearCurrentSelection();
    drawingMode = null;
    map.setOptions({ draggableCursor: null });
    setDrawingButtons();
    currentSelectionOverlay = new google.maps.Polygon({
        map,
        paths: path,
        strokeColor: '#ff9800',
        strokeOpacity: 1,
        strokeWeight: 2,
        fillColor: '#ff9800',
        fillOpacity: 0.14,
        clickable: false,
    });

    const bounds = new google.maps.LatLngBounds();
    path.forEach(point => bounds.extend(point));
    if (!bounds.isEmpty()) map.fitBounds(bounds, 40);

    // The manual Zone Setup panel is not used for a planning request; keep it
    // hidden so its local-selection counts can't be confused with this request.
    document.getElementById('zoneSetup').classList.add('hidden');
    const shopCount = request.shop_count != null ? request.shop_count : '—';
    setDrawingStatus(
        `SO Planning request loaded: ${planningRequestLabel(request)} — ${shopCount} shops from Market Intelligence`
    );
}

function previewPlanningRequest(requestId) {
    const request = pendingPlanningRequests.find(item => item.id === requestId);
    if (!request) return;
    try {
        showPlanningRequestArea(request);
    } catch (error) {
        alert(`Unable to preview request: ${error.message}`);
    }
}

async function waitForPlanningRequest(requestId) {
    while (true) {
        await new Promise(resolve => window.setTimeout(resolve, 2000));
        const response = await fetch(`/api/planning-requests/${requestId}`);
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || 'Unable to read request status');
        const request = data.request;
        if (request.status === 'completed') return request;
        if (request.status === 'failed' || request.status === 'rejected') {
            throw new Error(request.error || `Request ${request.status}`);
        }
        document.getElementById('loading').textContent = request.status === 'processing'
            ? `Creating ${request.zone_count || ''} zones and optimizing paths...`
            : 'Waiting for route optimization...';
    }
}

async function acceptPlanningRequest(requestId) {
    const request = pendingPlanningRequests.find(item => item.id === requestId);
    if (!request) return;
    if (!confirm(`Accept SO Planning request for ${planningRequestLabel(request)}?`)) return;
    try {
        showPlanningRequestArea(request);
        const loading = document.getElementById('loading');
        loading.textContent = 'Submitting accepted area for optimization...';
        loading.style.display = 'block';
        const response = await fetch(`/api/planning-requests/${requestId}/accept`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}'
        });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || 'Unable to accept request');
        pendingPlanningRequests = pendingPlanningRequests.filter(item => item.id !== requestId);
        renderPlanningRequests();
        const completed = await waitForPlanningRequest(requestId);
        await loadExistingZones();
        clearCurrentSelection();
        selectedStops = [];
        document.getElementById('zoneSetup').classList.add('hidden');
        setStep(1);
        setDrawingStatus(`Completed SO Planning request for ${planningRequestLabel(completed)}.`);
        alert(`Optimization completed.
${completed.zones_created} zones created.
Result: ${completed.result_file}`);
    } catch (error) {
        alert(`Planning request failed: ${error.message}`);
    } finally {
        const loading = document.getElementById('loading');
        loading.textContent = 'Optimizing route...';
        loading.style.display = 'none';
        loadPlanningRequests();
    }
}

async function rejectPlanningRequest(requestId) {
    const request = pendingPlanningRequests.find(item => item.id === requestId);
    if (!request || !confirm(`Reject request for ${planningRequestLabel(request)}?`)) return;
    try {
        const response = await fetch(`/api/planning-requests/${requestId}/reject`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}'
        });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || 'Unable to reject request');
        pendingPlanningRequests = pendingPlanningRequests.filter(item => item.id !== requestId);
        renderPlanningRequests();
    } catch (error) {
        alert(`Unable to reject request: ${error.message}`);
    }
}

async function loadExistingZones() {
    try {
        const response = await fetch('/api/zones');
        const data = await response.json();
        routeLayers.forEach(removeRouteLayer);
        routeLayers = [];
        zones = data.zones || [];
        zones.forEach((zone, index) => drawRoute(zone, index));
        updateZoneList();
    } catch (error) {
        console.error('Unable to load existing zones:', error);
    }
}
