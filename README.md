# SMR Route Optimizer

Create sales/delivery zones and compute an optimized route through the shops in
each zone. A Python backend exposes a JSON API and serves a Google Maps
frontend; zones and route geometry persist in `zones_routes.json`.

## Quick Start

```bash
python -m pip install -r requirements.txt
cp .env.example .env
python smr-po.py
```

Then open `http://localhost:9541`.

## Features

- Google Maps frontend with shop, zone, and route overlays
- Polygon and rectangle selection
- Manual zones with a selectable start shop
- Automatic distance-balanced zones (target stops per zone or explicit count)
- KMeans + nearest-neighbor zoning with convex-hull boundaries
- OR-Tools route ordering (Guided Local Search) with a Haversine fallback
- Google Directions road geometry with per-segment caching
- Numbered route markers with distinct start and end markers
- Rename, delete, focus, and clear zone actions
- Market Intelligence SO Planning integration (approve, zone, and optimize
  submitted shops)
- Local and local-network access; no frontend build step

## Project Structure

```text
SMR PO/
├── backend/
│   ├── __init__.py
│   └── app.py              # config, loading, clustering, routing, APIs
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── smr-po.py               # launcher
├── requirements.txt
├── .env.example
├── product_sense_public_shops_with_area.json   # input dataset
├── zones_routes.json       # saved zones (output)
├── data-json/              # SO Planning results
├── cache/                  # cached Google Maps responses
├── Assets/
└── README.md
```

The separate `market-intelligence/` app is gitignored; `cache/`, `data-json/`,
and `__pycache__/` are generated. The dataset and `data-json/` are gitignored.

## API

The server has no authentication — use it only on a trusted network (or behind
an authenticated reverse proxy).

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Serve the map |
| `GET` | `/api/zones` | Return saved zones |
| `POST` | `/api/optimize` | Optimize and save one zone |
| `POST` | `/api/auto-create-zones` | Generate, optimize, and save zones |
| `POST` | `/api/rename-zone` | Rename a zone by index |
| `POST` | `/api/delete-zone` | Delete a zone by index |
| `POST` | `/api/clear` | Delete all zones |
| `GET` | `/api/planning-requests` | List SO Planning requests (filter by status / area code) |
| `GET` | `/api/planning-requests/{id}` | Return one request and its processing status |
| `POST` | `/api/planning-requests` | Submit an area + shops + `so_count` for approval |
| `POST` | `/api/planning-requests/{id}/accept` | Accept and asynchronously process a request |
| `POST` | `/api/planning-requests/{id}/reject` | Reject a pending request |
| `POST` | `/api/blocking/generate` | Synchronously split submitted shops into `so_count` zones |
| `GET` | `/data-json/{file}` | Return a saved optimized GeoJSON result |

---

Internal use only.
