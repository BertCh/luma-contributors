# Showcase reference data

Reference and oracle JSON for the spatial-analysis showcase. The files are read by the scripts in
`examples/deck/spatial-analysis-showcase/scripts/`, so do not rename or move them. They are captures
from independent reference computations; regenerate one with the script named beside it, never by
hand.

The planning and research notes of the October 2026 showcase overhaul that used to sit here were
removed on 2026-11-13. They are in git history at `d9d56f0f`.

| File | Read by |
| --- | --- |
| `ais-encounter-reference.json` | `check-harbor-analysis-reference.mjs` |
| `aviation-capacity-flow-reference.json` | `check-aviation-capacity-flow-browser.mjs` |
| `aviation-network-reference.json` | `check-aviation-community-coarsening.mjs`, `check-aviation-network-reference.mjs` |
| `bixi-models-reference.json` | `check-bixi-models-reference.mjs` |
| `chicago-route-reference.json`, `chicago-restriction-reference.json` | `check-chicago-route-reference.mjs` |
| `county-query-reference.json`, `county-expression-reference.json`, `county-nan-sort-reference.json` | `check-county-query-reference.mjs` |
| `cta-geometry-validation.json` | `check-route-affine-reference.mjs`, `check-route-editing-reference.mjs`; written by `validate-cta-geometry.py` |
| `earth-source-oracles.json` | ten `check-*-browser.mjs` scripts (grid alignment, habitat patch perimeter, missing observations, pixel boundaries, spectral display, evidence and texture, surface preparation, tiled observations and their topology); written by the `make-*-oracle.py` scripts and `surface-preparation-reference.py` |
| `earth-spectral-regions-oracle.json` | `check-spectral-regions-browser.mjs`; written by `spectral-regions-reference.mjs` |
| `earth-terrain-boundaries-oracle.json` | `check-terrain-boundaries-browser.mjs`; written by `terrain-boundaries-reference.mjs` |
| `earthquake-standardization-reference.json` | `check-standardization-reference.mjs` |
| `fire-policy-reference.json` | `check-fire-policy-reference.mjs` |
| `fire-topology-reference.json`, `fire-topology-extended-reference.json` | `check-data-topology-reference.mjs` |
| `harbor-polygon-support-reference.json` | `check-harbor-quality-reference.mjs` |
| `movement-network-reference.json` | `check-harbor-quality-reference.mjs`, `check-rail-resilience-reference.mjs` |
| `nyc-taxi-rose-reference.json` | `check-nyc-taxi-rose-reference.mjs`; written by `nyc-taxi-rose-reference.mjs` |
| `osm-group-reference.json`, `osm-temporal-reference.json` | `check-osm-semantics-reference.mjs` |
| `pointcloud-data-validation.json` | `check-pointcloud-reference.mjs` |
| `pointcloud-dem-validation.json` | `check-survey-frame-reference.mjs` |
| `projection-contract-reference.json` | `check-projection-reference.mjs` |
| `rail-support-reference.json` | `check-rail-support-reference.mjs` |
| `retail-network-intensity-reference.json` | `check-retail-network-reference.mjs` |
