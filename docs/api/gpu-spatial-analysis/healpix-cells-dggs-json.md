# HEALPix cells and DGGS-JSON zone data

RTS-08. Import from `@luma-contributors/analysis/gpu-spatial-analysis`. HEALPix nested point-to-cell, aggregation, and DGGS-JSON output are implemented.
A5 point-to-cell and aggregation are not: see the last section.

## Point to cell: `GPUHealpixPointToCell`

Longitude/latitude (degrees, float32) to a 64-bit HEALPix cell key, orders 0 to 29, in the layout of `GPUPointToCell` (little-endian `(low, high)` words,
optional `validity`, masked and non-finite rows get key 0). The projection is the closed-form `ang2pix_nest` of Gorski et al. (2005) on the sphere: cells have equal area,
and there is no ellipsoid. `scheme: 'nested'` (default) is the nested pixel number; its cell 0 has key 0, so read `validity`.
`scheme: 'uniq'` adds `4^(order + 1)` (IVOA UNIQ), which is never zero and carries the order.

Reference: healpix-geo 0.4.1 `nested.lonlat_to_healpix` with `ellipsoid='sphere'` (`rts08-healpix-geo.py`, 3000 random points and 23 vertex probes, 14 orders, float32 inputs).
A binary64 TypeScript implementation of the same algorithm (a test file) equals healpix-geo on all random points at all 14 orders, including order 29.
A5 was installed for the same item (`pya5` 1.0.0, Apache-2.0) but no A5 fixture was written.

{/* lens:GPUHealpixPointToCell */}

### float32 accuracy of point-to-cell

The projection is evaluated in float32 (no double-float emulation), so a point within the rounding error of a cell edge can take the neighbouring cell.
Measured against healpix-geo on the fixture's random points (share of keys that differ; each difference was an adjacent cell, at most 1.6 cell widths between centers):

| Order | Cell width | Differing keys |
| --- | --- | --- |
| 0 to 12 | 58.6 to 0.014 degrees | none in 3000 points |
| 14 | 0.0036 degrees (about 400 m) | 0.5% |
| 16 | 0.0009 degrees (100 m) | 1.8% |
| 18 | 0.0002 degrees (25 m) | 6.8% |
| 20 | 0.00006 degrees (6 m) | 23% |
| 24 | 4e-6 degrees | 90%; the cell is still within 1e-3 degrees of the point |
| 29 | 1e-7 degrees | 99.6%; not usable for exact cells |

The share grows about linearly with `nside` (`4 nside 6e-8`). Float32 longitude itself is quantized to 1.5e-5 degrees near 180, so beyond order 14 the key is the cell of the rounded position even before the projection error.
Use orders 12 or coarser for exact cells, 14 for under 1% error. Points exactly on a cell vertex (for example the equator at longitudes that are multiples of 45 degrees) belong to several cells; healpix-geo and this kernel break the tie differently, always to an adjacent cell.
Sphere only: healpix-geo's WGS84 mode applies an authalic latitude conversion that is not done here.

## Aggregation: `GPUHealpixCellAggregation`

Takes the keys of `GPUHealpixPointToCell` (with `order` and `scheme`), an optional `mask` (pass `validity`) and optional `values`, and writes the distinct cells in ascending key order with
`counts` and optional `sums`, `minimums`, `maximums`, plus `status.count`, `requiredCount`, `overflow`, `invalidCount`. A stable lexicographic radix sort over only the key bits the order needs, a run-start inclusive scan and the
sorted segment sums of the other aggregation contributors; no atomics, so sums are reproducible (row order within a cell).

`GPUCellAggregation` itself is not reused: its table pipeline is built on the Quadbin and H3 bit layouts (header and resolution bits around the path), which a HEALPix key does not have, and it was being edited by another item.
Checked against an independent grouping of the read-back keys, including a high-word case (order 24), masks, non-finite values, overflow and cell 0.

{/* lens:GPUHealpixCellAggregation */}

## DGGS-JSON: `createHealpixDGGSJSON`

OGC API - DGGS Part 1 (approved 2025-10-24) zone data in DGGS-JSON, from read-back aggregation output: `dggrs`, `zoneId`, `depths`, `schema` and `values` with `depth`, `shape {count, subZones}` and a dense `data` array (null where a cell is absent), following `dggs-json.json` and the OGC example `1-temperature.json`.
The output is checked in a spec against the required members and shapes of that schema; the schema was not run through a validator, and DGGS-JSON-FG and the binary encodings are not produced.

The OGC HEALPix DGGRS defines its own zone IDs and sub-zone order (root rhombus hex digits, row-major scanlines, WGS84), not the nested index on the sphere. So `zoneId` is `healpix-nested:<order>:<index>` and the data follows nested order; a conforming OGC client would need a mapping that is not provided.
One ancestral zone and one relative depth (at most 10) per document.
Property names are preserved as own keys in both `schema.properties` and `values`, including
`__proto__`. These two dictionaries have null prototypes; serialization preserves every supplied
property name and its aligned data array. Non-finite values become `null`.

## Not done: A5

A5 point-to-cell and aggregation are still not implemented, and nothing A5-related is exported. The earlier "about 1500 lines" estimate was low.
`pya5` 1.0.0 (Apache-2.0) is a 0.44 MB wheel that holds both a Rust backend and the pure-Python reference (`A5_BACKEND=python`). Reading the package, the lookup `lonlat_to_cell` runs through about 3,100 lines of Python:
the authalic latitude series (`projections/authalic.py`), the dodecahedron face projection with quaternions (`projections/dodecahedron.py`, `core/dodecahedron_quaternions.py`, `math/quat.py`), the equal-area warp (`core/warp.py`, `projections/equal_area.py`),
the origin and quintant tables (`core/origin.py`), the pentagon lattice and margin tests (`core/tiling.py`, `core/pentagon.py`, `geometry/pentagon.py`), the Hilbert-like curve with its L-system tables (`lattice/*`, 700 lines), and the 64-bit ID serialization (`core/serialization.py`, 430 lines).
It also has a data-dependent boundary path (`_spherical_to_cell_boundary`: up to 15 re-lookups over the three nearest faces and all five quintants, with a lowest-ID tie-break), which does not map onto one fixed-size WGSL kernel without a bounded-candidate redesign.
A float32 port needs the per-resolution accuracy table before any export, since the curve level reaches 2^29 lattice cells per face edge; this was not measured. The Python wheel needs no environment to read (`python -m pip download pya5==1.0.0 --no-deps`); an oracle fixture would run it with the Python backend.
Next step: a binary64 TypeScript port of the lookup checked against pya5 fixtures, then the float32 kernel.

## Display boundary: `getHealpixCellBoundary`

`getHealpixCellBoundary(nestedKey, order, edgeSegments = 8)` returns a closed longitude/latitude ring with `4 * edgeSegments + 1` positions. It decodes the nested face and interleaved cell coordinates and samples the inverse spherical face projection along each edge. Orders 0–29 and 1–64 segments per edge are accepted. Longitudes follow a continuous branch and can extend beyond ±180°; a polar vertex uses its face meridian. This is sampled display geometry, not an ellipsoidal boundary or OGC ZIRS identifier.

The Kepler catalog's `healpix-cell-aggregation` binding uses this helper to publish GPU cell counts and optional sums/extrema as polygon geometry. The point-to-cell binding retains original point geometry and exact decimal cell ids. Node specs check face corners, seam continuity, validation and interior round trips against the existing CPU indexing twin through order 10; GPU catalog specs check cell ids, aggregate values, polygon publication and overflow refusal.
