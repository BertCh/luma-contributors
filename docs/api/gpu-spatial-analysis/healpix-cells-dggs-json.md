# HEALPix cells and DGGS-JSON zone data

RTS-08. Import from `@luma-contributors/analysis/gpu-spatial-analysis`. HEALPix nested point-to-cell, aggregation, and DGGS-JSON output are implemented.
A5 point-to-cell and aggregation are not: see the last section.

## Point to cell: `GPUHealpixPointToCell`

Longitude/latitude (degrees, float32) to a 64-bit HEALPix cell key, orders 0 to 29, in the layout of `GPUPointToCell` (little-endian `(low, high)` words,
optional `validity`, masked and non-finite rows get key 0). The projection is the closed-form `ang2pix_nest` of Gorski et al. (2005) on the sphere: cells have equal area,
and there is no ellipsoid. `scheme: 'nested'` (default) is the nested pixel number; its cell 0 has key 0, so read `validity`.
`scheme: 'uniq'` adds `4^(order + 1)` (IVOA UNIQ), which is never zero and carries the order.
All constructors, the pure CPU function, and both JSON builders accept only `nested` or `uniq`.
Only an omitted (`undefined`) scheme defaults to nested; RING numbering is unsupported and rejected.

Reference: healpix-geo 0.4.1 `nested.lonlat_to_healpix` with `ellipsoid='sphere'` (`rts08-healpix-geo.py`, 3000 random points and 23 vertex probes, 14 orders, float32 inputs).
A binary64 TypeScript implementation of the same algorithm (a test file) equals healpix-geo on all random points at all 14 orders, including order 29.
A5 was installed for the same item (`pya5` 1.0.0, Apache-2.0) but no A5 fixture was written.

### Graph operation forms

`GPUHealpixPointToCellOperation` names graph values with `positions`, optional `mask`, and
`output.cells` / optional `output.validity`; `rowCount` fixes their lengths. Outputs are
`uint32x2` cells and optional `uint32` validity per position, with no additional packed status word.
For example, `positions: 'lonlat', output: {cells: 'healpixCells', validity: 'healpixValid'}`
maps the contributor ports to logical graph values. It takes the same
`order`, `scheme` and `spatialContext` as the contributor and compiles with
`GPU_CONTRIBUTOR_OPERATION_LOWERING`. `CPUHealpixPointToCellOperation` is the binary64 CPU twin;
`computeHealpixPointToCellOnCPU` is its pure typed-array form. The CPU projection uses binary64
arithmetic, so points near cell edges can differ from the float32 GPU projection. All forms
preserve nested pixel zero and use validity to distinguish invalid rows. Supply the validity
output as the aggregation mask when composing either operation with aggregation.
The pure function returns the named `CPUHealpixPointToCellResult` type, with `cells: Uint32Array`
and `validity: Uint32Array`. At `rowCount: 0`, both named forms produce zero-length `cells`
(and zero-length `validity` when requested), with no phantom row or GPU dispatch. Resolved positions and optional mask lengths must still
match `rowCount`; declaring zero rows for a nonempty vector throws. The pure
function accepts an empty `Float32Array` and returns both empty arrays. The direct
`GPUHealpixPointToCell` contributor requires at least one position row; use the named form
for batches that may be empty.

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
Points a few metres west of a 90 degree meridian at the belt edge (latitude ±41.8103°) take one of the three cells meeting at that face corner (the kernel clamps the belt diagonals there, where float32 rounding would otherwise leave face 4 or reach the invalid face 12).
Longitude wrapping corrects residuals just west of a meridian even when reciprocal division
rounds the number of turns upward (for example `1799.9998779296875°` and `359.9998779296875°`).
Longitude may be any finite value: magnitudes from 2^24 up (where float32 longitudes are integers) are reduced modulo 360 exactly before the projection.
Sphere only: healpix-geo's WGS84 mode applies an authalic latitude conversion that is not done here.

## Aggregation: `GPUHealpixCellAggregation`

Takes the keys of `GPUHealpixPointToCell` (with `order` and `scheme`), an optional `mask` (pass `validity`) and optional `values`, and writes the distinct cells in ascending key order with
`counts` and optional `sums`, `minimums`, `maximums`, plus `status.count`, `requiredCount`, `overflow`, `invalidCount`. A stable lexicographic radix sort over only the key bits the order needs, a run-start inclusive scan and the
sorted segment sums of the other aggregation contributors; no atomics, so sums are reproducible (row order within a cell).

Pass point-to-cell `output.validity` as the aggregation `mask`. For nested keys, invalid rows
and real pixel zero both have key 0: without the mask both are grouped into pixel zero. For UNIQ
keys, zero is impossible and is always excluded, even without a mask. `status.invalidCount`
counts each excluded row once, including zero UNIQ keys, masked rows and non-finite values.
For example, one invalid row and one valid order-0 pixel zero produce keys `[0, 0]` (nested) or
`[0, 4]` (UNIQ), with validity `[0, 1]`. Passing validity retains one cell with count 1 in either
scheme; omitting it retains count 2 for nested and count 1 for UNIQ.
Use `createHealpixDGGSJSONFromReadback` for aggregation export: request and read `count`,
`requiredCount` and `overflow`, and pass them with the packed cell words. It refuses overflow
and reports `requiredCount` so dropped cells cannot become ordinary missing data. It bounds-checks
`count` and joins exactly that prefix; unused tail cells and property values are ignored.
Both JSON builders reject foreign keys with the offending key and scheme in the error.

`GPUCellAggregation` itself is not reused: its table pipeline is built on the Quadbin and H3 bit layouts (header and resolution bits around the path), which a HEALPix key does not have, and it was being edited by another item.
Checked against an independent grouping of the read-back keys, including a high-word case (order 24), masks, non-finite values, overflow and cell 0.

{/* lens:GPUHealpixCellAggregation */}

## DGGS-JSON: sparse cells and checked aggregation readback

OGC API - DGGS Part 1 (approved 2025-10-24) zone data in DGGS-JSON: `dggrs`, `zoneId`, `depths`, `schema` and `values` with `depth`, `shape {count, subZones}` and a dense `data` array (null where a cell is absent), following `dggs-json.json` and the OGC example `1-temperature.json`.
The output is checked in a spec against the required members and shapes of that schema; the schema was not run through a validator, and DGGS-JSON-FG and the binary encodings are not produced.

The OGC HEALPix DGGRS defines its own zone IDs and sub-zone order (root rhombus hex digits, row-major scanlines, WGS84), not the nested index on the sphere. So `zoneId` is `healpix-nested:<order>:<index>` and the data follows nested order; a conforming OGC client would need a mapping that is not provided.
One ancestral zone and one relative depth (at most 10) per document.
Property names are preserved as own keys in both `schema.properties` and `values`, including
`__proto__`. These two dictionaries have null prototypes; serialization preserves every supplied
property name and its aligned data array. Non-finite values become `null`.

`createHealpixDGGSJSON` retains the explicitly sparse-data API. `cells` accepts
`ArrayLike<bigint>`, including `bigint[]` and `BigUint64Array`; property `values` accepts
`ArrayLike<number>` aligned to those cells. No status is inferred for explicit sparse data.

```ts
import {createHealpixDGGSJSONFromReadback} from '@luma-contributors/analysis/gpu-spatial-analysis';

const document = createHealpixDGGSJSONFromReadback({
  order: 1,
  scheme: 'nested',
  ancestorOrder: 0,
  ancestor: 0n,
  output: {
    cells: Uint32Array.of(0, 0, 3, 0), // (low, high), exactly two cells
    status: {count: 2, requiredCount: 2, overflow: 0}
  },
  properties: {count: {values: Uint32Array.of(4, 7)}}
});
// document.values.count[0].data is [4, null, null, 7].
```

Readback `cells` accepts `Uint32Array`, `Uint8Array`, or `ArrayBuffer`; byte views preserve their
byte offset and length and must contain whole pairs of 4-byte words. `wordOrder` is
`'little-endian'` (default: low, high) or `'high-low'` (high, low); bytes within each word use
little-endian encoding. Property `values` must already contain decoded numeric elements
(for example `Uint32Array` counts or `Float32Array` sums). The readback helper rejects
`Uint8Array` and other raw byte views as ambiguous property values: decode the buffer read with
`getPackedUint32Values` or `getPackedFloat32Values` first, using the producer's element format.
For intentionally byte-sized numeric values, pass `Array.from(values)` to make that intent explicit.
The sparse builder continues interpreting all numeric array elements, including `Uint8Array`,
as values. Each status scalar accepts a number, a `Uint32Array`, raw `Uint8Array`
or `ArrayBuffer` containing at least one whole uint32. Missing or truncated words throw.
`count` is the readable number of cells, `requiredCount` is the total distinct cell count, and
`overflow` is nonzero when output capacity dropped cells. Export requires `overflow === 0`
and `count === requiredCount`; increase `cellCapacity` and rerun otherwise.

| Exact host interchange | Encoding and round trip |
| --- | --- |
| H3 hexadecimal text | `const key = BigInt('0x' + '85888bdbfffffff'); key.toString(16)` returns `'85888bdbfffffff'`. |
| S2 token | `const key = BigInt('0x' + '1'.padEnd(16, '0'))` gives `0x1000000000000000n`; `key.toString(16).replace(/0+$/, '')` returns `'1'`. |
| HEALPix nested | At order 1, pixel `3n` is packed `[3, 0]`; `scheme: 'nested'` exports it to descendant slot 3. |
| HEALPix UNIQ | At order 1, `3n + 4n ** 2n` is `19n`, packed `[19, 0]`; `scheme: 'uniq'` subtracts the marker and exports the same slot 3. |

Never convert a cell key through `Number`. Join little-endian words with
`BigInt(low) | (BigInt(high) << 32n)`; the JSON builder's `ancestor` is always a nested bigint,
even when cells use UNIQ. `order` and `ancestorOrder` are dimensionless integer levels (0–29);
`order - ancestorOrder` is at most 10, or 1,048,576 dense descendants per property. Property
`unit` is caller-supplied metadata; cell keys, row counts, validity, masks and capacities are
unitless. Position coordinates and boundary coordinates are in degrees.

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
