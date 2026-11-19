var e={slug:`api/gpu-spatial-analysis/healpix-cells-dggs-json`,title:`HEALPix cells and DGGS-JSON zone data`,html:`<h1 id="healpix-cells-and-dggs-json-zone-data">HEALPix cells and DGGS-JSON zone data</h1>
<p>RTS-08. Import from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>. HEALPix nested point-to-cell, aggregation, and DGGS-JSON output are implemented.
A5 point-to-cell and aggregation are not: see the last section.</p>
<h2 id="point-to-cell-gpuhealpixpointtocell">Point to cell: <code>GPUHealpixPointToCell</code></h2>
<p>Longitude/latitude (degrees, float32) to a 64-bit HEALPix cell key, orders 0 to 29, in the layout of <code>GPUPointToCell</code> (little-endian <code>(low, high)</code> words,
optional <code>validity</code>, masked and non-finite rows get key 0). The projection is the closed-form <code>ang2pix_nest</code> of Gorski et al. (2005) on the sphere: cells have equal area,
and there is no ellipsoid. <code>scheme: 'nested'</code> (default) is the nested pixel number; its cell 0 has key 0, so read <code>validity</code>.
<code>scheme: 'uniq'</code> adds <code>4^(order + 1)</code> (IVOA UNIQ), which is never zero and carries the order.</p>
<p>Reference: healpix-geo 0.4.1 <code>nested.lonlat_to_healpix</code> with <code>ellipsoid='sphere'</code> (<code>rts08-healpix-geo.py</code>, 3000 random points and 23 vertex probes, 14 orders, float32 inputs).
A binary64 TypeScript implementation of the same algorithm (a test file) equals healpix-geo on all random points at all 14 orders, including order 29.
A5 was installed for the same item (<code>pya5</code> 1.0.0, Apache-2.0) but no A5 fixture was written.</p>
<p>{/* lens:GPUHealpixPointToCell */}</p>
<h3 id="float32-accuracy-of-point-to-cell">float32 accuracy of point-to-cell</h3>
<p>The projection is evaluated in float32 (no double-float emulation), so a point within the rounding error of a cell edge can take the neighbouring cell.
Measured against healpix-geo on the fixture's random points (share of keys that differ; each difference was an adjacent cell, at most 1.6 cell widths between centers):</p>
<table>
<thead>
<tr>
<th>Order</th>
<th>Cell width</th>
<th>Differing keys</th>
</tr>
</thead>
<tbody>
<tr>
<td>0 to 12</td>
<td>58.6 to 0.014 degrees</td>
<td>none in 3000 points</td>
</tr>
<tr>
<td>14</td>
<td>0.0036 degrees (about 400 m)</td>
<td>0.5%</td>
</tr>
<tr>
<td>16</td>
<td>0.0009 degrees (100 m)</td>
<td>1.8%</td>
</tr>
<tr>
<td>18</td>
<td>0.0002 degrees (25 m)</td>
<td>6.8%</td>
</tr>
<tr>
<td>20</td>
<td>0.00006 degrees (6 m)</td>
<td>23%</td>
</tr>
<tr>
<td>24</td>
<td>4e-6 degrees</td>
<td>90%; the cell is still within 1e-3 degrees of the point</td>
</tr>
<tr>
<td>29</td>
<td>1e-7 degrees</td>
<td>99.6%; not usable for exact cells</td>
</tr>
</tbody>
</table>
<p>The share grows about linearly with <code>nside</code> (<code>4 nside 6e-8</code>). Float32 longitude itself is quantized to 1.5e-5 degrees near 180, so beyond order 14 the key is the cell of the rounded position even before the projection error.
Use orders 12 or coarser for exact cells, 14 for under 1% error. Points exactly on a cell vertex (for example the equator at longitudes that are multiples of 45 degrees) belong to several cells; healpix-geo and this kernel break the tie differently, always to an adjacent cell.
Sphere only: healpix-geo's WGS84 mode applies an authalic latitude conversion that is not done here.</p>
<h2 id="aggregation-gpuhealpixcellaggregation">Aggregation: <code>GPUHealpixCellAggregation</code></h2>
<p>Takes the keys of <code>GPUHealpixPointToCell</code> (with <code>order</code> and <code>scheme</code>), an optional <code>mask</code> (pass <code>validity</code>) and optional <code>values</code>, and writes the distinct cells in ascending key order with
<code>counts</code> and optional <code>sums</code>, <code>minimums</code>, <code>maximums</code>, plus <code>status.count</code>, <code>requiredCount</code>, <code>overflow</code>, <code>invalidCount</code>. A stable lexicographic radix sort over only the key bits the order needs, a run-start inclusive scan and the
sorted segment sums of the other aggregation contributors; no atomics, so sums are reproducible (row order within a cell).</p>
<p><code>GPUCellAggregation</code> itself is not reused: its table pipeline is built on the Quadbin and H3 bit layouts (header and resolution bits around the path), which a HEALPix key does not have, and it was being edited by another item.
Checked against an independent grouping of the read-back keys, including a high-word case (order 24), masks, non-finite values, overflow and cell 0.</p>
<p>{/* lens:GPUHealpixCellAggregation */}</p>
<h2 id="dggs-json-createhealpixdggsjson">DGGS-JSON: <code>createHealpixDGGSJSON</code></h2>
<p>OGC API - DGGS Part 1 (approved 2025-10-24) zone data in DGGS-JSON, from read-back aggregation output: <code>dggrs</code>, <code>zoneId</code>, <code>depths</code>, <code>schema</code> and <code>values</code> with <code>depth</code>, <code>shape {count, subZones}</code> and a dense <code>data</code> array (null where a cell is absent), following <code>dggs-json.json</code> and the OGC example <code>1-temperature.json</code>.
The output is checked in a spec against the required members and shapes of that schema; the schema was not run through a validator, and DGGS-JSON-FG and the binary encodings are not produced.</p>
<p>The OGC HEALPix DGGRS defines its own zone IDs and sub-zone order (root rhombus hex digits, row-major scanlines, WGS84), not the nested index on the sphere. So <code>zoneId</code> is <code>healpix-nested:&#x3C;order>:&#x3C;index></code> and the data follows nested order; a conforming OGC client would need a mapping that is not provided.
One ancestral zone and one relative depth (at most 10) per document.
Property names are preserved as own keys in both <code>schema.properties</code> and <code>values</code>, including
<code>__proto__</code>. These two dictionaries have null prototypes; serialization preserves every supplied
property name and its aligned data array. Non-finite values become <code>null</code>.</p>
<h2 id="not-done-a5">Not done: A5</h2>
<p>A5 point-to-cell and aggregation are still not implemented, and nothing A5-related is exported. The earlier "about 1500 lines" estimate was low.
<code>pya5</code> 1.0.0 (Apache-2.0) is a 0.44 MB wheel that holds both a Rust backend and the pure-Python reference (<code>A5_BACKEND=python</code>). Reading the package, the lookup <code>lonlat_to_cell</code> runs through about 3,100 lines of Python:
the authalic latitude series (<code>projections/authalic.py</code>), the dodecahedron face projection with quaternions (<code>projections/dodecahedron.py</code>, <code>core/dodecahedron_quaternions.py</code>, <code>math/quat.py</code>), the equal-area warp (<code>core/warp.py</code>, <code>projections/equal_area.py</code>),
the origin and quintant tables (<code>core/origin.py</code>), the pentagon lattice and margin tests (<code>core/tiling.py</code>, <code>core/pentagon.py</code>, <code>geometry/pentagon.py</code>), the Hilbert-like curve with its L-system tables (<code>lattice/*</code>, 700 lines), and the 64-bit ID serialization (<code>core/serialization.py</code>, 430 lines).
It also has a data-dependent boundary path (<code>_spherical_to_cell_boundary</code>: up to 15 re-lookups over the three nearest faces and all five quintants, with a lowest-ID tie-break), which does not map onto one fixed-size WGSL kernel without a bounded-candidate redesign.
A float32 port needs the per-resolution accuracy table before any export, since the curve level reaches 2^29 lattice cells per face edge; this was not measured. The Python wheel needs no environment to read (<code>python -m pip download pya5==1.0.0 --no-deps</code>); an oracle fixture would run it with the Python backend.
Next step: a binary64 TypeScript port of the lookup checked against pya5 fixtures, then the float32 kernel.</p>
<h2 id="display-boundary-gethealpixcellboundary">Display boundary: <code>getHealpixCellBoundary</code></h2>
<p><code>getHealpixCellBoundary(nestedKey, order, edgeSegments = 8)</code> returns a closed longitude/latitude ring with <code>4 * edgeSegments + 1</code> positions. It decodes the nested face and interleaved cell coordinates and samples the inverse spherical face projection along each edge. Orders 0–29 and 1–64 segments per edge are accepted. Longitudes follow a continuous branch and can extend beyond ±180°; a polar vertex uses its face meridian. This is sampled display geometry, not an ellipsoidal boundary or OGC ZIRS identifier.</p>
<p>The Kepler catalog's <code>healpix-cell-aggregation</code> binding uses this helper to publish GPU cell counts and optional sums/extrema as polygon geometry. The point-to-cell binding retains original point geometry and exact decimal cell ids. Node specs check face corners, seam continuity, validation and interior round trips against the existing CPU indexing twin through order 10; GPU catalog specs check cell ids, aggregate values, polygon publication and overflow refusal.</p>`};export{e as default};