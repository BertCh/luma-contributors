# Geometry and observation support: research and implementation review

Date: 2026-10-25. Scope: `points/`, `joins/`, `geometry/`, `cells/`, and `earth/` in
`examples/deck/spatial-analysis-showcase/src/scenes`. This record distinguishes source inspection,
planned behavior, implementation, and browser evidence. The global overhaul ledger owns aggregate
counts; the scope does not claim complete public API coverage.

## Current acceptance register after wave 21

This register supersedes earlier intermediate readiness descriptions below. The final block-only retry passed against wave 17 bundle SHA `56f7715e655d8fe93b59b426b475b87848b7947fb8e4d454faff5353908b4a02`; saved before/after numerical and visual evidence confirms it stayed unchanged. Unchanged suites retain their original tested fingerprints. Sampling additionally passed candidate 21 after the rendering-role order correction described below. Exact per-suite evidence and limits are retained in `geometry-wave15-acceptance.json`, `geometry-wave16-acceptance.json`, `geometry-wave17-acceptance.json` and `geometry-wave21-acceptance.json`.

| Investigation | Accepted evidence | Remaining bounded verification |
| --- | --- | --- |
| Spherical observation support | 5 story steps; 6 independent S2/search cases | No numerical regression identified |
| Boundary forensics | 5 story steps with computed precision/normalization output | No separate numeric harness claimed |
| Global point-pattern calibration | 7 matching-support/tie/alpha cases | Later density/pointwise work belongs to statistics |
| Observation hierarchy | 5 story steps; 9 independent OPTICS cuts | HDBSCAN is not independently oracle-validated |
| Repair and overlay | 5 story steps | No independent overlay geometry oracle claimed |
| Fiji spherical boundaries | 5 story steps; 14 containment/cover/status cases | Final camera shift needs rebuilt visual check; overlay geometry itself lacks an independent oracle |
| Site meshes | 4 independent Euler/boundary/courtyard cases | Other hull/Voronoi options remain source/story evidence |
| Route footpoints | 26 independent distance/query/unit cases | New evidence framing needs rebuilt visual check |
| Earthquake pair structure | 17 marked/constant-intensity cases; 11 additional independent intensity/correction/cache cases, including 5 km reflection evidence | Accepted cases cover selected meaningful combinations, not an exhaustive Cartesian option product |
| Pacific boundary logic | 18 containment/intersection/status cases | Arc vertex geometry not independently oracle-validated; legend clarification needs rebuilt visual check |
| County linework | 6 story steps; all 10 numerical/contract cases passed, including GPU face areas versus GEOS within 1 m²; core topology/ring assembly 29/29 passed | No numerical regression identified |
| Footprint shape lab | 7 story steps; all 27 numerical/contract cases passed, including six courtyard offsets and convex contour area versus GEOS within 0.02 m²; full core refinement 8/8 passed | Skeleton arc geometry is not independently oracle-validated |
| Coverage audit | 8 independent topology/status cases passed on corrected capacity contract | No numerical regression identified |
| Sampling frames | All 28 numerical/contract cases reran successfully on candidate 21; two focused native-role/source-membership/pixel checks passed after rendering-role ordering correction | Direct area coordinate membership covers the default one-wave 200 m design; no exhaustive distribution goodness-of-fit claim |
| Block relationships | All 48 cases passed: bounds, adjacency, visibility, exact tessellation sample counts, total cell area, capacity withholding, duplicate-site withholding and recovery; desktop/mobile tessellation inspected | No independent per-vertex or exact per-owner Voronoi geometry oracle claimed |

The prepared numerical batch totals **132 cases**, but wave 13 did **not** execute or pass all 132: each suite stops at its first failure. Inhomogeneous statistics stopped before its first acceptance because a harness numeric regex captured a trailing sentence period; that parser is corrected in source. A passed source, type, shader or story check is not substituted for an independent numerical comparison. Status/capacity checks are explicitly distinguished from geometry or statistical oracles throughout this record.

## Research question and selected delivery

**What does an observation window actually cover on a sphere?** The selected addition is
`spherical-observation-support`: archived USGS events provide geographic evidence, while a
user-defined cap provides an explicitly hypothetical search window. Compare the circle's spherical
area with S2 exterior and interior covers, change resolution, inspect a cell's area and hierarchy,
and expose required/written output counts rather than silently drawing an incomplete cover.

This is a meaningful gap in the existing H3/Quadbin work. `h3-aggregation` already exposes H3,
Quadbin, Quadkey, Geohash and S2 point indexing; generic polygon cover, compaction and topology are
also present. However its S2 display repeats an additive cell per observation and does not explain
the specialized exact integer hierarchy, spherical measurement, or cap covering contracts. A new
spherical support story therefore adds a distinct question rather than another city aggregation map.

A single Luna implementation agent owns the new scene and helpers. The Astra planner selected the
question, researched primary sources, inspected APIs, and reviews the result. No shared registry or
chapter schema edit is needed because discovery is filename-based.

## Primary-source research and design consequences

| Source inspected | Evidence and consequence |
| --- | --- |
| [S2 hierarchy](https://s2geometry.io/devguide/s2cell_hierarchy.html) | Cells are geodesic quadrilaterals on a unit sphere, recursively divided into four children. Show parent/children as a structural comparison; explain that integer cell identity and float32 point-to-cell boundary precision are separate issues. |
| [S2 cell statistics](https://s2geometry.io/resources/s2cell_statistics.html) | At a fixed level, area varies by location. A level is not a constant square-kilometre denominator. Compute the displayed cell areas rather than substituting the mean area. |
| [Google S2 region-coverer source](https://github.com/google/s2geometry/blob/master/src/s2/s2region_coverer.h) | Interior covers may be empty for nonempty regions. Tightness depends on subdivision and capacity. This motivates a visible empty-interior state and cover-area comparison, but the luma contributor is a bounded uniform enumeration, **not** this adaptive CPU algorithm. |
| [USGS ComCat](https://earthquake.usgs.gov/data/comcat/) | Event parameters come from multiple seismic networks. The bundled archive is observational evidence; density is not exposure, hazard, or a prediction. |
| [USGS event web service](https://earthquake.usgs.gov/fdsnws/event/1/) | Search supports explicit time, magnitude, geographic and event-type filters. Preserve the actual archived window and type selection. |
| [USGS latest-earthquakes description](https://www.usgs.gov/tools/latest-earthquakes) | The public near-real-time map has geographic magnitude thresholds. A map threshold does not establish completeness of this older archived selection, so no global-completeness claim is made. |

Visual direction: a restrained spherical scientific atlas, with warm observed-event marks, cool
index boundaries, and an achromatic radius circle. The cap is an analysis parameter, so it must not
use the same visual encoding as observed earthquakes. Area and count comparisons are explicit;
parent/child outlines explain hierarchy without using an arbitrary multicolor category palette.

## Dataset audit

The existing `poopdeck-earthquakes` payload is reused; no new source is claimed as acquired.
The local manifest declares 77,231 rows, EPSG:4326 positions, magnitude, depth, timestamp and type.
An independent binary inspection found magnitude minimum 4.0 and maximum approximately 8.2.
Event type counts are 77,215 earthquakes, 2 mine collapses, 2 mining explosions, 11 volcanic
eruptions, and 1 landslide. A scene calling its observations earthquakes must select `type == 0`.

The archived source description reports 522,982 upstream events while this exported window has
77,231 rows. The scene must describe the bundled archived selection rather than repeating the
larger number or asserting completeness. The observed timestamp range ends on 2024-12-30; do not
claim full coverage through 2024-12-31. Source event depth is kilometres below the source datum;
this scene uses epicentral surface positions only and must not call cap radius a hypocentral
distance. Sampling, if used for display, must be distinguished from full-catalog calculation.

## API and options audit

The reproducible baseline `node scripts/report-showcase-coverage.mjs --format=json` on this checkout
reported 492 public classes/recipes and 235 with direct runtime call evidence. This is a class/call
inventory, not option reachability, numerical verification, or proof that composed recipe internals
are absent. The following is a scoped inspection rather than an inflated all-features claim.

| Family | Existing working surface found | Gaps and disposition |
| --- | --- | --- |
| Point patterns | `point-patterns` has study-window preparation, city holes, rectangular contrast, Ripley K/L and G/F/J corrections, nearest-neighbor and quadrat indices, GPU random polygon samples, a 39-pattern envelope and duplicate-coordinate jitter. Density, clustering and LOF have separate stories. | Inhomogeneous intensity models, the standalone envelope contributor, continuous/balanced sampling, HDBSCAN and OPTICS remain research candidates. Existing manual simulation envelopes are not evidence that the standalone class is called. |
| Spatial joins | `near-transit`, `nearest-facility`, `observations-by-tract`, `raster-join`, and `topological-relations` cover metric buffers, nearest-feature ties/k/weights/candidates, point-in-polygon aggregation and predicates. | Direct primitive `geospatial` pairwise/spatial-query classes have no named-call evidence, although higher-level operations answer related questions. Avoid duplicating a story just to change the underlying class. A future precision comparison should demonstrate an observable difference. |
| Geometry | `polygon-measures` contrasts plane/sphere/authalic-WGS84 area, ring grouping, validity, compactness and label points. `line-operations`, `noding-and-coverage`, buffers, wildfire shapes and street enclosure already expose substantial geometry workflows. | Precision reduction/normalization, canonical equality, polygon overlay, polygonization, constrained triangulation, Voronoi, concave hulls, spherical overlay/containment and coverage validation remain direct-call gaps. Topology repair should become one cohesive diagnosis–repair–measure story, not disconnected algorithm cards. |
| Cells | H3/Quadbin cover, topology, outlines, compaction, aggregation, pyramid, hierarchy roll-up, classification and period comparison already exist. `GPUCellTableCompare` is composed through a recipe despite no direct named constructor. | **Selected:** specialized S2 cell measurement, exact integer hierarchy, cap interior/exterior cover, explicit units and bounded capacity. **Deferred:** S2 polygon cover, H3 identities and cell-grid paths. |
| Earth | Satellite ground tracks/swaths/constellations, hurricane families/seasons/landfall, drifter playback/model comparison, storm cells/exposure/warning verification. Storm verification already distinguishes spatial test, tolerance, grace, hazard compatibility and miss denominators. | Preserve distinct questions; no story removal is justified by this audit. Global spherical observation support complements rather than duplicates satellite/hurricane narratives. |

### Specialized S2 contract reviewed in local source

- `GPUPointToCell`: S2 point classification remains float32 coordinate computation; keys are
  little-endian low/high words, not JavaScript floating-point identifiers.
- `GPUS2CellMeasures`: one `(area, perimeter)` pair per cell, in steradians and radians; status 0
  valid, 1 invalid. A single named sphere radius converts both units consistently. The scene does
  not label this result ellipsoidal area.
- `GPUS2CellHierarchy`: parent, immediate four children, inclusive leaf range, and four edge
  neighbors. Parent target level is compile-time. Status 1 is invalid key, 2 unavailable level.
- `GPUS2CapCovering`: one unit-vector center and angular radius per cap; uniform level 0–8;
  exterior/interior mode; tolerance 1e-6–1e-2 rad (default 1e-5). Each invocation examines all
  `6 * 4^level` cells. The planned UI bounds levels to 2–5 for responsiveness. Output gives written
  and required counts; status 2 means truncated capacity and must block a completeness claim.
- `GPUCellGeometry`: the same returned cell IDs drive rendered outlines. A globe or dateline-safe
  tessellation must preserve geodesic boundaries; a line joining longitude endpoints across the
  map seam is not acceptable.

## Delivery and verification

Initial S2 implementation now exists in `cells/spherical-observation-support.scene.ts` and
`cells/spherical-observation-support.compute.ts`. Reuses the archived dataset and creates no new
acquisition claim. The parent production WebGPU sweep rendered all five steps with no uncaptured
WebGPU, console, or page errors. Planner screenshot review nevertheless found a missing radius
circle caused by passing nested path pieces as one path. That rendering defect is now corrected;
the corrected screenshots and independent numerical assertions await the next coordinated sweep.
The final Sunda step explicitly explains an empty interior cover and disables the parent outline.

Reachable functionality: levels 2/3/4/5; exterior/interior cap modes; 100–1500 km search radius;
Japan/Türkiye/Andes/Sunda query centers; selected center-cell parent toggle. All 77,215 earthquake
rows enter GPU indexing, area calculation and a separate great-circle `GPUNeighborSearch`; only
2,048 deterministic stride-sample event marks are drawn. GPU cap required/written count and search
overflow are exposed. Capacity reserves the full uniform grid, so this scene does **not** demonstrate
intentional truncation. Hierarchy exposes parent only; children/range/neighbors remain uncovered
specialized variants. Tolerance is the contributor default, not an exposed sensitivity control.

Rendering limitation: cell IDs, areas and corners are computed on the GPU; bounded cell-boundary
readback is then densified along spherical geodesics into scene-local `PathLayer` coordinates. This
is not a claim of direct GPU-resident geometry rendering. Global event-key/area columns are read
once per level/mode variant for deduplication of occupied cells and cached summaries. This is a
larger analytical readback, not a tiny GPU reduction. Cap geometry/scalars are refreshed when input
changes; old request generations cannot publish into a newer selection.

Planner review corrected duplicate node IDs, separate logical imports of the same writable GPU
buffer, discarded hierarchy geometry, stale parameter readback, async graph-build race/destruction
handling, default-only deep-link initialization, non-geodesic four-corner rendering, misleading
fixed-three-decimal density, and undisclosed non-earthquake rows. These are source-review findings,
not a substitute for GPU execution.

Scoped checks observed: `check-story-steps.mjs cells` passed; `check-cartography.mjs --strict cells`
passed. A showcase TypeScript pass reported no cells errors but failed on concurrent changes in
other owned chapters; final integration verification belongs to the parent. No root dependency,
format, build, website or repository test gate is claimed by this domain agent.

Independent reference: `verify-spherical-observation.py` reads the actual binary positions/type
columns and computes double-precision haversine counts and cap areas using the same IUGG mean
radius (6,371,008.8 m). Expected examples: Japan 500 km = 2,342 events and 784,995.1281 km²; Japan
1,000 km = 4,159 and 3,135,148.0594 km²; Sunda 250 km = zero events (the nearest event is more than
60 km outside). The first browser screenshots agree for Japan 700 km = 3,145 events and Sunda
250 km = zero. `scripts/check-spherical-observation-reference.mjs` independently reads the binary
source and computes double-precision counts/areas for six scenarios, then checks displayed counts,
area bracketing, complete cover capacity and browser errors. Its syntax check passes; execution is
coordinated with the parent GPU sweep. Remaining references are CPU values until those assertions run.

## Remaining research, intentionally not counted as delivered

1. **Geometry repair bench:** begin with genuine boundary/OSM defects and keep original geometry,
   repair flags, topology checks and area change visible. Perturbations used to teach failures
   must be explicitly simulated; do not attribute manufactured defects to the source agency.
2. **Exposure-aware point processes:** a valid observational-effort surface is required before
   adding an inhomogeneous null to citizen-science points. A smoothed map of the same events is
   not independent sampling effort and cannot silently stand in for it.
3. **Spherical polygon queries:** use a sourced dateline/polar boundary with declared boundary
   semantics and compare cell candidates to the exact predicate. Cap coverage is not evidence
   that polygon containment or spherical polygon overlay has shipped.

## Subsequent delivery waves authorized during the overhaul

These are sequenced work, not a claim that the preceding audit exhausted the domain. Finish the
current story's correctness review before adding the next workflow.

### Boundary forensics: representation, precision, and actual change

Use a deterministic bounded subset of the existing `chicago-buildings` extract, retain source
building IDs, and state that it represents the Loop/Near West Side window rather than all Chicago.
No source data acquisition is required. Two copies of the same source ring can have reversed order
or a shifted starting vertex while representing the same footprint. Those are **simulated
serialization changes**, not errors attributed to the city. Normalize both and compare their
canonical representation. Then change the coordinate grid, show returned geometry and actual
lattice, and measure what moves, changes area or collapses. Keep source outlines visible under the
result, with vertices as a second visual cue.

Planned surface: `GPUNormalizePolygons`, `GPUCanonicalPolygonEquality`,
`GPUPolygonPrecisionReduction`, relevant `GPUGeometryValidity`/`GPUGeometryMeasures`, and
`GPUMakeValid` only if a clearly labelled simulated crossing makes its contribution visible.
A smaller line-precision companion can expose retained-versus-removed collapse policy where that
choice actually changes output. Limits and overflow counters are part of the result.

Delivered implementation: `geometry/boundary-forensics.scene.ts`, `.compute.ts`, and `-data.ts`
now provide five chapters over twelve fixed source `bldg_id` records. Their 106 stored vertices
have 32,247.0649 m² total area in the declared local metric projection (independently recomputed
from the source GeoJSON and float32 projected coordinates). Two simulated serialization encodings
feed separate GPU normalization nodes and canonical equality. Precision variants at 0.1, 1, 5,
20 and 50 metres return actual polygon coordinates, lattice spacing, ring/polygon counts and
overflow/uncertain status. CPU shoelace calculation summarizes read-back output area and its
difference from source; this is not represented as a GPU area reduction. The final chapter selects
50 metres to reveal coarse-grid changes. Dedicated `GPUMakeValid` repair and line collapse-policy
comparison are still not delivered by this story.

Source review corrected a layer that originally drew the source instead of normalized output,
the interpretation of precision `status.count` (kept polygons, not rings), closing-edge/hole area
accounting, missing actual-lattice and uncertain-counter readback, repeated no-change story steps,
and resource cleanup on asynchronous compile failure. Geometry story checks and owned-file Biome
checks pass; strict cartography reports only existing `polygon-measures` findings. Browser rendering
and numeric equality/precision checks await the parent sweep. A scoped-formatting attempt through
the ocular wrapper ignored its file paths and formatted 28 shared files; the parent was informed,
no shared changes were reverted, and subsequent formatting used direct Biome on owned files.

Primary references:

- [PostGIS normalization](https://postgis.net/docs/ST_Normalize.html): canonical representation
  can reorder vertices, rings and multipart elements. This informs the serialization experiment.
- [PostGIS topological equality](https://postgis.net/docs/ST_Equals.html): equality of point sets is
  different from coordinate-sequence equality. The luma canonical contributor compares normalized
  structure and coordinates, so it must not be marketed as a general topological equality test.
- [PostGIS precision reduction](https://postgis.net/docs/ST_ReducePrecision.html): coarser grids can
  remove small features or split polygons. The experiment should show a coarse-grid failure rather
  than describing precision as free compression.
- [JTS precision reducer](https://locationtech.github.io/jts/javadoc/org/locationtech/jts/operation/overlayng/PrecisionReducer.html): snap-rounding can merge nearby components; separate versus joint
  processing is a modelling choice. The source supports the lesson, not an assertion of identical
  luma and JTS implementation.
- [PostGIS MakeValid](https://postgis.net/docs/manual-3.7/en/ST_MakeValid.html): repair methods and
  collapsed components have different semantics. The local `GPUMakeValid` contract specifically
  uses planar even-odd face reconstruction and must be described on those terms.
- [City of Chicago footprint portal](https://data.cityofchicago.org/Buildings/Building-Footprints-Map/hz9b-7nh8): provenance for the bundled footprint extract; its original manifest retains source
  endpoint, query bounds, download timestamp and checksum.

### Observation-process and geometry-query follow-on

1. An inhomogeneous point-process lab must expose intensity, edge corrections, seeds and null
   simulations. A synthetic exposure scenario laid over real points can teach the sensitivity, but
   must be labelled a scenario rather than measured effort or a fitted causal explanation.
2. Advanced cluster hierarchy (OPTICS/HDBSCAN) should show density varying across a real observation
   field, cluster/noise counts, reachability or stability, and parameter-driven merging. Merely
   recoloring DBSCAN results would not demonstrate the new contributors.
3. A primitive query lab should pair a small set of source points with exact segments/polygons,
   show foot points, distances, boundary semantics and selected query outputs, and compare
   Haversine versus planar measurements at geographic scales where the distinction is visible.
4. Spherical polygon containment/overlay should use a dateline-crossing sourced boundary and
   render its computed output geodesically. The S2 cap demonstration does not substitute for this.

### Statistical and cluster references for the next implementation pass

- [Myllymäki et al., Global envelope tests for spatial processes](https://arxiv.org/abs/1307.0239)
  distinguishes functional/global tests from pointwise envelopes. Extend the existing 39-pattern
  GPU simulation story with `GPUPointPatternEnvelope`, exposing maximum deviation, integrated
  squared deviation and extreme rank. A significance result is not established merely because one
  radius crosses the old pointwise band. Preserve all replicates; an invalid curve invalidates the
  test rather than being silently dropped. With 39 simulations the p-value resolution is 1/40.
  The global test is restricted to matching sampling and study-window supports: city land (or
  clipped city land) with the city null, or the bounding rectangle with its rectangle null. The
  existing deliberately mismatched-support contrast can remain exploratory, but cannot silently
  claim a calibrated conditional fixed-count test when sampled points are discarded by another
  study window. Choosing a radius range after seeing the observations is exploratory analysis.
- [Shaw, Møller and Waagepetersen, Globally intensity-reweighted estimators](https://arxiv.org/abs/2004.00527)
  studies inhomogeneous K and pair correlation under intensity-reweighted stationarity. The local
  `GPUInhomogeneousPointPattern` contract uses an estimated or supplied intensity; estimated
  Epanechnikov bandwidth/self-contribution is a sensitivity choice, not measured effort. A future
  display must show the intensity field and valid-radius/status mask with the K curve.
- [Ankerst et al., OPTICS](https://sigmodrecord.org/1999/06/25/optics-ordering-points-to-identify-the-clustering-structure/)
  makes the ordering/reachability plot the central explanatory object. A varying-density cluster
  story should show this computed plot and extraction controls, rather than only a colorful label
  map. Local `GPUOPTICS` supports DBSCAN extraction and xi hierarchy, predecessor correction,
  minimum samples/cluster size, maximum distance and bounded visits; roots have undefined
  reachability and must appear as gaps, not zero-distance valleys. The implementation is an exact
  bounded small-frame path (maximum 4096 rows), not a claimed large-data speedup.

### Spherical polygon follow-on research

- [S2 basic types](https://s2geometry.io/devguide/basic_types.html) and
  [design overview](https://s2geometry.io/about/overview.html) establish why spherical boundary
  orientation and complementary regions must be explicit, and why shortest great-circle edges
  do not follow constant-latitude lines. The luma API has its own hemisphere restrictions and
  boundary status; no identity with S2 robust predicates is claimed.
- [Natural Earth scale guidance](https://www.naturalearthdata.com/downloads/) describes 1:110m
  data as schematic global/locator geometry. The bundled Fiji multipolygon has 3 shells with
  8, 9 and 5 stored coordinates, spanning the map seam. It is a promising bounded real-data
  spherical-query example, but is unsuitable as a precise operational shoreline. Any analytical
  clipping window would be labelled a scenario; source islands remain genuine generalized data.

### Advanced density hierarchy selection (planned, not yet implemented)

A bounded unsampled frame is preferable to a stride sample when demonstrating density thresholds.
The existing 2023 Chicago nature archive contains exactly 246 bird-category rows in the fixed
Northerly Island rectangle `[-87.617, 41.849, -87.597, 41.87]`. All can fit the exact OPTICS path.
There are 220 unique coordinate pairs; 40 rows occupy coordinates shared by another row, with
maximum multiplicity 5. Retain and disclose those duplicates rather than moving them invisibly.
They are observation records, not counts of individual birds, abundance, or habitat quality.

Planned outputs: actual OPTICS ordering and reachability (roots as gaps), core distances, DBSCAN
distance extraction versus xi extraction, xi intervals, and HDBSCAN excess-of-mass labels and
cluster/core-distance summaries. Relevant parameters are minimum samples (self included), minimum
cluster size, extraction distance, xi and predecessor correction. Expose incomplete OPTICS status;
do not color incomplete extraction as a successful clustering.

Local HDBSCAN inspection corrects a possible conflation of bounds: `GPUOPTICS` supports at most
4096 rows, whereas `GPUHierarchicalDensityClustering` supports up to 2^20 rows and at most 32 core
neighbors. The latter exports labels, optional core distances and cardinality status; its public
surface does not export a condensed tree, cluster probabilities, leaf selection or allow-single-
cluster. The showcase must not invent those outputs or claim its small data frame benchmarks the
maximum capacity.

- [HDBSCAN authors' implementation comparison](https://github.com/scikit-learn-contrib/hdbscan/blob/master/docs/comparing_clustering_algorithms.rst)
  motivates comparing a single density cut with persistent clusters across density levels.
- [HDBSCAN extraction documentation](https://hdbscan.readthedocs.io/en/latest/dbscan_from_hdbscan.html)
  explains excess-of-mass selection; only the matching locally supported variant is planned.
- [iNaturalist Data Quality Assessment](https://help.inaturalist.org/en/support/solutions/articles/151000169936)
  describes quality grade as an observation metadata/identification assessment. It does not supply
  a measurement of local survey effort or make a dense cluster an abundance estimate. The bundled
  archive's own provenance excludes obscured locations and requires reported positional accuracy
  of 250 metres or better; no per-row accuracy radius survives in this payload, so do not draw a
  fabricated uniform uncertainty circle around each record.

### Production verification completed after implementation

The parent production browser sweep passed all five chapters of `spherical-observation-support`
and `boundary-forensics`. Reviewer inspection of the final spherical screenshot confirms the
radius circle now renders correctly, and the Sunda 250 km interior example honestly reports zero
events and zero cover cells. Boundary precision at 50 m shows 47 vertices, 10 polygons, and
40,000 m² against the independently computed source 32,247.0649 m² (+24.04%). The normalization
chapter reports all 12 canonical pairs equal; actual normalized outlines/start markers are visible.
Evidence: `/tmp/luma-showcase-overhaul-20261025/wave4-browser/`.

The durable `scripts/check-spherical-observation-reference.mjs` passed six production scenarios
against independent double-precision haversine counts over all 77,215 earthquake rows. Counts:
Japan 500 km = 2342; Japan 1000 km = 4159; Türkiye 500 km = 819; Andes 1000 km = 4267;
Sunda 250 km = 0; Japan 700 km = 3145. Displayed analytic cap areas agree within rounding;
interior/exterior cell areas bracket the analytic cap area in the expected direction, and written
counts equal required counts. No browser errors. Report:
`/tmp/luma-showcase-overhaul-20261025/spherical-numeric-reference.json`.

The durable `scripts/check-point-pattern-global-test.mjs` passed seven production calibration and
interaction states, requiring all 39 of 39 simulations. Maximum-deviation and integrated-squared
statistics returned conservative p = 0.025; extreme-rank returned p = 0.300 at α = 0.05 and 0.01.
All p values remain on the 1/40 grid, rank tie bounds are explicit, the conservative decision agrees
with α, and units differ correctly (m, m³, rank). Mismatched null support and observed-only jitter
immediately clear calibrated p values; rapid observed-radius/statistic changes recover a current
generation. No page, console, or uncaptured GPU errors. This script checks the real-data
calibration/interaction contract, not an independent reconstruction of statistical formulas.
Report: `/tmp/luma-showcase-overhaul-20261025/point-pattern-global-test.json`.

### Subsequent implementation queue and ownership

`observation-hierarchy` is implemented, pending production GPU review: all 246 archived bird rows,
OPTICS DBSCAN/xi extraction with predecessor correction and minimum-size controls, actual ordered
reachability/core-distance arrays, and HDBSCAN EOM labels. Scoped story/cartography checks and a
clean scoped typecheck passed; later all-scene typecheck failures were concurrent raster edits.
A reviewer requested neutral rendering rather than cluster colors for incomplete/invalid/overflow
results. No claim of full density-clustering capability coverage follows from these two algorithms.

A separately authorized gpt-6-luna CLI implements only `earth/spherical-boundaries*`: real Fiji
Natural Earth shell parts, explicitly synthetic grid/boundary probes and analytic window,
spherical polygon open/closed/complement membership, four Boolean overlay operations in a
hemisphere chart, and adaptive mixed-level S2 polygon exterior/interior support. Metadata is created
last. This is underway, not delivered. Source segmentation at ±180 is retained and disclosed.

The existing Luna implementer next handles planar validity/repair/overlay: simulated crossing
encoding of a real building, validity mask before/after, `GPUMakeValid` even-odd linework repair,
actual reconstructed rings/source IDs/diagnostics, and four-operation overlay plus dissolve against
an explicitly hypothetical work area. It must distinguish kept polygon count from required ring
capacity and expose uncertainty/truncation. Subsequent hull/Delaunay/Voronoi and direct pairwise
geospatial investigations remain queued, as does inhomogeneous intensity-reweighted point-process
analysis. These are not covered by metadata or the already delivered stories.

### Researched design for the next planar mesh investigation (queued)

Use all 82 archived Chicago library locations (`chicago-facilities`, category `library`), whose
coordinate pairs are all distinct. This is a comprehensible bounded public-location geometry,
not a sample of the other facility categories. Its original municipal source is dataset
`x8fc-8rcq`; the archived point counts must not be advertised as a current service directory.
The municipal dataset page was inaccessible to the browser tool on this pass, so the existing
manifest/source provenance remains the evidence for acquisition. Do not infer opening hours.

A coherent geometry question is how a set of sites becomes edges, areas, and nearest-site
boundaries. `GPUDelaunayTessellation` exposes actual triangles, adjacency, circumcenters and
convergence/exhaustion status. `GPUVoronoiDiagram` consumes those triangles and emits bounded
segments with site-pair IDs; this is a vector bisector diagram, not a network catchment or polygon
population allocation. Show the clip rectangle explicitly. `GPUConcaveHull` retains triangles by
maximum circumradius in metres; holes and disconnected components are retained, and isolated
sites/lines have no area boundary. Its local contract is radius-based, not PostGIS's ratio-based
concave hull, and is not a measured service territory. The radius control should make this
model choice visible while preserving every source point.

Primary method references:
- [CGAL 2D Alpha Shapes](https://doc.cgal.org/latest/Alpha_shapes_2/index.html): radius-dependent
  reconstruction, possible disconnection, and the relationship to Delaunay triangulation. CGAL's
  alpha parameter is squared radius; luma's `maximumCircumradius` is a radius, so the UI must not
  silently copy the squared-unit convention.
- [PostGIS ST_DelaunayTriangles](https://postgis.net/docs/ST_DelaunayTriangles.html): a point
  triangulation spans the convex hull and is different from a boundary-constrained polygon mesh.
- [PostGIS ST_VoronoiPolygons](https://postgis.net/docs/ST_VoronoiPolygons.html): nearest-site
  partition and envelope choices. The luma output is segments, not that function's polygon output.

A separate constrained-mesh view can reuse genuine Chicago building shell/hole geometry with
`GPUPolygonTriangulation` versus `GPUConstrainedDelaunayTriangulation`. It must render actual
indices, retain holes, and distinguish triangle validity from Lawson-flip convergence. A bounded
`maximumFlips` control can demonstrate an unfinished but still valid mesh only with explicit
status. This is better placed with footprint geometry than pretending library point sites impose
polygon boundary constraints.

### Researched design for direct pairwise geospatial queries (queued)

Use source CTA station-to-route associations for an aligned point/multipart-linestring comparison.
Each station is paired with a disclosed route; compute `GPUPairwisePointLinestringNearest` and
show the returned footpoint, local part and segment index. Shape parts must retain source
boundaries: never connect disjoint clipped parts implicitly. Compare distance to that particular
segment (`GPUPairwisePointSegmentDistance`), its endpoint (`GPUPairwisePointDistance`), and
spherical point-pair distance (`GPUHaversineDistance`) with explicit metre/kilometre units and
IUGG sphere. None is an entrance or travel distance. Aligned pairs are not an all-pairs spatial join.

`GPUPairwisePointInPolygon` can classify each source query against a declared analytical polygon
row with inside/boundary/outside/uncertain states. `GPUPointSpatialQuery` bounds/radius/polygon
operations should select actual rows from the same source set with visible support. Its radius
is planar and longitude/latitude polygons use linear edges; spherical polygon semantics belong
to the separate Fiji investigation. Raw binary64 variants can be an explicitly synthetic
coordinate-origin encoding experiment; they cannot restore precision already lost in archived
float32 source coordinates.

Primary API comparison reference:
[cuSpatial spatial operations](https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial/).
Local contributor contracts remain authoritative; no claim of identical numerical behavior is
made. These direct-query and mesh designs were plans at this checkpoint; the subsequent status
records implementation and review separately from production verification.

### Wave 8 implementation and independent-reference checkpoint

`observation-hierarchy` is reviewed and frozen for the parent's wave 8 GPU sweep. Incomplete,
invalid, or overflowing extraction results withhold cluster assignments. Exact-coordinate
duplicates remain without jitter. The durable `scripts/check-observation-hierarchy-reference.mjs`
independently implements sequential CPU OPTICS on the original 246 rows for nine k/epsilon
combinations. CPU references passed; the browser comparison awaits the exclusive GPU slot.
At k=8, epsilon=150 m its extraction has six clusters and ten noise records. An OPTICS cut need
not reproduce a fresh DBSCAN traversal's border assignments.

`repair-overlay` is implemented and reviewed for the same sweep. Source building 814558 remains
unchanged; the crossing order is explicitly simulated. The second operand is a hypothetical
local-metre work rectangle, not a source parcel. All four Boolean operations and the source
building collection dissolve expose actual output geometry and source boundary identifiers.
Review fixed polygon-versus-ring counts, repaired validity aggregation, incomplete-output gates,
callback option identity, and readback lifecycle. `GPUMakeValid` uses even-odd interpretation;
repair is not reconstruction of intended topology.

`spherical-boundaries` is implemented and reviewed, with a separate implementation ledger.
Its title is **Fiji on a sphere**. All three published Natural Earth parts are context; only the
eight-vertex Viti Levu shell is analyzed. Delivered: `GPUSphericalPolygonContains` open/closed
boundary and complement variants, `GPUSphericalPolygonOverlay` four operations within a common
hemisphere chart, and `GPUS2PolygonCovering` interior/exterior with min levels 3/5 and max levels
6/8/10 plus actual `GPUS2CellMeasures` and cell geometry. Capacity/traversal failures withhold
successful geometry/area claims. Complement is a containment control, not an overlay rendering
promise. No spherical convex or intrinsic arc overlay is claimed here.

The independent double-precision solid-angle and winding reference in
`scripts/check-spherical-boundaries-reference.mjs` passes its CPU-only check: 445 probes comprise
283 interior, eight exact boundary vertices, and 154 exterior probes. Closed/open membership
should accept 291/283, or 162/154 for the complement. The generalized great-circle shell has area
11,580.368460 km² on the IUGG sphere. Four containment states, four overlay status states, and six
S2 area-bracketing states await GPU execution. Overlay status checks are not an independent
polygon-geometry oracle.

`site-meshes` is actively implemented by the existing Luna worker and undergoing review;
`route-footpoints` is actively implemented by the separately authorized Luna CLI stream. Neither
is production-verified. The atlas agent now owns cell identities/grid paths/measures and line
editing in distinct new files; those are no longer queued in this domain.

### Marked and inhomogeneous analysis: researched next investigation

A bounded real-data investigation can select every earthquake-type row in the declared Japan
longitude/latitude frame 139–142 E, 35–38 N: 788 archived events, 660 with reported depth under
70 km and 128 at least 70 km. None has negative depth; 17 are exactly 10 km. These counts are
computed from the bundled source, not a fresh ComCat query. A local planar rectangular support
must be declared separately from the geographic selection so units and edge correction agree.
No stride sampling or jitter is necessary.

`GPUMarkedPointStatistics` supports product-depth mark correlation, all/cross-category K/L,
annular pair correlation, translation versus no edge correction, and optional positive supplied
intensity. Its directed nonself pair count, n² plug-in normalization, empty-annulus undefined
values, and per-bin status must be visible. Reported depth is not energy or a certainty measure.
The [USGS depth explanation](https://www.usgs.gov/faqs/why-do-so-many-earthquakes-occur-a-depth-10km)
notes that 10 km can be assigned when observations constrain depth poorly. The archive lacks
per-event depth uncertainty and fixed-depth flags, so exact-10 counts cannot identify individual
events as certainly fixed.

`GPUInhomogeneousPointPattern` can expose estimated Epanechnikov intensity, bandwidth and
self-inclusion choices, border/translation K, and translation pair-correlation smoothing choices.
Estimated intensity is a plug-in sensitivity analysis; it does not identify observation effort or
remove aftershock dependence. Supplied constant intensity is an explicit baseline model, not
known earthquake-generating intensity. Primary method documentation is the spatstat authors'
[Kinhom source documentation](https://github.com/spatstat/spatstat.explore/blob/master/man/Kinhom.Rd),
which distinguishes supplied intensity and leave-one-out estimation. Local luma contracts govern
its kernel, normalization, geometry and statuses; full spatstat parity is not claimed.

### Wave 8 GPU acceptance and next active work

Both durable browser checks now pass against the parent's production build on port 5313:

- `observation-hierarchy-reference.json`: all nine OPTICS DBSCAN cuts exactly match independent
  CPU cluster-size multisets and noise counts, retaining all 246 rows. This checks the DBSCAN
  extraction path, not Xi or HDBSCAN mathematics independently.
- `spherical-boundaries-reference.json`: all four membership states match the independent
  reference; all four overlay operations report complete output. Exterior/interior S2 results
  are 4/0 cells and 90,207/0 km² at levels 3–6; 15/3 cells and 20,948/4,190 km² at 3–8;
  95/36 cells and 14,317/9,167 km² at 5–10. Every cover brackets the independently computed
  shell area in the correct direction, reports zero status and measure errors, and has written
  count equal to required count. No browser errors. Two harness-only selector/parser mistakes
  were corrected before the final complete pass.

The parent also accepted every authored hierarchy, repair and spherical story step in production.
After visual review, spherical narrative diagnostics moved to Details, the map frame tightened,
and island labels were added. Those presentation changes preserve the accepted calculations;
updated desktop/mobile screenshots remain pending.

`site-meshes` is reviewed and source-ready, pending GPU validation. Public Delaunay output is
triangle indices/status only: the requested adjacency/circumcentre outputs do not exist in its
public contract. The scene therefore labels circumcentres as CPU-derived from GPU triangle
indices; no direct GPU circumcentre/adjacency coverage is claimed. Cache size is bounded, stale
overlays are hidden, and valid unfinished CDT refinement is distinguished from invalid input.
Source courtyard building 354712 has 24 stored positions over two rings; its camera was corrected
to the actual source location (-87.64917, 41.8837).

`earthquake-pair-structure` is now actively implemented. The earlier 788-row geographic frame
was replaced before implementation by a fixed AEQD rectangular support, origin (140.5, 36.5),
x ±135,000 m and y ±166,000 m. Filtering all type-0 source rows against this projected rectangle
gives 802 records: 673 below 70 km depth, 129 at least 70 km, 17 exactly 10 km, and no coincident
coordinate pairs. This makes the marked rectangular estimator and prepared polygon-window
inhomogeneous estimator share one declared support. It is a planned/modelled support, not an
estimated data envelope or an administrative boundary.

### Remaining coherent investigations (firm queue, not delivered)

**Coverage before dissolve.** Extend the existing county generalization investigation rather than
making a class-per-card gallery. `GPUCoverageValidity` should inspect the original coverage and
the actual shared/independent simplification results, render per-segment unmatched/crossing/gap/
overlap/uncertain flags, and expose gap-width sensitivity in projected metres. It does not check
segments within the same feature and is not a substitute for individual polygon validity.
`GPUCoverageDissolve` can group valid counties by declared state labels, show assembled rings and
source group IDs, and gate the result on validity/capacity. It is an exact matching-edge dissolve,
not arbitrary polygon overlay or automatic overlap repair. Its snap tolerance is a geometric
assumption, and split-touching-rings is an actual output-topology choice. This differs from the
general Boolean dissolve already implemented in `repair-overlay`.

**Sampling frame determines the design.** A single survey-design investigation should compare
finite mapped sites, continuous line length and continuous area, with actual GPU sample positions,
weights and replay provenance. Candidate existing data include the 82 library/92 fire-station
finite frames and the Northerly Island OSM park geometry (source row 14, osmId 2914325, 61 stored
positions); any park boundary used as a linear frame must be named a mapped boundary, not an
accessible path. Finite strata and unequal integer tickets are explicit design choices, not
observed demand or importance. Synthetic retained/legacy sites must be labeled as hypothetical.

`GPUSpatialBalancedSample` is a hierarchical circular-systematic design, not GRTS; its exact
inclusion probabilities and inverse weights should be shown with seed/replicate replay. Invalid
frame rows invalidate the design rather than silently disappearing. `GPUContinuousSpatialSample`
uses a random-shift lattice with line arc-length or area Euclidean spacing, inverse intensity
weights in metres or m², and fixed legacy exclusions. Each replacement wave is an independently
randomized complete design; selective substitution does not inherit a blanket unbiasedness claim.
`GPURandomPointsOnLine` supplies an iid length-uniform baseline, fixed per-line count and stable
prefix under increased count. Capacity/candidate exhaustion must remain visible. Triangulation
used to supply the area frame retains source holes, and its validity is checked before sampling.

Primary motivation: the EPA authors' [spsurvey technical account](https://usepa.github.io/spsurvey/articles/technical.html)
distinguishes the population from its eligible frame and defines weights through inclusion
probabilities. It motivates the story but does not establish algorithmic parity with luma.

**Pacific boundary logic.** One seam-crossing investigation will combine convex spherical
containment/intersection and intrinsic arc overlay. Author-defined great-circle windows, plainly
synthetic, can query the real earthquake archive while Natural Earth supplies geographic context.
`GPUSphericalConvexIntersection` returns actual unit-XYZ vertices, written/required counts and
optional intersection/union steradians and fractions. `GPUSphericalConvexContains` exposes
open/closed and complement behavior. `GPUSphericalArcOverlay` uses a separated reference point
with known operand membership, including a complementary region larger than a hemisphere;
its four operations return assembled rings and reference-membership parity. Render boundaries
and an explicit region interpretation, never fill the wrong hemisphere from a planar ring.
The [S2 basic-types guide](https://s2geometry.io/devguide/basic_types.html) supports the unit-vector
and spherical-loop conceptual explanation; local contributor contracts govern limits/statuses.
These are still plans, and do not add to verified behavior coverage.

### Wave 10 verification and next implementation streams

`site-meshes` passed four production WebGPU numeric scenarios against an independent CPU
monotone-chain hull/Euler count: 82 library sites produce 150 Delaunay triangles; ratio 1 returns
12 convex-boundary edges and all 150 triangles. The real courtyard's 22 distinct vertices and one
hole produce 22 Earcut and constrained triangles. A one-flip budget preserves a valid 22-triangle
mesh while correctly reporting nonconvergence. Browser console/page errors were empty. Durable
script: `scripts/check-site-meshes-reference.mjs`; report:
`/tmp/luma-showcase-overhaul-20261025/site-meshes-reference.json`.

`route-footpoints` did **not** pass its first runtime attempt: graph construction rejected two
`GPUPairwisePointDistance` instances with the same default ID. Explicit unique IDs now distinguish
the route-reference and returned-footpoint checks. Direct owned-file Biome passed; production
must rebuild before its 26 numeric scenarios can run. The harness now captures visible scene
initialization failures immediately, rather than waiting solely on a readiness flag. No route
numeric acceptance is claimed yet.

`earthquake-pair-structure` is source-reviewed, with terminal-radius GPU readouts added for both
marked and inhomogeneous estimators. The independent CPU reference computes all 16 combinations
of mask, category pair and edge correction, at all 16 radii. The browser harness will compare
terminal K, L-minus-r, annular pair correlation, product-mark correlation, directed counts, and
the supplied constant-intensity translation identity. It remains GPU-unverified.

`pacific-boundary-logic` is source-reviewed. Fixed results encode once, then reader callbacks
publish cached output for controls. The two windows are synthetic **great-circle** regions;
listed corner latitudes do not imply parallel-latitude edges. Eight containment variants, two
convex intersection capacities and eight arc variants are reachable. Direct GPU unit-vector
boundaries drive the map. Archive membership for Boolean combinations is explicitly CPU Boolean
logic over actual GPU containment arrays, not an invented arc-overlay point-query output.
Independent double-precision edge-plane tests find A=12,223 and B=13,243 real earthquake rows,
intersection=1,807, union=23,659; no archive row lies within the declared 1e-6 boundary band.
Spherical clipping yields four intersection vertices and 0.5949614348310951 sr
(24,149,337.7615 km² with the declared mean radius). Complement variants and reference parity
have corresponding independent expectations. The durable 18-case harness is prepared;
production GPU acceptance remains pending.

Current disjoint streams: the app Luna extends `noding-and-coverage` with coverage validity and
exact-edge dissolve; the completed Pacific CLI stream is reused for new `points/sampling-frames*`;
a second parent-authorized CLI owns new `geometry/footprint-shape-lab*`. Sampling combines finite
facility frames with the mapped Northerly Island boundary and triangulated park area, as planned
above. The shape investigation distinguishes straight-skeleton wavefront traces, maximum interior
circle, minimum clearance and signed buffers/offset components. `GPUOffsetExpansion` is an
indexing operation (offsets to owners/local indices), **not** geometric expansion; it is planned
as genuine ring/polygon provenance composition. `GPUOffsetCurveComponents` composes line noding.
Direct polygonization and shared-path behavior still require reconciliation after coverage work.
KDE and pointwise `GPURipleyEnvelope` transferred to the statistics planner; the already accepted
global PointPatternEnvelope invariants and durable seven-case harness were explicitly handed off.
SinusoidalProjection is assigned to the data planner and is not credited by local-frame stories.

Primary shape references: [CGAL straight skeletons](https://doc.cgal.org/6.2/Straight_skeleton_2/index.html),
[Shapely maximum inscribed circle](https://shapely.readthedocs.io/en/stable/reference/shapely.maximum_inscribed_circle.html),
[minimum clearance](https://shapely.readthedocs.io/en/stable/reference/shapely.minimum_clearance.html), and
[offset curves](https://shapely.readthedocs.io/en/stable/reference/shapely.offset_curve.html). These
support the conceptual distinctions; luma contributor contracts govern implementation limits.
Minimum clearance is a validity-sensitivity distance, not an available passage width. Neither
skeletons nor mapped offsets imply building-code setbacks, roof heights or accessible paths.

## Wave 11 independent acceptance and remaining bounded work

The production browser at port 5313 was held frozen while these four suites ran. All browsers were closed before the GPU slot was released to statistics.

- `check-route-footpoints-reference.mjs`: **26 cases passed**, no browser errors. Independent CPU nearest-segment/footpoint comparisons covered Blue, Red, Brown, and Green route subsets; radius/bounds/polygon queries and metres/kilometres were exercised. Green subset station 0 is 3,803.076 m from the archived direction-0 route geometry, confirmed independently. This remains a geometry-offset result, not entrance access or a diagnosis of source error.
- `check-earthquake-pair-reference.mjs`: **17 cases passed**, no browser errors. Sixteen depth-pair × edge-correction × exact-10-km filtering combinations matched CPU terminal K, L−r, pair correlation, mark correlation, and directed-pair counts. Constant-intensity inhomogeneous translation output matched the corresponding independently summed homogeneous statistic. Other inhomogeneous kernel/border variants remain source-ready pending broader interaction checks; this run does not claim their numerical acceptance.
- `check-pacific-boundary-reference.mjs`: **18 cases passed**, no browser errors. Eight convex containment variants matched independent double-precision classifications, including all 77,215 archive events and separate corner probes; bounded/full convex intersection capacity cases reported four required vertices and agreed with independently clipped spherical area; eight arc operation/complement cases returned status zero, correct reference parity, and correct CPU Boolean combinations of GPU containment arrays. Arc vertex geometry itself has not been independently validated. The first harness run stopped because `Number('1,807')` is NaN; removing thousands separators fixed the harness without changing the production implementation.
- `check-coverage-audit-reference.mjs`: **not accepted**. The scene reached ready and selected the new coverage view, but the first validity readout did not arrive within 120 seconds. No console errors were emitted. The initial failure report omitted hidden readouts; the durable harness now preserves them and recognizes asynchronous construction-failure text. Source now also clears outputs and calls `ctx.setStatus` when asynchronous coverage construction fails, so a hidden diagnostic cannot leave the public view appearing ready. A short next-slot diagnostic is required before deciding whether compilation, readback, or the test path caused the timeout.

Reports: `/tmp/luma-showcase-overhaul-20261025/{route-footpoints,earthquake-pair,pacific-boundary,coverage-audit}-reference-wave11.json`. These browser reports are transient evidence; durable scripts live under the showcase `scripts/` directory.

County linework is now source-ready, not GPU-accepted. Independent GEOS 3.13.1 on the same Albers float32 inputs confirms Curry County (41015, source row 2182, 77 stored vertices) and Josephine County (41033, source row 2191, 58 vertices) have 46 oppositely directed shared source segments totaling 116,339.2583 m. Reversing Josephine reverses shared-path direction without changing geometry. Union-noded linework has 87 atomic segments and two bounded faces; deliberately omitting Curry's first exterior edge leaves 86 unique atomic segments and one bounded face, with two GEOS dangle polylines. `GPULineNoding` preserves source duplicates, so its segment count must not be conflated with GEOS union-noding's 87. Durable reference/browser scripts are being prepared.

The footprint-shape corrective review is source-stable: unique offset-expansion IDs and producer order, cached-result generation guards, bounded cache/resource disposal, one-time execution/readback, signed curve sides, complete buffer topology/capacity gates, and explicit local projection/source caveats. On-device numerical acceptance is still pending. The sampling investigation is in corrective review for resident replay, finite gather write races, invalid-output gating, actual source-row weighting, control applicability, and explicit legacy support. Neither is counted as GPU verified.

The same application Luna now owns the final bounded `block-relationships` inquiry. It will combine bounds candidates, shared mapped-boundary adjacency, a GPU-shrunk boundary-sampled tessellation, and synthetic observer visibility over three archived West Erie Street footprints. The researched source IDs are 342227, 815550, and 815551; physical party walls, ownership, access, roof heights, and building-code setbacks are not inferred. The shrink, enclosure, and observers are explicit analyst assumptions. Conceptual sources are [momepy tessellation guidance](https://docs.momepy.org/v0.11.0/user_guide/elements/tessellation.html), [morphological tessellation API](https://docs.momepy.org/v0.10.0/api/momepy.morphological_tessellation.html), [building adjacency](https://docs.momepy.org/en/stable/generated/momepy.BuildingAdjacency.html), and [Turner et al. (2001), From isovists to visibility graphs](https://discovery.ucl.ac.uk/id/eprint/160/). These motivate methods; the contributor contracts define the actual implementation.

## Wave 12 source review and prepared numerical checks

The parent holds the GPU for the full story/phone sweep; no geometry browser was started during it. `county-linework` (10 cases), `footprint-shape-lab` (27 cases, including bounded/resolved wavefront status), and `sampling-frames` (28 cases) have durable numerical/interaction scripts prepared. These counts are planned checks, not acceptance results. County linework now reserves 512 intersection pairs after GEOS independently counted 275 raw intersecting unordered pairs. The footprint reference uses the same declared shell-centroid local projection as the uploaded input. Sampling checks actual source-row parity (41 library rows and 46 fire rows per parity), per-category allocation, reciprocal inclusion weights, continuous-wave/legacy roles, capacity withholding, and IID seed-prefix behavior.

The parent's frozen wave 12b story sweep exposed a sampling graph-construction error: the same physical selected-ID buffer was imported under separate input and output names. Corrective source now uses one cached graph importer per graph across finite, continuous-line, IID-line and continuous-area branches, including producer/consumer provenance buffers. The separate park triangulation graph has its own importer. Direct Biome and scoped showcase typecheck passed after the fix; runtime acceptance still requires a rebuilt production bundle.

`block-relationships` is source-stable after review of pair-matrix copy extent, shared-path completeness, source-feature ownership, asynchronous resource cleanup, public failure status, and strict occupied-prefix geometry validation. Bounds candidates expose intersects/within/contains. Tessellation exposes zero or 0.4 m hypothetical inset, 4/8/16 m boundary spacing, and 16/128 site capacity. Visibility exposes 25/55/100 m distance and 8/120 displayed-edge capacity, with full dense-graph metrics kept separate from a truncated drawing. All 13 recipe outputs are reachable through the metric selector: degree, reachability, mean hop distance, closeness, raw/normalized choice, relative/real relative asymmetry, Hillier–Hanson integration, pyramid, Teklenburg, Penn, and maximum depth. These are graph measures on synthetic outside observers, not measured pedestrian accessibility.

The block CPU reference uses GEOS 3.13.1 for source contacts/negative buffers and independent BFS/Brandes graph calculations for visibility. Shared mapped lengths are 20.7630 m (342227–815550), 17.7213 m (342227–815551), and zero (815550–815551). The synthetic enclosure is 4,304.6737 m². Review corrected the CPU Hillier–Hanson reference to include its diamond-graph normalization; inverse relative asymmetry alone is not that statistic. The 48-case browser harness is prepared but unrun. It deliberately does not equate GEOS boundary-site counts or per-owner Voronoi areas with GPU outputs when retained collinear vertices may change the sampling locations; it compares complete counts, enclosure-area conservation and the independently defined graph/contact results instead.

An additional finite earthquake inhomogeneous oracle covers representative intensity bandwidths/self-inclusion, border/translation correction and pair-correlation kernel/divisor/reflection settings. The 80 km readout alone cannot verify reflection at zero with bandwidths of 10/25 km; actual GPU values at 5 km were appended to the existing readout so the forthcoming check can observe that effect. Computation is unchanged. Prior 17-case marked/constant-intensity acceptance remains distinct from this pending broader validation.

Visual review tightened the earthquake fixed-window camera from zoom 4.8 to 6.4, and removed tiny explicit analysis-distance ticks from the route and sampling scale bars because they visually resembled a conflicting second scale. Responsive metric scale bars remain. These display changes await the next production build/screenshot.

The frozen wave 12b sweep resolved the previous coverage timeout into a concrete constructor failure: `coverage-dissolve-assembly output.polygons.positions must match output.positions length`. The polygon-position allocation/view was 3,511 vertices while the ring-position output reserved 7,022 (source vertices plus ring closure capacity). Both polygon allocation and view now reserve the same 7,022 vertices; topology/status gates are unchanged. This corrects the graph contract without changing the independent expected dissolved topology. Runtime retry is still pending.

Further screenshot review found that a regional CTA overview hid a 7.9 m station offset. The route scene now has evidence framing: opening/multipart retain the overview; nearest and interpretation fit the actual selected station and GPU footpoint within a minimum 40 m frame; reference fits the actual paired source vertex; polygon/query stages fit their declared support. `ctx.fitBounds` accounts for the free map area, with 56 px padding and maximum zoom 19.5. Selection changes follow after a current GPU readback, rather than moving every frame. County linework framing was tightened from zoom 7.1 to 8.0. Footprint clearance now marks the actual GPU witness endpoints, so a 0.152 m segment is locatable without exaggerating its geometric length.

`check-geometry-observation-batch.mjs` serializes the six remaining numerical suites and binds reports to served `build-evidence.json` / `bundleSha256`, checking that the bundle remains unchanged after each suite and at the end. The additional inhomogeneous harness now has 11 planned cases, including cache return, and compares 5 km and 80 km outputs; the 5 km comparison is necessary for zero-reflection sensitivity. Syntax and CPU-oracle runs passed. Browser acceptance remains pending the parent's rebuilt bundle and GPU handoff.

The independent Hillier–Hanson formula was cross-checked against the developers' [PST documentation, section 6.2, page 10](https://research.chalmers.se/publication/514499/file/514499_Fulltext.pdf), which defines mean topological depth, relative asymmetry, the diamond normalizer, real relative asymmetry, and its reciprocal integration. The showcase applies these graph definitions to the reachable synthetic observer component; it does not inherit PST's street-network interpretation or claim empirical accessibility.

Root's expanded WGSL preflight found three generated block copy bindings named reserved `target`, which the earlier scoped story/type checks could not detect. All three bindings and their shader offsets now use `copiedOutput`. The **full** checker then passed across 33,801 templates (`integration-wgsl-wave12c.log`); this is stronger evidence than the earlier source-readiness statements and is explicitly required before the next runtime attempt.


## Wave 13 evidence and bounded fixes

Reports are saved under `/tmp/luma-showcase-overhaul-20261025/geometry-wave13-numerics` (coverage, county, footprint) and `geometry-wave13-numerics-retry` (sampling, block, inhomogeneous). Two earlier wrapper hash fetches hit Node keep-alive `ECONNRESET`; fresh requests served the identical bundle. The harness now closes its evidence HTTP connection, and the final retry wrote matching before/after evidence. No production rebuild occurred during the GPU slot; all browsers closed before release.

- Coverage passed all 8 cases after the ring/position capacity correction.
- County first case was partial: 133 complete noded input segments, 0 bounded faces, 133 open chains/cut edges, 0/77 written/required rings. The caller used 0.001 m endpoint tolerance on absolute Albers coordinates. SegmentRingAssembly recommends at least four float32 coordinate ULPs; a local rebase and physically explicit tolerance are under review. A shortest-edge bound alone does not prove distinct vertices cannot merge.
- Footprint passed 16 cases, then withheld courtyard left-round selected components with noding status 1. That combined flag can mean capacity **or** uncertain predicates. No capacity cause or selected geometry validity is asserted. `?geometryDiagnostics=1` adds a duplicate direct noding pass only for diagnosis, exposing required count, candidate overflow and uncertainty separately.
- Sampling passed 18 cases; first 200 m area/no-legacy case returned flags 1 with zero output. Triangulation count/index handling matches the public contract. An opt-in diagnostic analyzes already-required GPU triangle readback for invalid indices, finite/bounds checks and float32-emulated determinants. No triangles are removed or repaired to force acceptance.
- Block bounds relation metadata was marked compile-time, but four compile controls lacked `setOption` rebuild handling. A serialized GraphSwap wrapper now rebuilds the core for bounds relation, inset, visibility distance and compact edge capacity, guards stale callbacks, clears pending geometry/readouts, destroys old resources, and replays concurrent parameter changes. Rebuilt numeric and A–B–A evidence remain required.
- Observation hierarchy legends now list only actual GPU-returned cluster labels and observed noise, with a pending/withheld legend while recompiling; this is a visual correctness change awaiting the rebuilt view.
- Inhomogeneous statistic parsing now uses a complete decimal/exponent grammar, so sentence-final periods are excluded. The actual first captured constant-translation values were finite; no additional statistical case is marked passed from the failed harness.

`check-geometry-contract-diagnostics.mjs` is explicitly a two-scene bounded diagnostic capture, **not** a numerical acceptance test. It saves screenshots/readouts and before/after bundle evidence.


County rebase review: independent Python/GEOS reference rerun passed using `/tmp/ahk-showcase-venv/bin/python`; local origin is `[-2234877, 888870.1875]` m, maximum coordinate magnitude 65,554.0625 m, four-ULP tolerance **0.03125 m**, minimum distinct vertex separation **52.524399 m**. The oracle asserts adding the origin exactly restores every original globally quantized float32 source vertex. GEOS shared length, 46 shared segments, direction reversals and 2/2/1 polygonized faces remain identical. This establishes source preservation and a tolerance far below distinct source-vertex separation, not yet GPU polygonization correctness. Matching tolerance is disclosed in the public readout.


## Wave 14 bounded runtime and contributor diagnosis

Served bundle **`872af62d507f3cae0bed2749a0b9f8c7736739cf2d99cab71ba8658bf5e817ae`** was unchanged before/after both `/tmp/luma-showcase-overhaul-20261025/geometry-wave14-diagnostics` and `geometry-wave14-bounded`. All browsers closed before the shared slot was released.

- **Inhomogeneous 11/11 passed**, errors empty: ten selected intensity/correction/kernel/divisor/reflection combinations compare actual GPU K, L−r and g at both 5 km and 80 km against independent double-precision ordered-pair sums; clipped kernel mass uses independent 128-interval Simpson quadrature. Tolerance is max(0.0001, 0.15% of expected magnitude). Returning to the original constant-translation combination verifies cache behavior. Reflection is tested at 5 km, where it changes the kernel result. No invalid/coincident-input acceptance is inferred because this source window has no coincident pairs.
- **Block 3/3 A–B–A bounds smoke passed** at an 8 m half extent against independent AABB reference counts. It verifies the repaired compile-control lifecycle. This does not replace the full 48-case suite.
- **County polygonization still failed identically after local rebasing:** 0 faces, 133 open/cut edges. The tolerance hypothesis is therefore disproven as the root cause. The exact-preserving rebase remains numerically useful, but is not presented as the solution.
- **Footprint diagnosis is now specific:** raw curve noding wrote 64 segments, required 203, overflow 1, candidate overflow 0, uncertainty 0. The output segment capacity was undersized. The source correction uses the contributor's documented conservative bound of input raw rows plus eight vertices per intersection event; all completeness and uncertainty gates remain. No selected-offset geometry is accepted until retry.
- **Sampling diagnosis:** actual GPU triangle readback reports 58 triangles, no invalid indices/nonfinite/out-of-bounds coordinates, but all 58 gathered triangles have zero determinant under f32 emulation. Independent CPU Earcut on the identical source/projection/f32 coordinates gives 58 nondegenerate triangles, minimum area 22.1011 m². No triangles are dropped or replaced. Opt-in diagnostics expose raw indices and gathered/source coordinates. The subsequent caller audit identified the byte-decoding defect described below; no triangulation-method change is retained.

The county source trace found a documented contributor defect candidate: `GPUPolygonize` preserves duplicate atomic shared boundaries, expands them into paired half-edges, but `GPUSegmentRingAssembly` skips only each row's exact twin. Another duplicate reverse edge remains a preferred geometric U-turn (turn key +2). A CPU mirror of its match and injective-successor passes on two adjacent unit squares gives no positive faces with duplicated shared boundaries and two positive faces with unique undirected edges. The parent authorized a narrow core correction and GPU regression, preserving grouping, original provenance and diagnostics; source preprocessing is not being used to hide the documented supported case.


Sampling caller correction after wave 14: `GPUReadbackRing.ticket.read()` returns `Uint8Array`, while `new Uint32Array(bytes.slice(...))` converts each byte into a separate integer. The triangle adapter mistakenly used that conversion for indices, count and validity; small counts appeared plausible while triangle indices were interspersed with zero bytes. All three now reinterpret an owned byte slice's `.buffer`. The existing SummaryReader callbacks elsewhere in this lane are explicitly `ArrayBuffer` and remain correct; the direct ocean readback already converts to an ArrayBuffer. No source triangle is changed, removed or replaced, and no triangulation contributor change is needed. The tentative Earcut-only diagnostic flag was removed before any runtime test. Rebuilt full sampling acceptance remains required.

Core polygonization source correction, reviewed after wave 14: paired-opposite ring traversal now chooses the lowest-index owner for exactly equal stored directed endpoints within each source group using the existing sorted spatial hash. Exact equality makes owner selection transitive; ordinary ring connectivity retains its declared tolerance. Matching ignores non-owner duplicates and all geometrically reversed continuations. Original rows, source IDs and diagnostics remain present; duplicate rows receive their owner's ring/vertex/flags, while only owners write geometry. Propagation preserves originally cancelled rows so the final open-count cancellation subtraction occurs once. This extra work is confined to `pairedOpposites`; ordinary outline assembly is unchanged.

The focused GPU regression now exercises both adjacent-square input orders, reversed shared-edge orientation, a corner-attached spur (eight used boundary rows and one dangle), and two identical squares in distinct groups. It asserts two unit-area bounded faces, actual bounds and group IDs, not face count alone. These tests are source-ready, **not executed or accepted yet**. Scoped Biome and diff checks passed. Showcase typecheck passed and full WGSL identifier/nonfinite preflight scanned 33,905 templates without findings. Parent owns the required package build and focused headless GPU test before the full geometry numerical retry.

Final core review additionally corrected two edge cases before integration. The match pass uses atomic flag loads/stores throughout, avoiding cross-invocation reads racing with flag writes without adding a ninth grouped storage binding. Duplicate open-count correction uses the traced owner cycle leader rather than its written ring slot, so capacity truncation cannot misclassify closed edges as open. A valid paired-half-edge duplicate-square regression expects two required cycles, zero written at three-vertex capacity, overflow 1 and open count 0. The CPU binding-limit test now constructs both paired modes with groups/cancellation. Final scoped typecheck and full WGSL preflight passed again; GPU tests remain pending the allocated wave 15 slot.

## Wave 15 numerical results and remaining caller corrections

Production bundle **`5c13c77a7d5dd844146b724ae5c70d49272c63080f028da1228e99f5e22fac65`** stayed unchanged throughout the full focused core GPU tests and six-suite browser batch. Reports and before/after evidence are in `/tmp/luma-showcase-overhaul-20261025/geometry-wave15-numerics`; the block parser retry is in `geometry-wave15-block-retry`. All browsers closed before explicit slot release.

- **Core topology and ring assembly: 29/29 GPU tests passed** across both full focused files. This includes duplicate shared boundaries in both input orders/orientations, spur diagnostics, identical geometry in separate groups, and capacity-independent closed/open classification.
- **Coverage 8/8, sampling 28/28, and inhomogeneous statistics 11/11 passed.** Sampling now includes all area and IID cases previously unreachable after the byte-decoding bug; source triangles are unchanged.
- **County first case now computes the independently expected two bounded faces**, with zero open/cut/dangle/unused edges and all overflow/uncertainty flags zero. The caller still wrongly labeled this partial by comparing two bounded output rings to the total assembly requirement of three rings (including the unbounded face). Source correction removes that invalid equality while retaining all overflow/candidate/uncertainty/noding gates. Its next oracle also compares actual GPU-readback face areas against GEOS to an absolute 1 m² tolerance, not merely face counts. The ten-case suite is not yet accepted.
- **Footprint 25 cases passed**, including the six formerly withheld courtyard offset combinations. The next case, resolved convex wavefront, returned unsupported flag 1 for a genuinely strictly convex four-corner source shell. No convex output was accepted. One subsequent case was not reached.
- **Block 41 cases passed after a harness-only decimal regex correction**, covering bounds, adjacency and visibility metric/capacity variants. The first inset tessellation case was withheld because the caller incorrectly treated newly generated dissolved output feature IDs as archived source IDs. The remaining tessellation cases were not reached. The correction buffers each of three source buildings independently and retains its known owner explicitly, avoiding guessed output order or a fourth redundant buffer computation.

A minimal device arithmetic diagnostic reproduced the convex rejection using the exact uploaded float32 source corners. Its own-edge endpoint determinants were −6.43e−8, −2.58e−7 and −2.93e−7 m² despite being mathematically zero; all actual corner determinants exceeded 5,548 m² with the correct winding sign. Original half-plane checks returned unsupported=1; excluding only the two defining endpoints returned unsupported=0. Evidence: `geometry-wave15-numerics/convex-wavefront-arithmetic.json`, reproducible with `scripts/check-convex-wavefront-diagnostic.mjs`. This is an arithmetic diagnosis, not acceptance of skeleton geometry. The narrow contributor correction skips those mathematically redundant checks, retains all other-point/finite/strict-corner checks, and adds both-winding rotated-quad regression. No tolerance is relaxed and no hull is substituted.

Final wave 16 source batch is frozen: county count semantics and GPU-readback area oracle (1 m² absolute), three independent per-building buffer passes with explicit source ownership, convex endpoint exclusion with both-winding regression, and vector-wavefront full-slot reciprocal-loop decoding. Convex offset contour area is compared with the same-float32 GEOS inward 4 m mitre buffer at 0.02 m² absolute tolerance, allowing float32 coordinate rounding over the roughly 300 m footprint perimeter without tolerating material shape change. Final scoped TypeScript and Biome checks passed; WGSL preflight scanned 33,928 templates without findings. These pending source changes are not counted as accepted before the next built GPU run.

## Wave 16 acceptance and final validator correction

Bundle `31e545703f8e9c2990ab47eb3c8ec879c9e3532b9a386c16da552c59782e7eaf` remained unchanged; all browsers closed before slot release. Full focused geometry-refinement GPU spec passed **8/8**, county **10/10**, and footprint **27/27**. County now validates actual GPU-readback face areas against independent GEOS within 1 m²; the archived pair measures 4,228,824,805.750 and 4,257,556,395.938 m². The convex inward 4 m wavefront contour measures 4,411.050 m² versus GEOS 4,411.049677 m², inside the fixed 0.02 m² tolerance. The vector-wavefront display validates all fixed slots and reciprocal closed loops before drawing. Skeleton arc geometry is still not claimed independently oracle-validated.

Block again passed 41 cases, then the first tessellation returned 61/61 valid sites, no sampling flags and completion flags 1,1,1, but the caller rejected its geometry. Root found the validator's second ring loop starts with the first loop's final vertex offset, so every nonempty ring is rejected. A cursor reset and direct CPU valid/invalid multi-ring fixtures are the remaining bounded source correction; no status or area tolerance is relaxed. Exact commands, report SHA256 and acceptance boundaries are in `geometry-wave16-acceptance.json`.

The final block cursor correction is exactly a reset to zero between monotonic ring-offset validation and per-ring length/finite-coordinate validation. `scripts/check-block-tessellation-offsets.mjs` extracts the production pure validator and its capacity constants through the TypeScript AST, then validates a shell+hole and rejects required-count mismatch, decreasing offsets, short rings and nonfinite occupied coordinates. Removing the reset reproduces rejection of the valid fixture. Scoped typecheck, Biome and diff checks passed. This source change is outside the wave 16 fingerprint and awaits the block-only rebuilt GPU retry; prior county/footprint acceptance is unchanged.

Before the final block retry, the harness was strengthened without runtime changes: complete inset cases must match independent boundary-sample counts exactly (61, 40 and 32 sites at 4, 8 and 16 m spacing), and the 16-site budget must report 16 written of 40 required. Each complete case must expose all three GPU-readback area chart values before its displayed total is compared with the independent 4,304.673730089795 m² enclosure at the existing 3.5 m² absolute tolerance (including displayed integer-area rounding). Missing area charts no longer count as a passed numerical case. This tightening does not retroactively change earlier partial reports.


## Wave 17 final block acceptance

The complete **48/48 block suite passed**, errors empty, against unchanged bundle `56f7715e655d8fe93b59b426b475b87848b7947fb8e4d454faff5353908b4a02`. Three inset cases produced exactly **61, 40 and 32** independently expected sites at **4, 8 and 16 m** spacing. Each exposed three GPU-readback owner areas; displayed sums were **4,305 m²** versus the independent **4,304.673730089795 m²** enclosure, inside the predeclared **3.5 m²** tolerance including integer display rounding. This establishes total support, not exact per-owner Voronoi geometry.

At 16-site capacity the scene correctly reports **16 of 40**, overflow 1 and withheld output. At zero inset it reports **42 of 42** sites but incomplete owner geometry, matching the independently identified four cross-owner duplicate-site pairs; this withheld case is accepted as an honest contract response, not a valid numerical geometry. Restoring the 0.4 m inset and full capacity recovers complete output. Graph metric, query, cache and bounded drawn-edge tests also pass.

Desktop and mobile tessellation screenshots were inspected: three distinct computed cells, archived footprint outlines, scale and legend are visible; the desktop enclosure is fully framed. Both capture and numerical browsers closed before explicit GPU slot release. `geometry-wave17-acceptance.json` retains commands, raw report/reference/screenshot paths and SHA256 hashes. Earlier partial failures remain intact in their wave-specific records. No runtime source changed during this accepted run.


## Candidate 21 sampling render correction and acceptance

A completed candidate 20 visual revealed a green point at the local frame origin in Lake Michigan. Source tracing showed that the area branch decoded rendering roles before the sampler wrote provenance. Thus zero-initialized roles colored all unused slots as primary points; their documented zero coordinates piled up at the origin, and the actual zero-weight legacy site was also green. This was a rendering adapter sequencing defect. The sampler’s accepted status/weights/provenance were unaffected; the original 28-case harness did not assert rendered roles or coordinate membership and therefore did not detect it.

The minimal correction moves the unchanged area role decoder immediately after its `GPUContinuousSpatialSample` producer, matching the existing line branch. No core algorithm, source geometry, sample position, weight or status computation changes. Scoped formatting/typecheck passed. Candidate **21**, bundle **`9a5f744c3e1344cc04be45c9626e293af98d36d77f2e058b27acbfe26c476f82`**, then passed the unchanged **28/28** suite and **2/2** new focused checks, with no errors and unchanged before/after hashes.

The new test-only verifier reads the exact native production buffers: all **2,048** rendering roles match producer provenance, all **2,040** unused slots carry the no-data sentinel, and all **8** occupied points lie inside the independent source GeoJSON polygon. Seven random points retain **40,000 m²** weights; the one legacy point has weight zero. The actual PNG contains **46 orange pixels** at the projected legacy coordinate and **zero green pixels** at the former origin artifact. The image was also visually inspected. This additional coordinate oracle applies to the default one-wave, 200 m spacing, seed 47021 design, not all combinations. Both browsers closed before GPU slot release. Exact commands, raw buffer evidence, report and image hashes are in `geometry-wave21-acceptance.json`.

The final county framing adjustment is metadata-only: all six cameras share the union of archived Curry/Josephine bounds with 24 px padding, preserving transitions, the fallback initial view and shared user-gesture guards. Candidate 20/21 whole-story visual acceptance is owned by the root integration sweep; earlier county numerical acceptance is unchanged.


## Final full-suite RS02 fixture reconciliation

The first full `yarn test` completed all eight headless shards with one failure: RS02 composed polygon coverage returned 21 compact tuples versus an exact 18-tuple assertion. A focused diagnostic rerun preserved that failure and captured all fractions. The extra tuples `(zone 0, cell 7)`, `(zone 1, cell 4)` and `(zone 1, cell 7)` each have coverage **2.9802322387695312e−8** versus independently expected zero, exactly **1/16** of the existing zero-reference bound **4 × 2⁻²³**. All expected positive cells were present. These are permitted float32 boundary slivers following the separately reviewed cell-local coverage correction, not missing positive geometry.

The authorized correction changes only the composed test: compare all **40 zone/cell pairs**, including independent expected zeros, at the **unchanged** absolute/relative geometry bounds. Strict unique ordered compact IDs, positive bounded fractions, complete required count and zero overflow are also checked. Actual represented weights remain in the independent categorical winner calculation; no slivers are filtered out. Existing direct exactextract reference bounds, categorical winner semantics and class/missing-area conservation assertions remain intact. No runtime algorithm or tolerance changed.

Focused source-based GPU verification passed **3 files / 25 tests**: full RS02 categorical, fractional zonal distribution and fractional coverage rounding specs. Browsers closed before slot release. `rs02-wave22-acceptance.json` retains the expected failure, exact tuples, focused command/report and source/fixture hashes. The root owns the required subsequent full build and full-suite rerun; this focused pass is not their substitute.
