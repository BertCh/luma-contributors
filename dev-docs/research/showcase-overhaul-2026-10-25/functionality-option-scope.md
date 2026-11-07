# Option scope and remaining assurance

Option assurance is deliberately incomplete; no whole-union or Cartesian-product completion claim.

- A literal absent from static call arguments may be supplied by a control, default or composed contributor; it is a candidate for review, not automatically a missing behavior.
- A literal present at a call site is source evidence only, not proof the branch executed or the result is numerically correct.
- Numeric, resource, status and layout properties remain in the full inventory; not every property merits a reader-facing control.
- Known meaningful unexercised variants are retained explicitly below. Rebuilt suites must pass before pending rows become accepted.

## Dataframe order and validity

Owner: `data_coverage`. Decision: `bounded-accepted-not-union-complete`.

Wave 14 county 17 passes nullable expressions, four joins, global/state-batch ordering and source-null versus explicit valid-NaN interpretation.

Ordering evidence is bounded to checked head/tail and state-head rows; arbitrary fluent compositions are not certified.

## Raster selection and adaptive region merging

Owner: `data_coverage`. Decision: `bounded-accepted-not-union-complete`.

Wave 14 fire 22 passes above/below/range, inclusive/exclusive source equality, and fixed/adaptive count-weighted means under 4/8 watershed connectivity and 1/4/16 rounds.

Only stated source support, thresholds and authored basins are certified; this is an analytical scenario, not observed classification.

## Standardization and exact temporal reduction

Owner: `data_coverage`. Decision: `bounded-accepted`.

36 normalization configurations; seven grouped-moment cases; six exact-word sorted/full-scan temporal cases including source-row tie identity.

These bounded cases do not certify every input distribution, timestamp range or null pattern.

## Affine transform

Owner: `atlas_experience`. Decision: `bounded-accepted-plus-unexercised-variants`.

Wave 15 CTA 15-case suite accepts declared translation and inverse with exact float32 coordinates and bounded source order/length.

Rotation, skew, nonuniform scale and origin/centroid alternatives remain unexercised.

## SMRF classification writes

Owner: `atlas_experience`. Decision: `explicit-unexercised-output-policy`.

Ground mask/support demonstrated with default onlyGround=false.

onlyGround=true preserves existing non-ground class labels and is not demonstrated; it does not change the ground detector.

## Point-cloud neighborhood strategy

Owner: `atlas_experience`. Decision: `execution-strategy-not-separate-story`.

Default serial decomposition demonstrated.

Cooperative decomposition unexercised; implementation strategy, not another scientific estimator.

## Raster projection precision

Owner: `earth_systems`. Decision: `bounded-correction-accepted`.

Wave 14 adaptive provider/local-frame GPU correction passes all 4,096 center comparisons at maximum Euclidean error 0.000763260 m and all eight masks exactly.

Prior Float32 baseline remains a rejected negative control. Sampled fit certificate is distinct from full-grid measured accuracy; no sensor geolocation accuracy claim.

## Border and missing-data policies

Owner: `earth_systems`. Decision: `bounded-accepted`.

Wave 16 accepts all 48 median border/missingness cases and six include/exclude crop-border exposed-face cases.

Other algorithm defaults do not imply every missing-data policy was demonstrated.

## Convolution and curvature

Owner: `earth_systems`. Decision: `configured-contract-plus-composed-behavior`.

All three curvature controls represented; signed rectangular derivatives intentionally unnormalized; positive weighted smoothing normalizes weights.

Standalone generic positive-kernel normalization spelling is not explicitly exercised; signed zero-sum normalization is mathematically inappropriate.

## Observer heights and extrema support

Owner: `earth_systems`. Decision: `explicit-scientific-contract`.

Ground-relative observer height is an explicit source contract; extrema coordinates follow the selected ray interpolation footprint.

Absolute-datum observer alternatives are unclaimed. Do not invent independent coordinate choices incompatible with the sampled support.

## Tile ownership and reconciliation

Owner: `earth_systems`. Decision: `bounded-accepted-with-reference-limits`.

Wave 16 accepts all 58 tiled baseline records and both cross-tile topology cases, with corrected categorical display reviewed.

Tiled versus whole-window numerical values are GPU self-consistency; independent NumPy validates support/statistics and independent flood-fill validates topology. Graph-cache acquisition remains uninvoked.

## Network direction, waits and weighted adjacency

Owner: `movement_networks`. Decision: `bounded-accepted`.

Chicago 17, flight matrix 10, and aviation direction/coarsening eight configurations have bounded numerical acceptance with separate fingerprints.

Louvain aggregation is checked conditional on actual GPU membership, not global optimality or independent heuristic labels. Flight bins retain documented padding/global-maximum semantics.

## Network storage and sorting variants

Owner: `movement_networks`. Decision: `execution-storage-contracts`.

Accessibility rows/columns compose CostWeights; snapping spatialSort is a BVH/Morton strategy; weighted buffer output computes the statistical measure.

textureStatistic serialization and alternative storage layouts do not represent additional statistical measures and are not credited as executed.

## Trajectory support and uncertainty

Owner: `movement_networks`. Decision: `bounded-accepted-wave13`.

Wave13 ten independent quality cases accepted full covariance filter/RTS, exact epoch preparation and polygon/hole compact/merged intervals.

Acceptance is confined to the stated ten cases; arbitrary covariance parameters, trajectories and supports are not exhaustively certified.

## Directional moments

Owner: `movement_networks`. Decision: `bounded-accepted`.

Wave 15 accepts all eight directional/axial × equal/segment-length × all/fare-group combinations.

Modeled routes are not GPS tracks; length weighting is descriptive and does not establish a Rayleigh test.

## Population redistribution support

Owner: `spatial_statistics`. Decision: `bounded-accepted-plus-explicit-weight-contract`.

Wave 15 accepts 14 exact-ID alignment, count-weighted allocation/exposure and source → allocation → source display checks.

Density weights remain unexercised. Built-up redistribution is an assumption, not observed occupancy; tree cover is not health risk.

## Conditional simulation preparation

Owner: `spatial_statistics`. Decision: `bounded-accepted-plus-uninvoked-lifecycle`.

Accepted22 cases cover analytic moments, finite draw counts, offsets and observation-error changes via invalidatePreparation.

invalidateObservations with retained geometry is unused; exact PRNG stream/per-query draw certification and all seed choices are unclaimed.

## Quantile LISA inference

Owner: `spatial_statistics`. Decision: `descriptive-bounded-accepted`.

Wave 14 descriptive membership/join-count suite passes 19 bounded cases against independent type-5 quantiles and directed joins.

Legacy pseudo-p inference remains unused; no significance or complete null calibration is inferred.

## ML reference backends

Owner: `spatial_statistics`. Decision: `explicit-reference-backend-scope`.

Accepted GPU spatial ML estimands checked by independent NumPy/SciPy oracle.

Public fitSpatialLagMaximumLikelihoodOnCPU/fitSpatialErrorMaximumLikelihoodOnCPU are not directly invoked; no backend-selector credit.

## Variogram fitting and conditional LOO

Owner: `spatial_statistics`. Decision: `bounded-accepted-not-union-complete`.

Wave 14 passes 18 fitted combinations and conditional LOO identity/probe checks. Maximum GPU objective relative difference 7.78e-7; prediction probes 6.30e-5 mm and variance probes 4.15e-4 mm².

Fourteen fits hit disclosed boundaries/status 3. LOO conditions on a variogram fitted to all data; it is not nested spatial validation.

## Class breaks and integer colors

Owner: `spatial_statistics`. Decision: `bounded-accepted-not-union-complete`.

Wave 14 five cases pass bounded New Jersey 21-county exact/approximate Jenks and integer numeric/ordinal color contracts.

No exact-Jenks scale generalization to the national set and no arbitrary color/input union guarantee.

## Spherical geometry

Owner: `geometry_observation`. Decision: `bounded-accepted-geometry-limit`.

Membership/S2 bracketing/Pacific Boolean cases have independent evidence.

Chart overlays and intrinsic arc vertices are not independently geometry-oracle certified merely because statuses pass.

## Shape, sampling and polygonization variants

Owner: `geometry_observation`. Decision: `bounded-accepted-with-geometry-limits`.

County10, footprint27 and core8 accepted on wave 16; block48 accepted on wave 17 with exact sites 61/40/32 and total displayed area 4305 m² versus GEOS 4304.673730 m².

Block acceptance checks total enclosure support, not exact per-owner/per-vertex Voronoi geometry. Offset skeleton arc geometry and broader spherical limits remain explicitly bounded. No finite geometry suite is pending.

## Clustering extraction and null ensembles

Owner: `geometry_observation`. Decision: `bounded-accepted-not-union-complete`.

Nine OPTICS DBSCAN cuts and stated envelope calibration states independently accepted.

Those checks do not exhaust Xi, EOM, every epsilon/minimum-size or all null/inference assumptions.

## Circuit conditioning and publication

Owner: `earth_systems`. Decision: `bounded-numerics-and-explicit-withholding`.

Four mild cases pass independent resistance/energy tolerances. Four strong cases fail the strict certificate and correctly suppress charts/maps; one mild return verifies recovery. Separate 16² comparison can also withhold independently.

Strong 64² values are unavailable numerical predictions. No general convergence, calibrated habitat connectivity or physical-residual validation of rejected solutions is claimed. Experimental solver modifications were removed after failed GPU checks.

The unchanged static literal-candidate inventory remains in [the JSON record](functionality-option-scope.json). Missing literal spelling does not establish missing behavior.
