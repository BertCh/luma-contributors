# County coverage audit implementation

The coverage-audit tool is a lazy-loaded extension of the noding-and-coverage scene. It selects all 266 archived county features in Illinois (17), Indiana (18), and Wisconsin (55), preserving their original feature order, polygon parts, rings, and vertices. The subset contains 268 polygon parts/rings and 3,511 stored vertices. It uses the archived Census county boundaries dataset (2022 vintage), which contains 3,109 county features overall; this is not a live or complete current boundary service.

`coverage-audit-data.ts` filters the existing GeoArrow layout by state FIPS and projects all selected coordinates to the scene's Albers CONUS frame. The alternate input moves one copy of an exactly shared boundary vertex by 50 projected metres. That deliberate scenario tests whether validity flags respond to a boundary mismatch; it must never be described as a defect in the archived source. The selection is bounded for interactive GPU topology work and should remain clearly named in all readouts.

The compute helper builds one graph per `(input, dissolve label)` key. `GPUCoverageValidity` compares segments across different features and emits the actual per-vertex flags, per-feature invalid-segment counts, whole-coverage validity, and total invalid count. `gapWidth` is a frame parameter measured in projected metres. This check does not test self-intersections within each polygon. The flag chart and edge colors are derived from GPU readback, and all summary readbacks are small and event-driven rather than copied on every frame.

`GPUCoverageDissolve` consumes the same effective positions and labels, with zero snapping, a 1 mm vertex matching tolerance, and touching-ring splitting enabled. Its GPU outputs include ring geometry, shell/hole classifications, GeoArrow polygon offsets, source label IDs, ring counts, overflow, open/touching segments, and boundary segment count. A small GPU pass derives the ring vertex count and polygon feature count for display. The layer is shown only if the effective coverage is valid, the required rings fit, and no open segments remain. Grouping by state retains the three state outlines; a single label also removes those borders. This is a coverage-specific operation that assumes valid matched edges; it neither performs general Boolean overlay nor repairs invalid input. It is distinct from `GPUSegmentRingAssembly` and `GPUCoverageSimplification`.

Both inputs use the same Albers CONUS projected metre frame as the existing US map tools. It is appropriate for the bounded topology demonstration, but its scale and area distortion vary across the three-state extent. The map is not a measurement of uniform ground distances. Capacity/status outputs remain visible, including when output is withheld.

## Sources

- Census county boundary archive: `us-counties` bundled GeoArrow dataset, Census 2022 vintage; 3,109 county features in the archived file, with `stateFips` and county `fips` columns.
- GPU operator contracts: local `GPUCoverageValidity` and `GPUCoverageDissolve` implementations and API docs in `modules/graph/src/gpu-spatial-analysis/polygon-coverage-topology/`.
- Coverage validity semantics: [Shapely `coverage_is_valid`](https://shapely.readthedocs.io/en/stable/reference/shapely.coverage_is_valid.html) and [Shapely `coverage_invalid_edges`](https://shapely.readthedocs.io/en/stable/reference/shapely.coverage_invalid_edges.html).
- Coverage union semantics: [Shapely `coverage_union_all`](https://shapely.readthedocs.io/en/stable/reference/shapely.coverage_union_all.html).
