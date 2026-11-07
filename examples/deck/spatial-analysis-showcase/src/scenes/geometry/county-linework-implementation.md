# County linework investigation

This scene isolates three operators that answer different questions about a bounded real county pair: `GPUSharedPaths` reports common runs and relative direction, `GPULineNoding` publishes atomic segments with source-edge intervals, and `GPUPolygonize` creates bounded faces from arbitrary linework while reporting the remaining edge classes.

## Source geometry

The input is the bundled archived Census county GeoArrow dataset (`us-counties`, 3,109 features), not a live Census service. It selects Curry County, Oregon (FIPS 41015; source feature row 2,182) and Josephine County, Oregon (FIPS 41033; source feature row 2,191). The selected source geometries are each one polygon with one exterior ring: Curry has 77 stored vertices and Josephine has 58; stored vertex counts include the repeated closing coordinate. Their source longitude/latitude bounds are Curry `[-124.5524368, 41.9956201, -123.7164917, 42.9549484]` and Josephine `[-124.0270691, 41.9951401, -123.2296219, 42.7840614]`.

The coordinates are projected into the showcase Albers CONUS metre frame. In the archived inputs, an independent exact-coordinate comparison found 46 unique shared directed edges along the common border, each present in both rings (92 directed occurrences). These figures describe the bundled archived extract and the selected pair only.

## GPU stages

- `GPUSharedPaths` compares the two source rings and reports run offsets, source line IDs, orientation and bounded-output status. Reversing the Josephine coordinate order preserves its boundary while changing the directional relationship.
- `GPULineNoding` receives both lines together and reports output endpoints, source county FIPS, ring and edge IDs, parameter intervals, required segment count, candidate overflow and uncertainty.
- `GPUPolygonize` receives the same combined linework and performs its own noding. It does not consume the separate `GPULineNoding` result. Its outputs include bounded faces, noded-segment diagnostics, face edge classes, and required ring capacity. A common grouping is used so the result is polygonization of the combined two-county linework; output faces are not attributed to either county.
- The optional open-line scenario omits Curry source edge 0 and marks its original location in red. This synthetic input is intentionally open; it does not identify an error in the archived source.

All result overlays are withheld unless output counts, capacities, candidate status and exact-or-uncertain predicate status support a complete display. The 1 mm vertex identity tolerance applies in projected Albers coordinates; float32 storage still places a finite limit on numerical resolution. Polygonization returns all bounded faces, including any face that fills a polygon hole, and does not preserve source fill parity or make-valid semantics.

## References

- US Census Bureau, [Cartographic Boundary Files](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html).
- Shapely, [`polygonize`](https://shapely.readthedocs.io/en/stable/reference/shapely.polygonize.html) and [`shared_paths`](https://shapely.readthedocs.io/en/stable/reference/shapely.shared_paths.html) semantics, used for conceptual comparison. The local GPU operators' contracts and outputs are authoritative for this implementation.
