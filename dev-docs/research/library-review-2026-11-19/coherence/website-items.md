## c01-naming
- NAM-03: `website/src/catalog/entries/cells-index-join.ts:2389`: ADDITIVE FORM: Publish `'lowest-row'` as the canonical default, explain the feature-row tie rule and document `'lowest-id'` as its deprecated spelling. Shared with: spatial-geometry, core-contract, consumer-seams-lenses. After: spatial-geometry.
- NAM-04: `website/src/catalog/entries/raster-embeddings-flow.ts`, `website/src/catalog/entries/terrain-hydrology-flow.ts`, `website/src/gpu/runs/entries/raster-embeddings-flow/lagrangian-flow-map.ts`, `website/src/gpu/runs/entries/raster-embeddings-flow/lagrangian-finite-time-lyapunov-exponent.ts`, `website/src/gpu/runs/entries/terrain-hydrology-flow/particle-advection.ts`, `website/src/gpu/runs/entries/terrain-hydrology-flow/line-integral-convolution.ts`, `website/src/gpu/runs/entries/terrain-hydrology-flow/streamlines.ts`, `website/src/gpu/runs/entries/terrain-hydrology-flow/terrain-flow-field.ts`: Use the named placement objects in catalog examples and runnable packer calls, preserving each origin and signed cell size. Shared with: raster-terrain, spatial-geometry, consumer-seams-lenses. After: raster-terrain.

## c05-defaults-required
- DEF-01: `website/src/gpu/runs/entries/recipes-trajectories/quantile-lisa.ts`: Pass the runner’s intended `alternative` explicitly and identify the rule alongside the p-value. Shared with: core-contract, spatial-stats, cells-trajectories-recipes, consumer-seams-lenses. After: spatial-stats.
- DEF-01: `website/src/catalog/entries/recipes-trajectories.ts`: Update the catalog constructor example and tail description to match the executable runner, using `'geoda-legacy'` for an intentional GeoDa comparison. Shared with: core-contract, spatial-stats, cells-trajectories-recipes, consumer-seams-lenses. After: spatial-stats.

## c08-egress-decoders
- EGR-04: `website/src/gpu/runs/entries/flows-cubes/cell-period-keys.ts:87,171,234`: ADDITIVE FORM: Select the same explicit layout for producing keys, masking period bits and decoding displayed cells. Use zero bits for new one-period identity keys and the legacy layout for existing one-bit keys. Shared with: dataframe-pointcloud-record, core-contract, api-docs-catalog. After: dataframe-pointcloud-record, core-contract.
- EGR-05: `website/src/gpu/runs/entries/flows-cubes/cell-period-keys.ts:87`: ADDITIVE FORM: Call the checked unpack helper and guard `undefined` before destructuring. Skip the excluded row and surface a malformed encoded period as an error. Shared with: dataframe-pointcloud-record, core-contract, api-docs-catalog. After: dataframe-pointcloud-record, core-contract.

## c09-forms-parity
- FRM-13: `website/build/markdown-plugin.ts`: Render each contributor’s catalog-backed forms table on its own evidence page when compiling the Markdown. Show the qualified exports, actual signatures, supported inputs, differences in defaults and limits, and a runnable CPU invocation where available. Shared with: api-docs-catalog. After: api-docs-catalog.

## c14-docs-structure
- DST-07: `website/build/markdown-plugin.ts:102`, `website/src/routes/docs/$.tsx:32`: Render `::lens-catalog` as a labelled link to the family catalog and each `::lens` recording as a labelled card or link to its recorded example. Handle adjacent directives in the same paragraph and verify the HTML delivered by the docs route. Shared with: api-docs-catalog, network, raster-terrain.
- DST-13: `website/build/markdown-plugin.ts:154`, `website/src/docs.ts:18,28`: Distinguish registered docs pages from repository files during link rewriting. Resolve `examples/deck/kepler-compute/VALIDATION-CATALOG.md`, `dev-docs/research/spatiotemporal-review-2026-11-09/README.md` and `examples/deck/kepler-compute/README.md#point-cloud-contributor-catalog` through the configured source base, or serve equivalent docs destinations. Preserve the point-cloud fragment. Shared with: core-contract.

## d02-completeness-travel
- CPL-09: `website/src/gpu/runs/entries/flows-cubes/kernel-flow-smoothing.ts:58`: ADDITIVE FORM: Pass the runner’s fixed uploaded rows as `{count, completeness: 'known-complete'}`. Shared with: network, core-contract. After: network.
- CPL-09: `website/src/catalog/entries/flows-cubes.ts:146`: ADDITIVE FORM: Update the displayed example to use producer `status: {count, overflow, candidateOverflow?}` or explicitly known-complete imported rows, and show `output.status.complete`. Document `validPairCount` as the deprecated old spelling. Shared with: network, core-contract. After: network.

## d03-host-packers-borders
- PCK-04: `website/src/gpu/runs/entries/network-routing-access/network-subgraph-filter.ts:74`: ADDITIVE FORM: Use `edgeAttributeRanges` when packing the interactive speed brush. Verify the displayed `[minimumSpeed, maximumSpeed)` selects the same representable edge speeds at decimal boundaries. Shared with: network, consumer-seams-lenses. After: network.

## d07-degenerate-inputs
- DGN-01: `website/src/gpu/runs/entries/categories-cells-tests/raster-false-discovery-rate-mask.ts:61,74`: After the Pettitt decision, verify the direct correction wiring with one tested and one all-missing cell. If the legacy p-value is retained, apply the support mask before correction. The missing cell must leave the tested cell’s significance decision unchanged.

## f06-raster-terrain
- URT-01: `website/src/catalog/entries/raster-composites-climate.ts:212`: Correct the catalog claim and include the three-date example. NDVI of the jointly fitted composite is `0.2600577`. Taking NDVI per valid acquisition and then its temporal median gives `0.3333333`. Shared with: raster-terrain, LEAD.
