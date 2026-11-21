# Decisions of 2026-11-20 (library review R19, wave 6)

Made by the lead session under the maintainer's instruction "make the right decisions", and carried out in wave 6. Report subsections cited by ID are under `coherence/` (the 32 coherence reports); the 28 review decisions are listed in [the README](./README.md).

Governing principle: the package is unpublished. Before the alpha there is **one name per concept and
no compatibility aliases**. A rename is done directly: the old name is removed, every in-repository
caller (modules, tests, examples, docs, website, lens scenarios, Kepler bindings) is migrated in the
same wave, and the CHANGELOG lists it under "Breaking changes before the alpha". Serialized identifiers
that exist outside code (lens scenario IDs, committed record names) are NOT renamed.

## Breaking proposals (report subsections in [coherence/](./coherence/))

| ID | Decision |
|---|---|
| NAM-13 | `GPUSpatialKMeans.k` becomes `clusterCount` (as in the raster embedding k-means). The four neighbour APIs (`GPUNeighborSearch`, `GPUJacquezTest`, `GPUNearestFeatureJoin`, `GPUNearestFeatureWeights`) rename `k` to `neighborCount` directly (no alias). `k` stays in mathematical prose; `slotCapacity` stays distinct. Every packer, operation form, CPU twin and catalog record follows. |
| NAM-14 | Rename to `GPUDBSCAN`, `GPUHDBSCAN`, `GPUPycnophylacticInterpolation` (props, output types, packers and operation forms follow the same stem). DBSCAN's self-inclusive threshold is `minimumSamples` (scikit-learn's name and meaning: the point itself counts); document self-inclusion. Ontology, docs, lenses and website catalog records migrate; lens scenario IDs stay. |
| OUT-11 | `GPUPairGather` output `rowCount` becomes `status.count`. `matchedCount` keeps its input-domain meaning. Missing destination: `pair-gather output.status.count is required`. |
| STA-13 | The five word-index maps become `GPU_LOCATION_ALLOCATION_STATUS_LAYOUT`, `GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT`, `GPU_AZP_STATUS_LAYOUT`, `GPU_WARD_STATUS_LAYOUT`, `GPU_MAX_P_STATUS_LAYOUT`; the old `_STATUS` names are removed. Word order unchanged. |
| ING-20 | `prepareGPUNetworkRestrictionStates({nodeCount, edges, restrictions})` replaces the positional form; endpoint input type shared with flow preparation; returns formats and counts. Migrate the three executable calls and the displayed Chicago example. |
| FRM-06 | The drainage recipe forwards `flow.cellSizeMode` to both length-producing stages (behaviour fix: geographic and Web Mercator distances change). Document output units; pin with the two-cell equatorial DEM against the standalone geographic contributors. |
| CMP-13 | Resident colour-scale `domainCount` becomes `classCount`; numeric `domainCount` keeps counting entries. Add the threshold-settings overload that needs no dummy edge count. |
| TYP-09 | Narrow the five transfer-mask bindings to `GraphDataView<'uint32'>`; keep runtime validation; mark the intentional compile errors in the negative tests with `@ts-expect-error`. |
| TYP-11 | Encode valid configurations in props: terrain derivatives require at least one of the five outputs; distance fields take point seeds with an optional mask, or a mask alone without `seedIds`/`seedCount`. |
| TYP-12 | Literal keys for the three critical levels plus a checked lookup for dynamic numbers; dynamic operator maps typed `Readonly<Partial<Record<string, Value>>>`. |
| TYP-13 | `Readonly<...>` on the spectral base props and on constructed operation descriptors. |
| TYP-14 | Replace the writable `recorder.graph` getter with an inspection lookup; `recordKernels(graph)` keeps returning the caller's typed graph. |
| TYP-15 | `GPUPopulationSupportAlignmentProps` requires all four outputs; new `GPUPopulationSupportAlignmentRecipeProps` for the optional-output shape. |
| ENT-10 | Remove the parent spatial barrel's re-export of the ontology; migrate the 69 imports to `@luma-contributors/analysis/gpu-spatial-analysis/ontology`. |
| ENT-13 | Remove `getCholeskyWGSL`, `getSymmetricEigenvalueRangeWGSL`, `getSpatialWeightsTracesProbeWGSL` from the regression barrel (defining modules keep them). |
| LIF-09 | Remove `reusePreparation` from `ConditionalGaussianSimulationOperationProps` (direct producer keeps it; docs show the producer for repeated draws). |
| WRT-13 | `GPUPickRegionMask` gets `GPUPickRegionMaskOutput {mask, status: GPUOverflowStatusPort}`; reject the two uncertain counters at construction. |
| UST-15 | The averaging transform is renamed to say what it does: `average-reciprocal` (follow the transform's existing option-value spelling convention: lowercase, hyphenated). The name `symmetrize` is removed, not reused. Docs map PySAL `W.symmetrize` to transpose plus support union; a true support-union transform is a roadmap row, not built now. |
| URT-10 | Hydrologic `slope` output becomes `descentTangent`; terrain derivative `slopeUnits` stays; solar inputs are labelled "Horn or compatible slope in degrees". |
| UDF-14 | `GPUKeyJoin` becomes `GPUKeyLookup`; `kind` is replaced by `compactMatchedLeftRows?: boolean`. Pair-join terminology is reserved for an operation that emits each (left, right) pair. Lead the docs example with duplicate keys. |
| UCO-10 | Kepler discovery enumerates `getGPUAnalysisOperatorCatalog()` once and keys on `{package, exportName}`; remove the 22-name network supplement and the point-cloud, categorical and spacetime passes; distinguish discoverable from runnable with existing fields. The bare-name form is retired. |

## Planner leftovers (planner leftovers)

- EGR-04 / EGR-05: remove the implicit legacy one-period packing and the unchecked unpacker.
- NAM-09: remove `symmetricDifference`; `symmetric-difference` only.
- W2-03: accept the reduced admission domain of location allocation (keep the accurate node estimate); state the limit on the page.
- SSR-121: keep the explicit-grid untruncated Gaussian GGWR route with its documented exhaustive cost; no new ceiling.

## Deprecated aliases added by this pass

Every `@deprecated` alias added since `bd8279a4` (about 186 lines) is retired: callers move to the new name and the old one is removed, by the same rule as above. Exception: an alias whose old name is a serialized identifier read from committed data stays, and the stream says so.

## The review's decision list ([README](./README.md), "Decisions for the maintainer")

| # | Decision |
|---:|---|
| 1 | Package identity: **left to the maintainer** (owner, URL and name of a public package are theirs). |
| 2 | The alpha stays held until this wave lands and one full gate passes on the committed tree. |
| 3 | Adopt stable/experimental tiers. Every catalog record gains `stability: 'stable' \| 'experimental'`, default `'experimental'`; nothing is `'stable'` until the R19-22 admission rule (which includes a second GPU vendor) is met. |
| 4 | The five journeys are confirmed (hot spots, rate maps, kriging, DEM hydrology, network catchments); built after the alpha (R19-20). |
| 5 | luma.gl: land this wave first, then assess alpha.17 with `scripts/upstream/diff.mjs`; then a weekly bump with skipped versions recorded. Not in this wave. |
| 6 | Second GPU vendor and Safari/Firefox runs: needs hardware the lead does not have; stays open. |
| 7 | Keep LRN-06, EMB-03, EMB-04, SIM-03; drop the other LRN, EMB, SIM, CART, ENC rows from the roadmap (lead, at landing). |
| 8 | Done: the known-issues pass is merged. |
| 9 | The budgeted trajectory contributors (thinning, staypoints, locations) **require** their status view. Mobility indicators: a row with an unsettled or missing location key is excluded and counted in status, never counted as a category. |
| 10 | `'unknown'` for unaudited records in the all-entry catalog (applied in wave 5; keep). |
| 11 | Per-model regression status decoders (applied; keep). |
| 12 | Signed zero canonicalised at class breaks (applied; keep). |
| 13 | Probit `tolerance` ceiling 1e-2 through `validateRequirement` (apply if not already in). |
| 14 | Regression recipe `invalidCount` holds a real excluded-row count where the model knows it, and is removed where it does not; the solver code lives only in `code`. |
| 15 | p-median `costLimit` behaviour stays as documented. |
| 16 | Loop allowances cite a named bound (wave-5 rule; keep). |
| 17-22 | All six provisional changes are kept. |
| 23 | Reprioritisations accepted (lead, roadmap at landing). |
| 24 | Roadmap split (R19-25) done at landing by the lead, not in this wave. |
| 25 | Upstream policy accepted. |
| 26 | Legacy status views (STD-04/STD-09): where a vocabulary-conforming status port exists for the same producer, the legacy view is removed and callers migrate. |
| 27 | Kept. |
| 28 | Lenses stay private. Compatibility statement: luma.gl `~10.0.0-alpha.16` as tested, Arrow `>=17`. Packages ship `dist` (ESM, CommonJS, declarations, source maps) only; `src` and `dist/.build-record.json` are excluded from the tarball. |
