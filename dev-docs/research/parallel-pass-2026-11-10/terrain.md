# terrain report (2026-11-10)

## 1. Done
- STD-03: the four bare asserts in gpu-terrain are gone (peak-prominence x2, summits, summed-area table) and are now
  validateRequirement / validateMaximumCount messages. Kernel-audit case + rejection added for GPUTerrainPeakProminence
  ("terrain-peak-prominence grid cells must be at most 3355443").
  Files: terrain-features/gpu-terrain-peak-prominence.ts, gpu-terrain-summits.ts, topographic-position/terrain-summed-area-table.ts,
  test/analysis-core/kernel-audit/cases-terrain.ts, rejections-terrain.ts.
- STD-04: named ports added beside what existed (all optional, additive):
  - GPUTerrainDepressionHierarchy: output.requiredCount, output.overflow (copied from the loose status triple, which stays).
  - GPUTerrainSpikeRepair: output.converged, output.iterationLimitReached (beside statistics word 4).
  - GPUTerrainHeightAboveDrainage, GPUTerrainWatersheds, GPUTerrainStreamOrder: status.iterationLimitReached.
  - GPUProfilePeaks: status.iterationLimitReached.
  New helpers (terrain-processing-utils.ts): createTerrainStatusPortNodes, validateTerrainStatusPorts;
  (hydrology/terrain-flow-tracing.ts): createTerrainIterationLimitNode, iterationLimitReached in createTerrainFlowPointerNodes.
  Specs extended: gpu-terrain-drainage(.node).spec.ts, gpu-terrain-spike-repair.spec.ts, terrain-processing.spec.ts, gpu-profile-peaks.spec.ts.
  Docs: docs/api/gpu-terrain/operators.md, terrain-processing.md.
- STD-07 and STD-06: no code change needed. Every gpu-terrain class outside terrain-processing already takes elevation as a
  GPURasterBand (validity, nodata, scale and offset are canonicalized from the band), so the input no-data rule exists. The six
  terrain-processing classes got validity on 2026-11-09. Classes without cellSizeMode are pixel-space or vertical-only by documented
  design (see table). The library review (C1, C2) predates 2026-11-09: flow distance and stream network already have cellSizeMode.

## 2. Verified
- `npx vitest run --project node` kernel-audit.node.spec.ts: 8/8; gpu-terrain/{terrain-features, terrain-processing, hydrology, terrain-decode, topographic-position}: 33, 6, 68, 11, 6 tests, all pass.
- `npx vitest run --project headless` gpu-profile-peaks, gpu-terrain-spike-repair, terrain-processing, gpu-terrain-drainage specs: 4 files, 50 tests pass (new ports asserted on a real device, including the limit-reached cases).
- `node scripts/typecheck-status.mjs 'gpu-terrain|terrain'`: 0 errors (cycle finished 6:30 PM, after the source edits; spec edits made around the same time were not separately confirmed by a later cycle).
- `node scripts/check-contributor-contract.mjs --all`: no terrain bare-assert findings left. Remaining: single-invocation-data-loop in depression-hierarchy (TE03, left alone).
- biome check --write on gpu-terrain src and tests.
- Not run: the full gpu-terrain node directory in one invocation (it hit a startup error from vitest when combined with a second path; ran per directory instead), parity specs, other headless terrain specs.

## 3. Not done
- STD-07 (new validity inputs): nothing applies, see above. Output "validity" ports: terrain-processing outputs and flow/HAND/indices outputs use NaN or sentinel words (0xffffffff, 0xfffffffe) rather than a separate validity raster; I added none.
- STD-06 point-horizon: has its own `projection` ('planar' | 'web-mercator' | spherical) model in the parameters packer; not mapped to cellSizeMode (would change its contract).
- STD-04: GPUTerrainFlow keeps its stage-named ports (fillConverged, flatsConverged, accumulationConverged, *IterationCount); no overall iterationLimitReached added. TiledFlow and CatchmentSummary have only per-cell status words, no scalar count (would need new GPU reductions). Contours already has status.overflow.

## 4. Wave 2 requests
- None required. Optional: GPUTerrainSpikeRepair `statistics` loose view could be deprecated in favour of named ports once consumers move.
- Shared file wanted (not mine): `createRasterIterationFinalizeNode` (raster-analysis/cost-distance/raster-relaxation.ts) could take `iterationLimitReached` like `createRasterJacobiLoopNodes` does; then createTerrainIterationLimitNode can go.

## 5. Roadmap
- STD-03: terrain directories done (no gpu-terrain bare asserts remain).
- STD-04: add "terrain: iterationLimitReached on HAND, watersheds, stream order, profile peaks, spike repair; requiredCount and overflow on depression hierarchy. Remaining: tiled flow and catchment summary scalar status".
- STD-06 and STD-07: terrain needs nothing further (band validity is the no-data input; remaining classes are pixel-space by design). Remove terrain from the remaining lists; point-horizon keeps its own projection model.

## 6. Noticed
- library-review code-terrain-pointcloud.md C1/C2 is out of date for terrain-processing, flow distance and stream network.
- The vitest run with a `gpu-terrain` directory plus kernel-audit path in one call fails at startup ("Cannot convert undefined or null to object"); per-directory works.

## Per-class state (V in = validity input, V out = output validity, CSM = cellSizeMode, Status = named status)
band = through GPURasterBand validity; NaN/sentinel = invalid marked in output values; n/a = not applicable
| Class | V in | V out | CSM | Named status |
|---|---|---|---|---|
| GPUTerrainDerivatives | yes (band) | yes (outputValidity) | yes | n/a |
| GPUTerrainCurvature | yes (band) | yes | yes | n/a |
| GPUGeomorphons | yes (band) | yes | yes | n/a |
| GPUTerrainContours | yes (band) | n/a (geometry) | n/a (grid space) | yes (status.overflow) |
| GPUTerrainViewshed / CumulativeViewshed / LineOfSight | yes (band) | NaN/sentinel | yes | n/a |
| GPUTerrainSummits | yes (band) | NaN drop | yes | yes (compact status, radiusClamped) |
| GPUTerrainPeakProminence | yes (band) | NaN | n/a (vertical) | n/a |
| GPUTerrainPeakSnap | yes (band) | snapStatus | yes | yes (per-row status) |
| GPUTerrainCriticalPoints | yes (band) | class code | n/a (documented) | n/a |
| GPUProfilePeaks | yes (validity) | NaN | n/a (1-D) | yes (converged, iterationLimitReached new, compact status) |
| GPUTerrainRuggedness / TopographicPosition | yes (band) | yes | n/a (pixel windows, documented) | n/a |
| GPUTerrainVectorRuggedness / WeissLandforms | yes (band) | yes | yes | n/a |
| GPUSimpleLocalRelief / LocalDominance / MultiScaleRelief / ReliefBlend | yes (band; blend takes layers) | yes | MSR yes, others pixel-space (documented) | n/a |
| GPUReliefShading / TextureShading / SolarIrradiance / SolarShadowMask / SolarPosition | yes (band or n/a) | yes (where raster) | ReliefShading yes; others pixel-space or n/a | n/a |
| GPUTerrainHorizon / CastShadow | yes (band) | yes | yes | n/a |
| GPUTerrainFlow | yes (band) | sentinel | yes | yes (stage-named converged / iterationCount) |
| GPUTerrainHeightAboveDrainage | yes (band) | NaN | n/a (vertical) | yes (converged, iterationCount, iterationLimitReached new) |
| GPUTerrainWatersheds / StreamOrder | n/a (directions) | sentinel | n/a | yes (iterationLimitReached new) |
| GPUTerrainHydrologicIndices | yes (band) | NaN | yes | n/a |
| GPUTerrainFlowDistance / StreamNetwork | n/a (directions) | yes / sentinel | yes | yes (unresolvedCount, status.overflow) |
| GPUTerrainFlowField | yes (band) | NaN | yes | n/a |
| GPUTerrainRGBDecode | yes | yes | n/a | n/a |
| GPUTerrainSpikeRepair | yes (band) | yes | n/a (heights only) | yes (converged, iterationLimitReached new) |
| GPUPointHorizonProfile / Visibility | yes (band) | noData | own `projection` model | n/a |
| terrain-processing (Breach, DepressionHierarchy, FlowFractions, BranchingFlow, CatchmentSummary, DEMCoregistration) | yes (validity, 2026-11-09) | NaN | yes except Breach (vertical) | Hierarchy yes (new), Breach/Branching unresolvedCount, others per-cell |
| GPUTerrainTiledFlow | n/a (runoff NaN) | status word | n/a (transport sum) | per-cell only |
