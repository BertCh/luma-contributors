# fixups

## 1. Done
Task 1, six status fields moved. All six were introduced today (none existed before). In four classes the output already has `status` as a packed/per-row GraphDataView, so named ports cannot hang under it; they sit under `ports.status` (path contains the `status` segment the vocabulary spec requires). Spike repair had no `status`, so it got a plain `status` object.
- GPUTerrainDepressionHierarchy: `output.ports.status.requiredCount` / `.overflow` (gpu-terrain-depression-hierarchy.ts, new local getOutputViews; terrain-processing-utils.ts validateTerrainStatusPorts takes an optional message prefix, default `output.status`).
- GPUTerrainSpikeRepair: `output.status.converged` / `.iterationLimitReached`.
- GPUOPTICS: `output.ports.status.candidateOverflow` (rejection message and docs updated; helper getOPTICSOutputViews).
- GPUNetworkKernelDensity: `output.ports.status.invalidCount` (invalidCount is an accepted status word).
- Specs/docs updated: terrain-processing.spec.ts, gpu-terrain-spike-repair.spec.ts, optics-grid-index.spec.ts, network-kernel-density-invalid-count.spec.ts, kernel-audit/cases-coverage-point-process.ts; docs/api/gpu-terrain/terrain-processing.md, docs/api/gpu-terrain/operators.md, docs/api/gpu-spatial-analysis/point-process.md. Operator catalog lists only vocabulary words, nothing to change. No allowlist entries added.
Task 2, all listed specs typed properly (generic helper formats instead of union formats, OPTICSRun result type, ArrayLike for expectClose, weak-type props annotation, missing readout fields ringCount/ringStatusRequiredCount). No source typing defect found.

## 2. Verified
- `vitest --project node` vocabulary, geospatial-validation, raster-input-mask, overlay-face-labelling-executed, raster-terrain-expansion, kernel-audit dir: 55 files, 2776 tests passed (includes both previously unfinished specs: overlay-face-labelling-executed.node.spec.ts and geospatial-expansion/raster-terrain-expansion.node.spec.ts both pass; slow, took several minutes).
- earlier node run of vocabulary + terrain-processing.node + spike-repair node + kernel-audit: 53 files, 2710 tests passed.
- `vitest --project headless` for pair-statistics-grid-paths, optics-grid-index, island-attachment-merge, fuzzy-contiguity-index, network-kernel-density-invalid-count, raster-walking-cost-parameters, terrain-processing, spike-repair: 8 files, 58 tests passed.
- `NODE_OPTIONS=--max-old-space-size=8192 node scripts/typecheck-tests.mjs`: analysis and lenses 0 errors; kepler and example 0 errors at the last run (total 0).
- `node scripts/typecheck-status.mjs`: 0 errors. `yarn check-contributor-contract`: no new findings. biome check --write on edited files: no remaining issues.

## 3. Not done
Nothing outstanding.

## 4. Wave 2 requests
None. The new `ports.status` shape is only on fields added today; consumers: none found in modules/lenses, modules/kepler, website/src, examples.

## 5. Roadmap
No rows touched.

## 6. Noticed
- An earlier biome run over `modules/analysis/src/gpu-terrain` reformatted 2 files, which may include other agents' files (formatting only).
