# w2-snippets

Script: examples/deck/spatial-analysis-showcase/scripts/check-snippet-imports.mjs (TypeScript compiler API; analysis entries resolve to modules/analysis/src/<entry>/index.ts, everything else through Bundler resolution to published .d.ts).

Checked: 229 import statements, 529 names, 96 files (all strings in both examples' .ts/.tsx plus .md). The explorer has no snippets.
Misses found first run: 144 names in 51 statements in 49 scene files, from '@luma.gl/graph/gpu-dataframe' (73), gpu-raster (68), gpu-pointcloud (3). 81 distinct names, all exist in exactly one analysis entry (dataframe-analysis, raster-analysis, pointcloud-analysis, gpu-spatial-analysis). Statements were rewritten by a one-off script into '@luma-contributors/analysis/<entry>' imports, keeping names still exported by the luma.gl module. Two names were not exported anywhere and were fixed by hand:
- wind-flow: getGPUParticleAdvectionWordParameterValues -> getGPUParticleAdvectionUint32ParameterValues (plus LIC and streamline Uint32 packers).
- wildfire-terrain: getGPUPolygonRasterizationExtentValues -> getGPUPolygonRasterizationParameterValues.
Rerun: "Every snippet import resolves."

Step 3 (prop and output names against the real *.compute.ts): I built a throwaway checker (not kept) that parses every `new GPUX({...})` in snippets and compares keys with the class props type. 356 constructions; the noise (parse artefacts of ${} and union outputs) was triaged by hand. Fixed:
- raster/wind-flow: wordParameters -> uint32Parameters; particle output {speeds, trails}; LIC width/height -> grid: makeGPURasterMetadata; streamlines roundCount -> maximumIterations and output {lines:{ids,status}, lineOffsets, positions, positionCount, status}.
- raster/wildfire-terrain: snippet rewritten to the real shape (grid, parameters, positions, output.status, jumpFloodRefinementIterationCount, slopeUnits, cellCounts).
- dataframe/osm-history-contributors: GPUGroupStatistics output {count, overflow} -> status:{count, overflow}; added 'count' statistic and counts column output.
- geometry/buffers-and-shapes: rectangle clip status now {count, overflow, vertexCount, requiredVertexCount} (count counts lines; requiredCount was dropped by the real code).
- geometry/line-operations: segmentize, chunk, smooth status gain requiredVertexCount (real code uses importPathOutput).
- geometry/polygon-measures: GPUMapColoring output {colors, colorCount, conflictCount, status:{converged, iterationCount}}.
- points/nature-kernel-density: one-liner passed `bandwidth` to the constructor; now shows parameters + getGPUKernelDensityParameterValues({bandwidth, window}, grid).
- networks/retail-network-k-function: constructor lacked `parameters`.
- interpolation/change-of-support: GPUPycnophylactic width/height -> grid.
No snippet constructs the walking-cost or fractional-coverage classes, GPUKMeans (satellite-constellations and hurricane-families already say GPUSpatialKMeans), or the point-cloud statistical filter; no snippet passes `validity` to isolines/isobands/sampling/profile/stretch/reclassify (the only `validity` is a raster band field in viewshed, which the real code still passes). Contours' GPUIsolines/Isobands/Rings snippet only needed the import fix; its keys type-check.

Cannot check: prop values and types, required-but-elided props (snippets abbreviate by design), names used but not imported, helper calls (importToGraph, write) and their arguments, and semantics (what count counts) except where the neighbouring compute code was read by hand.

Commands: `node scripts/check-snippet-imports.mjs` (first run exit 1 with 144 misses; final exit 0); `NODE_OPTIONS=--max-old-space-size=8192 npx tsc --noEmit -p examples/deck/spatial-analysis-showcase` and `-explorer`: 0 errors both; `node scripts/check-story-steps.mjs`: all ok; `node scripts/check-cartography.mjs`: exit 0 (warnings only, unrelated). `npx biome check --write` on the edited files: biome ignores examples/**, so nothing was processed.
A stray empty file raster/ind-flow.scene.ts (my BSD sed mishap) was created and removed.
