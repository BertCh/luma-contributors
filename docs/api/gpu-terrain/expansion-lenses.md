# Terrain hydrology recordings

<a id="expansion-computation-lenses"></a>

Import from `@luma-contributors/analysis/gpu-terrain`.

These executable examples use an explicit synthetic 3 × 3 grid. Each graph runs the named public contributor and records real GPU inputs and outputs. The graph rung exposes procedural stages and completeness evidence.

## Recordings by task

Start from the [terrain hydrology contracts](./operators.md#gputerrainflow),
[HAND and fractional drainage](./height-above-drainage.md), or
[terrain processing](./terrain-processing.md). For related operations see
[raster morphology](../gpu-raster/expansion.md#morphology-and-surface-preparation),
[cost surfaces](../gpu-raster/expansion.md#costs-and-travel), and
[geometry and coverage](../gpu-raster/expansion.md#geometry-and-coverage).
Historical capability IDs and class fragments below remain stable.

## GPUTerrainFlowDistance

[Inputs, outputs and limits](./terrain-processing.md).

{/* lens:GPUTerrainFlowDistance */}

Follow each D8 receiver and sum affine transition lengths until a stream or terminal cell; cycles and truncated paths stay unresolved.

::lens{scenario=gpu-terrain.terrain-flow-distance rung=graph}

## GPUTerrainStreamNetwork

[Inputs, outputs and limits](./terrain-processing.md).

{/* lens:GPUTerrainStreamNetwork */}

Trace valid stream reaches between headwaters, confluences and outlets, retaining original raster identities and affine cell-center vertices.

::lens{scenario=gpu-terrain.terrain-stream-network rung=graph}
