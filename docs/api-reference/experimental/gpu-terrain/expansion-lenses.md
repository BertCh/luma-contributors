# Expansion computation lenses

These executable examples use an explicit synthetic 3 × 3 grid. Each graph runs the named public contributor and records real GPU inputs and outputs. The graph rung exposes procedural stages and completeness evidence.

## GPUTerrainFlowDistance

{/* lens:GPUTerrainFlowDistance */}

Follow each D8 receiver and sum affine transition lengths until a stream or terminal cell; cycles and truncated paths stay unresolved.

::lens{scenario=gpu-terrain.terrain-flow-distance rung=graph}

## GPUTerrainStreamNetwork

{/* lens:GPUTerrainStreamNetwork */}

Trace valid stream reaches between headwaters, confluences and outlets, retaining original raster identities and affine cell-center vertices.

::lens{scenario=gpu-terrain.terrain-stream-network rung=graph}
