var e={slug:`api/gpu-terrain/expansion-lenses`,title:`Expansion computation lenses`,html:`<h1 id="expansion-computation-lenses">Expansion computation lenses</h1>
<p>These executable examples use an explicit synthetic 3 × 3 grid. Each graph runs the named public contributor and records real GPU inputs and outputs. The graph rung exposes procedural stages and completeness evidence.</p>
<h2 id="gputerrainflowdistance">GPUTerrainFlowDistance</h2>
<p>{/* lens:GPUTerrainFlowDistance */}</p>
<p>Follow each D8 receiver and sum affine transition lengths until a stream or terminal cell; cycles and truncated paths stay unresolved.</p>
<p>::lens{scenario=gpu-terrain.terrain-flow-distance rung=graph}</p>
<h2 id="gputerrainstreamnetwork">GPUTerrainStreamNetwork</h2>
<p>{/* lens:GPUTerrainStreamNetwork */}</p>
<p>Trace valid stream reaches between headwaters, confluences and outlets, retaining original raster identities and affine cell-center vertices.</p>
<p>::lens{scenario=gpu-terrain.terrain-stream-network rung=graph}</p>`};export{e as default};