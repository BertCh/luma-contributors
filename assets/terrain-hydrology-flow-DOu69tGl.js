var e=`@luma-contributors/analysis/raster-analysis`,t=`@luma-contributors/analysis/gpu-terrain`,n=`api/gpu-raster/operations-analysis`,r=`api/gpu-terrain/operators`,i=`api/gpu-terrain/terrain-processing`,a={name:`grid`,type:`GPURasterMetadata`,required:!0,description:`Raster grid: width, height, cell size and extent.`},o={name:`flowDirections`,type:`GraphDataView<"uint32">`,required:!0,description:`ESRI D8 codes per cell, from GPUTerrainFlow.`},s={name:`streams`,type:`GraphDataView<"uint32">`,required:!0,description:`Stream mask per cell; nonzero marks a stream cell.`},c={name:`parameters`,type:`GraphDataView<"float32">`,required:!0,description:`Per-frame float32 settings from the matching packer.`},l={name:`maximumIterations`,type:`number | undefined`,description:`Cap on gated pointer-jumping rounds, 1 to 1024.`},u={name:`velocities`,type:`GraphDataView<"float32x2">`,required:!0,description:`Row-major (u, v) vector field, fieldWidth x fieldHeight rows.`},d={name:`fieldWidth`,type:`number`,required:!0,description:`Vector field width in cells.`},f={name:`fieldHeight`,type:`number`,required:!0,description:`Vector field height in cells.`},p={name:`values`,type:`GraphDataView<"float32">`,required:!0,description:`Row-major float32 raster, at least width x height values.`},m={name:`status.converged`,type:`GraphDataView<"uint32">`,description:`Optional one-row flag: 1 when the GPU loop reached its fixpoint.`},h=[{id:`terrain-flow`,symbol:`GPUTerrainFlow`,kind:`class`,importPath:t,title:`Terrain flow routing`,summary:`Fills depressions, routes D8 flow and accumulates upslope area from a DEM, then marks streams.`,explanation:"Start here for any drainage analysis. Optional depression filling feeds steepest-descent directions, an exact accumulation count and a threshold stream mask. Set `flowRouting` to `d-infinity` or an MFD scheme to split flow.",dataTypes:{input:[`terrain`],output:[`terrain`,`raster`]},useCases:[`simulate`,`measure`],domains:[`hydrology`],params:[a,{name:`elevation`,type:`GPURasterBand`,required:!0,description:`Elevation band; invalid cells are skipped.`},{...c,description:`Packed fillEpsilon, streamThreshold and flowExponent.`},{name:`fillDepressions`,type:`boolean | undefined`,default:`false`,description:`Fill sinks before routing flow.`},{name:`resolveFlats`,type:`boolean | undefined`,default:`false`,description:`Route flow across flats without editing elevations.`},{name:`flowRouting`,type:`GPUTerrainFlowRouting | undefined`,default:`'d8'`,description:`'d8', 'd-infinity', 'mfd-freeman' or 'mfd-quinn'.`},{name:`runoff`,type:`GraphDataView<"float32"> | undefined`,description:`Optional per-cell weight for weighted accumulation.`},{name:`accumulationUnits`,type:`GPUTerrainFlowAccumulationUnits | undefined`,default:`'cells'`,description:`Count 'cells' or sum ground 'area'.`}],outputs:[{name:`filledElevation`,type:`GraphDataView<"float32">`,description:`Depression-filled DEM; needs fillDepressions.`},{name:`flowDirections`,type:`GraphDataView<"uint32">`,description:`ESRI D8 code toward the receiver; 0 for outlets.`},{name:`cellClasses`,type:`GraphDataView<"uint32">`,description:`Flat, pit and outlet class per cell.`},{name:`accumulation`,type:`GraphDataView<"float32">`,description:`Upstream weight including the cell itself.`},{name:`streams`,type:`GraphDataView<"uint32">`,description:`1 where accumulation reaches streamThreshold.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainFlow, getGPUTerrainFlowParameterValues} from '@luma-contributors/analysis/gpu-terrain';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {makeGPURasterBand, makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';

const grid = makeGPURasterMetadata({width, height, cellSize: 30});
const graph = new GPUCommandGraph(device, {id: 'flow'});
const parameters = new GPUParameterBuffer(device, {id: 'flow-parameters', format: 'float32', length: 8});

// elevation: float32 GraphDataView on graph; the outputs are views you own
graph.add(new GPUTerrainFlow({
  grid, elevation: makeGPURasterBand(elevation, grid, {id: 'dem'}),
  parameters: parameters.importToGraph(graph), fillDepressions: true,
  output: {flowDirections, accumulation, streams}
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUTerrainFlowParameterValues({fillEpsilon: 0.01, streamThreshold: 500}, grid));
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`flow-accumulation`,seed:301,caption:`Upslope area gathers into a stream network.`},related:[`terrain-watersheds`,`terrain-stream-order`,`terrain-height-above-drainage`,`terrain-hydrologic-indices`,`terrain-stream-network`],lensScenario:`gpu-terrain.terrain-flow`,reference:r,maturity:`stable`},{id:`terrain-watersheds`,symbol:`GPUTerrainWatersheds`,kind:`class`,importPath:t,title:`Watershed delineation`,summary:`Labels every cell with the outlet or pour point its D8 flow path drains to.`,explanation:`Carves a DEM into catchments, either by natural outlets or by your own pour points.`,dataTypes:{input:[`terrain`],output:[`raster`]},useCases:[`summarize`,`simulate`],domains:[`hydrology`],params:[a,o,{name:`pourPoints`,type:`GraphDataView<"uint32"> | undefined`,description:`Pour point cell indices (row * width + column).`},l],outputs:[{name:`labels`,type:`GraphDataView<"uint32">`,description:`Pour point index or terminal cell id; 0xffffffff for none.`},m],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainWatersheds} from '@luma-contributors/analysis/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'watersheds'});

// flowDirections: GPUTerrainFlow output; pourPoints: uint32 cell indices
graph.add(new GPUTerrainWatersheds({
  grid, flowDirections, pourPoints,
  output: {labels, status: {converged}}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`flow-accumulation`,seed:302,caption:`Each color is the area draining to one outlet.`},related:[`terrain-flow`,`terrain-catchment-summary`,`terrain-stream-order`],lensScenario:`gpu-terrain.terrain-watersheds`,reference:r,maturity:`stable`},{id:`terrain-stream-order`,symbol:`GPUTerrainStreamOrder`,kind:`class`,importPath:t,title:`Strahler stream order`,summary:`Assigns Strahler order to every stream cell: headwaters are 1, equal tributaries raise the order.`,explanation:`Ranks rivers by branching hierarchy. It reads D8 directions and a stream mask, then ranks the receiver forest on the GPU with an Euler tour.`,dataTypes:{input:[`terrain`],output:[`raster`]},useCases:[`classify`,`measure`],domains:[`hydrology`],params:[a,o,s,l],outputs:[{name:`streamOrder`,type:`GraphDataView<"uint32">`,description:`0 for non-stream cells, at least 1 on streams.`},m],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainStreamOrder} from '@luma-contributors/analysis/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'stream-order'});

// flowDirections and streams come from GPUTerrainFlow
graph.add(new GPUTerrainStreamOrder({
  grid, flowDirections, streams,
  output: {streamOrder, status: {converged}}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`network`,seed:303,caption:`Line weight grows with Strahler order.`},related:[`terrain-flow`,`terrain-stream-network`,`terrain-watersheds`],lensScenario:`gpu-terrain.terrain-stream-order`,reference:r,maturity:`stable`},{id:`terrain-height-above-drainage`,symbol:`GPUTerrainHeightAboveDrainage`,kind:`class`,importPath:t,title:`Height above nearest drainage`,summary:`Computes HAND: each cell's elevation above the stream cell its D8 path reaches first.`,explanation:`Best for flood-inundation screening and wetland mapping, where relative height beats absolute elevation. Feed it the filled elevation, D8 directions and a stream mask.`,dataTypes:{input:[`terrain`],output:[`raster`]},useCases:[`measure`,`simulate`],domains:[`hydrology`],params:[a,{name:`elevation`,type:`GPURasterBand`,required:!0,description:`Routing surface, normally the filled DEM.`},o,s,{name:`output`,type:`GPUTerrainHeightAboveDrainageOutput`,required:!0,description:`Caller-owned output views.`},l],outputs:[{name:`heightAboveDrainage`,type:`GraphDataView<"float32">`,description:`Elevation minus drainage elevation; 0 on streams.`},{name:`drainageCells`,type:`GraphDataView<"uint32">`,description:`Stream cell index each cell drains to.`},m],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainHeightAboveDrainage} from '@luma-contributors/analysis/gpu-terrain';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';

const graph = new GPUCommandGraph(device, {id: 'hand'});

// filledElevation, flowDirections, streams: GPUTerrainFlow outputs
graph.add(new GPUTerrainHeightAboveDrainage({
  grid,
  elevation: makeGPURasterBand(filledElevation, grid, {id: 'filled'}),
  flowDirections, streams,
  output: {heightAboveDrainage}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`terrain`,seed:304,caption:`Low-lying cells near streams stand out.`},related:[`terrain-flow`,`terrain-hydrologic-indices`,`terrain-flow-distance`],lensScenario:`gpu-terrain.terrain-height-above-drainage`,reference:r,maturity:`stable`},{id:`terrain-hydrologic-indices`,symbol:`GPUTerrainHydrologicIndices`,kind:`class`,importPath:t,title:`Wetness and stream power indices`,summary:`Derives slope, specific catchment area, the topographic wetness index and stream power from accumulation.`,explanation:"Finds where soil stays wet or where flow can erode. It combines upslope area with local slope: wetness is `ln(a / tan(beta))`, stream power is `a * tan(beta)`.",dataTypes:{input:[`terrain`,`raster`],output:[`raster`]},useCases:[`measure`,`model`],domains:[`hydrology`,`ecology`],params:[a,{name:`elevation`,type:`GPURasterBand`,required:!0,description:`Routing surface the slope is measured on.`},{name:`accumulation`,type:`GraphDataView<"float32">`,required:!0,description:`Upslope area in m^2, including the cell itself.`},{...c,description:`Packed minimumSlope from the packer.`},{name:`cellSizeMode`,type:`GPURasterCellSizeMode | undefined`,default:`'uniform'`,description:`Cell size: uniform, web-mercator or geographic.`},{name:`output`,type:`GPUTerrainHydrologicIndicesOutput`,required:!0,description:`Pick any of the four index views.`}],outputs:[{name:`slope`,type:`GraphDataView<"float32">`,description:`tan(beta) after the minimumSlope clamp.`},{name:`specificCatchmentArea`,type:`GraphDataView<"float32">`,description:`Contributing area per unit contour width, in meters.`},{name:`wetnessIndex`,type:`GraphDataView<"float32">`,description:`Topographic wetness index.`},{name:`streamPowerIndex`,type:`GraphDataView<"float32">`,description:`Stream power index in meters.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainHydrologicIndices, getGPUTerrainHydrologicIndicesParameterValues} from '@luma-contributors/analysis/gpu-terrain';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';

const graph = new GPUCommandGraph(device, {id: 'indices'});
const parameters = new GPUParameterBuffer(device, {id: 'index-parameters', format: 'float32', length: 8});

// accumulation: GPUTerrainFlow output with accumulationUnits: 'area'
graph.add(new GPUTerrainHydrologicIndices({
  grid, accumulation, parameters: parameters.importToGraph(graph),
  elevation: makeGPURasterBand(filledElevation, grid, {id: 'filled'}),
  output: {wetnessIndex, streamPowerIndex}
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUTerrainHydrologicIndicesParameterValues({minimumSlope: 0.001}, grid));
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`raster`,seed:305,caption:`Wetness peaks in flat, high-accumulation valleys.`},related:[`terrain-flow`,`terrain-height-above-drainage`],lensScenario:`gpu-terrain.terrain-hydrologic-indices`,reference:r,maturity:`stable`},{id:`terrain-flow-distance`,symbol:`GPUTerrainFlowDistance`,kind:`class`,importPath:t,title:`Downstream flow distance`,summary:`Measures the D8 path length from each cell downstream to a stream or outlet.`,explanation:`Best for travel time to the channel, or flow-path length weighted by your own edge costs. Invalid directions and bounded-search failures are preserved, never guessed.`,dataTypes:{input:[`terrain`],output:[`raster`]},useCases:[`measure`,`route`],domains:[`hydrology`],params:[a,o,{name:`streams`,type:`GraphDataView<"uint32"> | undefined`,description:`Nonzero cells stop tracing before their outgoing edge.`},{name:`transitionCosts`,type:`GraphDataView<"float32"> | undefined`,description:`Cost of each cell's outgoing edge, in your units.`},{name:`maximumSteps`,type:`number | undefined`,description:`Bound on each path walk.`}],outputs:[{name:`distance`,type:`GraphDataView<"float32">`,description:`Accumulated path length or cost to the end cell.`},{name:`validity`,type:`GraphDataView<"uint32">`,description:`1 where the distance is resolved.`},{name:`terminalCells`,type:`GraphDataView<"uint32">`,description:`Cell where each path ends.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainFlowDistance} from '@luma-contributors/analysis/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'flow-distance'});

// stop each path at the first stream cell
graph.add(new GPUTerrainFlowDistance({
  grid, flowDirections, streams, maximumSteps: 4096,
  output: {distance, validity, terminalCells}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`isochrones`,seed:306,caption:`Distance to the channel grows up the hillslopes.`},related:[`terrain-flow`,`terrain-height-above-drainage`,`terrain-branching-flow`],lensScenario:`gpu-terrain.terrain-flow-distance`,reference:`api/gpu-raster/expansion`,maturity:`stable`},{id:`terrain-stream-network`,symbol:`GPUTerrainStreamNetwork`,kind:`class`,importPath:t,title:`Stream network reaches`,summary:`Extracts directed stream reaches between headwaters, confluences and terminals as compact polylines.`,explanation:`Turns a stream mask into vector lines you can draw or analyze as a network. Each reach keeps its source and target cell ids so attributes can be joined back to the raster.`,dataTypes:{input:[`terrain`],output:[`lines`,`network`]},useCases:[`transform`,`measure`],domains:[`hydrology`,`cartography`],params:[a,o,s,{name:`maximumSteps`,type:`number | undefined`,description:`Bound on each downstream traversal.`},{name:`output`,type:`{sourceCells, targetCells}`,required:!0,description:`Reach, vertex and count views you own.`}],outputs:[{name:`sourceCells`,type:`GraphDataView<"uint32">`,description:`Raster cell where each reach starts.`},{name:`targetCells`,type:`GraphDataView<"uint32">`,description:`Raster cell where each reach ends.`},{name:`vertexOffsets`,type:`GraphDataView<"uint32">`,description:`CSR offsets into positions, one line per reach.`},{name:`positions`,type:`GraphDataView<"float32x2">`,description:`Reach polyline vertices.`},{name:`lengths`,type:`GraphDataView<"float32">`,description:`Length of each reach.`},{name:`reachCount`,type:`GraphDataView<"uint32">`,description:`Number of reaches.`},{name:`vertexCount`,type:`GraphDataView<"uint32">`,description:`Number of polyline vertices.`},{name:`status, unresolvedCount`,type:`mixed views`,description:`Further result fields; see the reference.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainStreamNetwork} from '@luma-contributors/analysis/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'stream-network'});

// capacities of the output views bound how many reaches and vertices are kept
graph.add(new GPUTerrainStreamNetwork({
  grid, flowDirections, streams,
  output: {
    sourceCells, targetCells, vertexOffsets, positions, lengths,
    reachCount, vertexCount, status: {overflow}, unresolvedCount
  }
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`network`,seed:307,caption:`Reaches split at every confluence.`},related:[`terrain-stream-order`,`terrain-flow`,`terrain-watersheds`],lensScenario:`gpu-terrain.terrain-stream-network`,reference:`api/gpu-raster/expansion`,maturity:`stable`},{id:`terrain-flow-field`,symbol:`GPUTerrainFlowField`,kind:`class`,importPath:t,title:`Terrain-following wind field`,summary:`Deflects a uniform horizontal wind over a DEM so it flows around and over relief.`,explanation:`Makes a plausible vector field from elevation alone. The output (u, v) rows plug straight into particle advection, streamlines or line integral convolution.`,dataTypes:{input:[`terrain`],output:[`raster`]},useCases:[`simulate`],domains:[`hydrology`,`cartography`],params:[a,{name:`elevation`,type:`GPURasterBand`,required:!0,description:`Elevation band, buffer or texture.`},{...c,description:`Packed wind and verticalExaggeration from the packer.`},{name:`cellSizeMode`,type:`GPURasterCellSizeMode | undefined`,default:`'uniform'`,description:`Cell size: uniform, web-mercator or geographic.`},{name:`output`,type:`GPUTerrainFlowFieldOutput`,required:!0,description:`Holds the velocities view.`}],outputs:[{name:`velocities`,type:`GraphDataView<"float32x2">`,description:`(u, v) per cell; NaN where the DEM is invalid.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainFlowField, getGPUTerrainFlowFieldParameterValues} from '@luma-contributors/analysis/gpu-terrain';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';

const graph = new GPUCommandGraph(device, {id: 'wind'});
const parameters = new GPUParameterBuffer(device, {id: 'wind-parameters', format: 'float32', length: 8});

graph.add(new GPUTerrainFlowField({
  grid, parameters: parameters.importToGraph(graph),
  elevation: makeGPURasterBand(elevation, grid, {id: 'dem'}),
  output: {velocities}
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUTerrainFlowFieldParameterValues({wind: [8, 0], verticalExaggeration: 2}, grid));
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`flows`,seed:308,caption:`Wind bends around ridges and speeds over summits.`},related:[`particle-advection`,`streamlines`,`line-integral-convolution`],lensScenario:`gpu-terrain.terrain-flow-field`,reference:r,maturity:`stable`},{id:`terrain-flow-fractions`,symbol:`GPUTerrainFlowFractions`,kind:`class`,importPath:t,title:`Branching flow fractions`,summary:`Writes the eight per-neighbor flow fractions of D-infinity or MFD routing for branching consumers.`,explanation:"Best when downstream steps need explicit splits instead of one receiver. It reuses the exact routing kernel of `GPUTerrainFlow`, so the fractions match its accumulation.",dataTypes:{input:[`terrain`],output:[`raster`]},useCases:[`simulate`],domains:[`hydrology`],params:[a,{name:`elevation`,type:`GraphDataView<"float32">`,required:!0,description:`Same routing surface as GPUTerrainFlow.`},o,{...c,description:`Same parameters as GPUTerrainFlow, incl. MFD exponent.`},{name:`routing`,type:`TerrainFlowMultipleRouting`,required:!0,description:`'d-infinity', 'mfd-freeman' or 'mfd-quinn'.`},{name:`cellSizeMode`,type:`GPURasterCellSizeMode | undefined`,default:`'uniform'`,description:`Cell size: uniform, web-mercator or geographic.`}],outputs:[{name:`fractions`,type:`GraphDataView<"float32">`,description:`Eight fractions per cell, east first, clockwise.`},{name:`edgeLengths`,type:`GraphDataView<"float32">`,description:`Ground length of each of the eight edges.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainFlowFractions} from '@luma-contributors/analysis/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'fractions'});

// elevation, flowDirections and parameters are the ones given to GPUTerrainFlow
graph.add(new GPUTerrainFlowFractions({
  grid, elevation: filledElevation, flowDirections,
  parameters: flowParameters, routing: 'd-infinity',
  output: {fractions, edgeLengths}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`flow-accumulation`,seed:309,caption:`Flow splits across neighbors on divergent slopes.`},related:[`terrain-branching-flow`,`terrain-flow`,`terrain-tiled-flow`],lensScenario:`gpu-terrain.terrain-flow-fractions`,reference:i,maturity:`stable`},{id:`terrain-branching-flow`,symbol:`GPUTerrainBranchingFlow`,kind:`class`,importPath:t,title:`Branching flow transport`,summary:`Propagates mass, flow distance or outlet probability through a DAG of eight-neighbor flow fractions.`,explanation:"Pick a `mode`: `transport` accumulates injected mass, the distance modes give expected, minimum or maximum path length, and `fractional-catchment` gives the chance each cell drains to an outlet. Results are deterministic.",dataTypes:{input:[`terrain`],output:[`raster`]},useCases:[`simulate`,`measure`],domains:[`hydrology`],params:[a,{name:`fractions`,type:`GraphDataView<"float32">`,required:!0,description:`Eight fractions per cell, east first, clockwise.`},{name:`values`,type:`GraphDataView<"float32">`,required:!0,description:`Injection, or outlet membership (0 or 1) for catchments.`},{name:`mode`,type:`"transport" | "expected-distance" | "minimum-distance" | "maximum-distance" | "fractional-catchment"`,required:!0,description:`Recurrence applied along the DAG.`},{name:`edgeLengths`,type:`GraphDataView<"float32"> | undefined`,description:`Eight edge lengths per cell, for the distance modes.`},{name:`maximumIterations`,type:`number | undefined`,description:`GPU sweep budget, 1 to 1024.`},{name:`maximumWalkLength`,type:`number | undefined`,default:`32`,description:`Bounded successor walk after each publication.`}],outputs:[{name:`values`,type:`GraphDataView<"float32">`,description:`Result per cell; NaN when unresolved.`},{name:`status`,type:`GraphDataView<"uint32">`,description:`0 invalid, 1 resolved, 2 cycle or budget exhausted.`},{name:`unresolvedCount`,type:`GraphDataView<"uint32">`,description:`One-row count of unresolved cells.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainBranchingFlow} from '@luma-contributors/analysis/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'branching'});

// fractions from GPUTerrainFlowFractions; rainfall: one injected value per cell
graph.add(new GPUTerrainBranchingFlow({
  grid, fractions, values: rainfall, mode: 'transport',
  output: {values: transported, status, unresolvedCount}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`flow-accumulation`,seed:310,caption:`Mass fans out and rejoins along multiple paths.`},related:[`terrain-flow-fractions`,`terrain-tiled-flow`,`terrain-flow-distance`],lensScenario:`gpu-terrain.terrain-branching-flow`,reference:i,maturity:`stable`},{id:`terrain-tiled-flow`,symbol:`GPUTerrainTiledFlow`,kind:`class`,importPath:t,title:`Tiled flow accumulation`,summary:`Accumulates flow across tiles of a large DEM by exchanging owned cells in one resident global graph.`,explanation:`Best when the DEM arrives as tiles with halos and flow must cross tile seams. Owned rectangles keep accumulation and reach ids exact across seams.`,dataTypes:{input:[`terrain`],output:[`raster`]},useCases:[`simulate`,`summarize`],domains:[`hydrology`],params:[a,{name:`tiles`,type:`readonly GPUTerrainFlowTile[]`,required:!0,description:`Disjoint owned rectangles with their views; must cover the grid.`},{name:`routing`,type:`"d8" | "branching" | undefined`,default:`'d8'`,description:`D8, or branching with prepared fractions.`},{name:`maximumIterations`,type:`number | undefined`,description:`GPU DAG sweep budget, 1 to 1024.`}],outputs:[{name:`tiles[].output.accumulation`,type:`GraphDataView<"float32">`,description:`Global upstream weight for each tile.`},{name:`tiles[].output.status`,type:`GraphDataView<"uint32">`,description:`0 halo, 1 resolved, 2 cycle or budget exhausted.`},{name:`tiles[].output.reachIds`,type:`GraphDataView<"uint32">`,description:`Source cell of each stream chain; 0xffffffff otherwise.`},{name:`status`,type:`{invalidCount?, converged?, iterationLimitReached?}`,description:`Optional aggregate diagnostics over global owned cells, excluding halos.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainTiledFlow} from '@luma-contributors/analysis/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'tiled-flow'});

// one entry per tile: its buffer origin and size, owned rectangle and views
const tiles = tileViews.map(tile => ({
  column: tile.column, row: tile.row, width: tile.width, height: tile.height,
  owned: tile.owned, flowDirections: tile.flowDirections, runoff: tile.runoff,
  streams: tile.streams,
  output: {accumulation: tile.accumulation, status: tile.status, reachIds: tile.reachIds}
}));
graph.add(new GPUTerrainTiledFlow({grid, routing: 'd8', tiles}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`flow-accumulation`,seed:311,caption:`Streams stay continuous across tile seams.`},related:[`terrain-flow`,`terrain-branching-flow`,`terrain-flow-fractions`],lensScenario:`gpu-terrain.terrain-tiled-flow`,reference:i,maturity:`stable`},{id:`terrain-breach`,symbol:`GPUTerrainBreach`,kind:`class`,importPath:t,title:`Depression breaching`,summary:`Cuts the cheapest bounded channel from each pit to an outlet, with optional explicit stream burning.`,explanation:`Best instead of filling when you want to preserve terrain: it minimizes total excavation depth for the pits you name. Simultaneous edits take the minimum elevation, so results do not depend on thread order.`,dataTypes:{input:[`terrain`],output:[`terrain`]},useCases:[`transform`,`simulate`],domains:[`hydrology`],params:[a,{name:`elevation`,type:`GraphDataView<"float32">`,required:!0,description:`Canonical float32 elevation surface.`},{name:`pitCells`,type:`GraphDataView<"uint32">`,required:!0,description:`Explicit pit cell indices; nothing is filled or burned implicitly.`},{name:`maximumSteps`,type:`number`,required:!0,description:`Longest D8 breach path per pit.`},{name:`maximumCutDepth`,type:`number`,required:!0,description:`Deepest excavation allowed along a path.`},{name:`elevationDecrement`,type:`number`,required:!0,description:`Positive drop per path step, in elevation units.`},{name:`burnDepth`,type:`GraphDataView<"float32"> | undefined`,description:`Explicit burn depth per cell, applied after breaching.`}],outputs:[{name:`elevation`,type:`GraphDataView<"float32">`,description:`Edited elevation surface.`},{name:`cutDepth`,type:`GraphDataView<"float32">`,description:`Depth removed from each cell.`},{name:`modificationMask`,type:`GraphDataView<"uint32">`,description:`Bit 1 breach, 2 burn, 4 invalid burn depth.`},{name:`pathCells`,type:`GraphDataView<"uint32">`,description:`Breach path cells per pit, pit first.`},{name:`pathLengths`,type:`GraphDataView<"uint32">`,description:`Path length per pit.`},{name:`pitStatus`,type:`GraphDataView<"uint32">`,description:`0 invalid, 1 breached, 2 no outlet within bounds.`},{name:`unresolvedCount`,type:`GraphDataView<"uint32">`,description:`Pits left unresolved.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainBreach} from '@luma-contributors/analysis/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'breach'});

// pitCells: uint32 cell indices of the depressions to open
graph.add(new GPUTerrainBreach({
  grid, elevation, pitCells,
  maximumSteps: 32, maximumCutDepth: 5, elevationDecrement: 0.001,
  output: {
    elevation: breached, cutDepth, modificationMask,
    pathCells, pathLengths, pitStatus, unresolvedCount
  }
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`terrain`,seed:312,caption:`A narrow cut opens the pit to its outlet.`},related:[`terrain-depression-hierarchy`,`terrain-flow`],lensScenario:`gpu-terrain.terrain-breach`,reference:i,maturity:`stable`},{id:`terrain-depression-hierarchy`,symbol:`GPUTerrainDepressionHierarchy`,kind:`class`,importPath:t,title:`Depression hierarchy`,summary:`Builds the merge tree of nested depressions with area and volume curves at chosen elevation stages.`,explanation:`Shows which sinks fill first, where they spill and how much water they hold. The build is serial on the GPU, so keep grids moderate.`,dataTypes:{input:[`terrain`],output:[`table`,`raster`]},useCases:[`measure`,`simulate`],domains:[`hydrology`],params:[a,{name:`elevation`,type:`GraphDataView<"float32">`,required:!0,description:`Float32 elevation per cell.`},{name:`outlets`,type:`GraphDataView<"uint32">`,required:!0,description:`Explicit ocean or drain cells.`},{name:`stages`,type:`GraphDataView<"float32">`,required:!0,description:`Query elevations for area and volume curves.`},{name:`nodeCapacity`,type:`number`,required:!0,description:`Maximum merge-tree nodes stored.`}],outputs:[{name:`nodes`,type:`GraphDataView<"uint32">`,description:`Parent, minimum cell, outlet and validity per node.`},{name:`levels`,type:`GraphDataView<"float32">`,description:`Birth and merge elevation per node.`},{name:`cellNodes`,type:`GraphDataView<"uint32">`,description:`Activation node that owns each cell.`},{name:`curves`,type:`GraphDataView<"float32">`,description:`Inundated area and volume per node and stage.`},{name:`status`,type:`GraphDataView<"uint32">`,description:`Stored count, required count and overflow flag.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainDepressionHierarchy} from '@luma-contributors/analysis/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'depressions'});

// outlets: boundary or ocean cells; stages: e.g. 10 m steps above the minimum
graph.add(new GPUTerrainDepressionHierarchy({
  grid, elevation, outlets, stages,
  nodeCapacity: grid.width * grid.height,
  output: {nodes, levels, cellNodes, curves, status}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`terrain`,seed:313,caption:`Basins merge into one as the water level rises.`},related:[`terrain-breach`,`terrain-flow`,`terrain-catchment-summary`],lensScenario:`gpu-terrain.terrain-depression-hierarchy`,reference:i,maturity:`stable`},{id:`terrain-catchment-summary`,symbol:`GPUTerrainCatchmentSummary`,kind:`class`,importPath:t,title:`Catchment summary`,summary:`Summarizes each basin: area, elevation statistics, drainage density, hypsometry and stream profiles.`,explanation:`Describes watersheds after delineation. It reduces elevation per basin and measures downstream stream profiles by pointer jumping.`,dataTypes:{input:[`terrain`,`raster`],output:[`table`]},useCases:[`summarize`,`measure`],domains:[`hydrology`],params:[a,{name:`elevation`,type:`GraphDataView<"float32">`,required:!0,description:`Float32 elevation per cell.`},{name:`basinIds`,type:`GraphDataView<"uint32">`,required:!0,description:`Dense zero-based basin ids; 0xffffffff excludes.`},{name:`basinCount`,type:`number`,required:!0,description:`Number of basins.`},{name:`streams`,type:`GraphDataView<"uint32">`,required:!0,description:`Resolved 0/1 stream mask.`},o,{name:`histogramBins`,type:`number`,required:!0,description:`Hypsometric curve bins per basin.`},{name:`maximumSteps`,type:`number | undefined`,description:`Bound on profile walks.`}],outputs:[{name:`statistics`,type:`GraphDataView<"float32">`,description:`Per basin: area, mean, min, max, stream length, density.`},{name:`hypsometry`,type:`GraphDataView<"float32">`,description:`Cumulative area by normalized elevation bin.`},{name:`profiles`,type:`GraphDataView<"float32">`,description:`Per cell: stream distance, elevation, gradient.`},{name:`profileStatus`,type:`GraphDataView<"uint32">`,description:`0 excluded, 1 complete, 2 cycle or exhausted walk.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainCatchmentSummary} from '@luma-contributors/analysis/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'catchments'});

// basinIds: dense watershed labels, e.g. remapped from GPUTerrainWatersheds
graph.add(new GPUTerrainCatchmentSummary({
  grid, elevation, basinIds, basinCount, streams, flowDirections,
  histogramBins: 20,
  output: {statistics, hypsometry, profiles, profileStatus}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`histogram`,seed:314,caption:`Hypsometric curve per catchment.`},related:[`terrain-watersheds`,`terrain-stream-network`,`terrain-depression-hierarchy`],lensScenario:`gpu-terrain.terrain-catchment-summary`,reference:i,maturity:`stable`},{id:`terrain-dem-coregistration`,symbol:`GPUTerrainDEMCoregistration`,kind:`class`,importPath:t,title:`DEM coregistration`,summary:`Estimates a robust vertical shift between two DEMs on stable ground and returns the corrected difference.`,explanation:"Best before change detection, to remove a constant bias between epochs. It fits the median difference over a stable mask, like xDEM's `VerticalShift`.",dataTypes:{input:[`terrain`],output:[`raster`]},useCases:[`detect-change`],domains:[`hydrology`,`remote-sensing`],params:[a,{name:`reference`,type:`GraphDataView<"float32">`,required:!0,description:`Reference DEM.`},{name:`moving`,type:`GraphDataView<"float32">`,required:!0,description:`DEM to align to the reference.`},{name:`stableMask`,type:`GraphDataView<"uint32">`,required:!0,description:`Nonzero where terrain is assumed unchanged.`},{name:`differenceVariance`,type:`GraphDataView<"float32">`,required:!0,description:`Known variance of the raw difference per cell.`},{name:`correlationLength`,type:`number | undefined`,default:`0`,description:`Exponential covariance range; zero means independent.`}],outputs:[{name:`fit`,type:`GraphDataView<"float32">`,description:`Median shift, residual NMAD, stable count, bias error.`},{name:`differences`,type:`GraphDataView<"float32">`,description:`Corrected difference and standard deviation per cell.`},{name:`validity`,type:`GraphDataView<"uint32">`,description:`1 where the difference is valid.`},{name:`volume`,type:`GraphDataView<"float32">`,description:`Integrated change, its deviation and valid area.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainDEMCoregistration} from '@luma-contributors/analysis/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'coregistration'});

// stableMask: 1 on bedrock or roads, 0 on glaciers, slides and vegetation
graph.add(new GPUTerrainDEMCoregistration({
  grid, reference, moving, stableMask, differenceVariance,
  correlationLength: 200,
  output: {fit, differences, validity, volume}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`raster`,seed:315,caption:`Elevation change after removing the vertical bias.`},related:[`terrain-flow`,`terrain-breach`],lensScenario:`gpu-terrain.terrain-dem-coregistration`,reference:i,maturity:`stable`},{id:`cost-distance`,symbol:`GPUCostDistance`,kind:`class`,importPath:e,title:`Cost distance`,summary:`Computes least accumulated travel cost from source cells across a friction raster, with back-links and bands.`,explanation:`Best for accessibility, habitat connectivity and least-cost corridors. Eight-connected moves cost distance times mean friction; invalid or negative friction is a barrier.`,dataTypes:{input:[`raster`],output:[`raster`]},useCases:[`route`,`measure`],domains:[`ecology`,`transport`,`hydrology`],params:[a,{name:`friction`,type:`GPURasterBand`,required:!0,description:`Cost per ground meter; invalid or negative is a barrier.`},{...c,description:`Packed costLimit from getGPUCostDistanceParameterValues.`},{name:`sources`,type:`GraphDataView<"uint32"> | undefined`,description:`Source cell indices (row * width + column).`},{name:`sourceMask`,type:`GraphDataView<"uint32"> | undefined`,description:`Per-cell mask; nonzero cells are sources at cost 0.`},{name:`costBreaks`,type:`GraphDataView<"float32"> | undefined`,description:`Ascending isochrone breaks for the bands output.`},{name:`maximumIterations`,type:`number | undefined`,description:`Gated relaxation rounds before giving up.`},{name:`output`,type:`GPUCostDistanceOutput`,required:!0,description:`Caller-owned output views.`}],outputs:[{name:`costs`,type:`GraphDataView<"float32">`,description:`Accumulated cost; Infinity when unreached.`},{name:`backLinks`,type:`GraphDataView<"uint32">`,description:`D8 code toward the next cell on the least-cost path.`},{name:`bands`,type:`GraphDataView<"uint32">`,description:`Isochrone band per cell, from costBreaks.`},{name:`bandCounts`,type:`GraphDataView<"uint32">`,description:`Cell count per band.`},m],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUCostDistance, getGPUCostDistanceParameterValues, makeGPURasterMetadata}
  from '@luma-contributors/analysis/raster-analysis';

const grid = makeGPURasterMetadata({width, height, cellSize: 10});
const graph = new GPUCommandGraph(device, {id: 'cost'});
const parameters = new GPUParameterBuffer(device, {id: 'cost-parameters', format: 'float32', length: 8});

// friction: GPURasterBand; sources: uint32 cell indices; the outputs are views you own
graph.add(new GPUCostDistance({
  grid, friction, parameters: parameters.importToGraph(graph), sources,
  maximumIterations: 512, output: {costs, backLinks, status: {converged}}
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUCostDistanceParameterValues({costLimit: 5000}, grid));
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`isochrones`,seed:316,caption:`Cost rings bend around high-friction terrain.`},related:[`cost-distance-path`,`distance-field`,`isolines`],lensScenario:`gpu-raster.cost-distance`,reference:n,maturity:`stable`},{id:`cost-distance-path`,symbol:`GPUCostDistancePath`,kind:`class`,importPath:e,title:`Least-cost path`,summary:`Traces the least-cost path from a target cell back to its source by following cost-distance back-links.`,explanation:"Best after `GPUCostDistance` to pull out one route: a corridor, a pipeline or an evacuation path. The target is a per-frame value, so you can move it without recompiling.",dataTypes:{input:[`raster`],output:[`lines`]},useCases:[`route`],domains:[`ecology`,`transport`],params:[a,{name:`backLinks`,type:`GraphDataView<"uint32">`,required:!0,description:`Back-link raster from GPUCostDistance.`},{name:`target`,type:`GraphDataView<"uint32">`,required:!0,description:`One-row target cell index (row * width + column).`},{name:`output`,type:`GPUCostDistancePathOutput`,required:!0,description:`Compact path output with optional status.`}],outputs:[{name:`path`,type:`GPUCompactOutput`,description:`Cell indices from target to source, with a count.`},{name:`status`,type:`GPUCostDistancePathStatus`,description:`Optional one-row status words.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCostDistancePath} from '@luma-contributors/analysis/raster-analysis';

const graph = new GPUCommandGraph(device, {id: 'path'});

// backLinks: GPUCostDistance output; target: one uint32 row, e.g. an imported GPUParameterBuffer
graph.add(new GPUCostDistancePath({
  grid, backLinks, target,
  output: {path}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`shortest-path`,seed:317,caption:`The route follows the cheapest valley.`},related:[`cost-distance`,`distance-field`],lensScenario:`gpu-raster.cost-distance-path`,reference:n,maturity:`stable`},{id:`distance-field`,symbol:`GPUDistanceField`,kind:`class`,importPath:e,title:`Distance field`,summary:`Computes Euclidean distance and nearest-seed allocation (Voronoi zones) on a raster from seed points or a mask.`,explanation:"Best for proximity surfaces, buffers-as-rasters and straight-line catchments. `exact` mode is an exact distance transform; `jump-flood` trades exactness for speed.",dataTypes:{input:[`points`,`raster`],output:[`raster`]},useCases:[`measure`,`route`],domains:[`urban`,`ecology`],params:[a,{...c,description:`Packed maximumDistance and localOrigin from the packer.`},{name:`seedPositions`,type:`GraphDataView<"float32x2"> | undefined`,description:`Seed points in ground coordinates, snapped to cells.`},{name:`seedIds`,type:`GraphDataView<"uint32"> | undefined`,default:`row index`,description:`Seed id per point.`},{name:`seedMask`,type:`GraphDataView<"uint32"> | undefined`,description:`Per-cell seed raster; nonzero cells are seeds.`},{name:`mode`,type:`GPUDistanceFieldMode | undefined`,description:`'exact' or 'jump-flood'.`},{name:`output`,type:`GPUDistanceFieldOutput`,required:!0,description:`Caller-owned output views.`}],outputs:[{name:`distances`,type:`GraphDataView<"float32">`,description:`Ground distance to the nearest seed; Infinity if none.`},{name:`allocation`,type:`GraphDataView<"uint32">`,description:`Id of the nearest seed (Voronoi zones).`},{name:`nearestCells`,type:`GraphDataView<"uint32">`,description:`Cell index of the nearest seed.`},{name:`withinDistance`,type:`GraphDataView<"uint32">`,description:`1 when a seed lies within maximumDistance.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUDistanceField, getGPUDistanceFieldParameterValues} from '@luma-contributors/analysis/raster-analysis';

const graph = new GPUCommandGraph(device, {id: 'distance'});
const parameters = new GPUParameterBuffer(device, {id: 'df-parameters', format: 'float32', length: 8});

// seedPositions: float32x2 view of stations in ground coordinates
graph.add(new GPUDistanceField({
  grid, parameters: parameters.importToGraph(graph), seedPositions, seedCount,
  output: {distances, allocation, withinDistance}
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUDistanceFieldParameterValues({maximumDistance: 500}, grid));
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`cells`,seed:318,caption:`Voronoi zones and distance rings from seed points.`},related:[`cost-distance`,`isolines`,`isobands`],lensScenario:`gpu-raster.distance-field`,reference:n,maturity:`stable`},{id:`isolines`,symbol:`GPUIsolines`,kind:`class`,importPath:e,title:`Isolines`,summary:`Extracts marching-squares contour lines of a raster at several levels, as segments and stitched polylines.`,explanation:`Best for elevation contours, isochrone edges or any threshold line on a surface. Levels and extent are per-frame buffer writes, so sliding a level never recompiles.`,dataTypes:{input:[`raster`,`terrain`],output:[`lines`]},useCases:[`transform`,`classify`],domains:[`cartography`,`hydrology`],params:[a,p,{name:`levels`,type:`GraphDataView<"float32">`,required:!0,description:`Level values; the first levelCount rows are active.`},{...c,description:`Packed levelCount, extent and localOrigin.`},{name:`mask`,type:`GraphDataView<"uint32"> | undefined`,description:`Nonzero-is-valid mask over the raster.`},{name:`noDataValue`,type:`number | undefined`,description:`Finite sentinel treated as nodata.`},{name:`output`,type:`GPUIsolinesOutput`,required:!0,description:`Segments, optional polylines and status.`}],outputs:[{name:`segments`,type:`GraphDataView<"float32x4">`,description:`Segment endpoints (x0, y0, x1, y1), ordered by level.`},{name:`segmentLevels`,type:`GraphDataView<"uint32">`,description:`Level index of each segment.`},{name:`polylines`,type:`GPUIsolinesPolylineOutput`,description:`Optional stitched vertices, offsets, levels and closed flags.`},{name:`status.overflow`,type:`GraphDataView<"uint32">`,description:`1 when the segment capacity was exceeded.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUIsolines, getGPUIsolinesParameterValues} from '@luma-contributors/analysis/raster-analysis';

const graph = new GPUCommandGraph(device, {id: 'isolines'});
const parameters = new GPUParameterBuffer(device, {id: 'iso-parameters', format: 'float32', length: 8});

// values: row-major float32; levels: float32 view holding e.g. [100, 200, 300]
graph.add(new GPUIsolines({
  grid, values, levels, parameters: parameters.importToGraph(graph),
  output: {segments, segmentLevels, status: {count, overflow}}
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUIsolinesParameterValues({levelCount: 3, extent: [minX, minY, maxX, maxY]}, grid));
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`contours`,seed:319,caption:`Contour lines at fixed levels over a surface.`},related:[`isobands`,`isoband-rings`,`cost-distance`],lensScenario:`gpu-raster.isolines`,reference:n,maturity:`stable`},{id:`isobands`,symbol:`GPUIsobands`,kind:`class`,importPath:e,title:`Isobands`,summary:`Builds filled contour bands of a float32 raster as triangles, with band classes per sample and boundary edges.`,explanation:`Best for filled contour maps and isochrone polygons. Breaks are a per-frame buffer; each triangle carries its band index, ready to draw with a color ramp.`,dataTypes:{input:[`raster`,`terrain`],output:[`polygons`]},useCases:[`classify`,`transform`],domains:[`cartography`,`transport`],params:[a,p,{name:`breaks`,type:`GraphDataView<"float32">`,required:!0,description:`Ascending break values.`},{...c,description:`Packed breakCount, extent and band window.`},{name:`mask`,type:`GraphDataView<"uint32"> | undefined`,description:`Nonzero-is-valid mask over the raster.`},{name:`noDataValue`,type:`number | undefined`,description:`Finite sentinel treated as nodata.`},{name:`output`,type:`GPUIsobandsOutput`,required:!0,description:`Pick any of classes, triangles and edges.`}],outputs:[{name:`bandClasses`,type:`GraphDataView<"uint32">`,description:`Band class per sample; 0xffffffff for nodata.`},{name:`triangles`,type:`GraphDataView<"float32x2">`,description:`Triangle vertices, three rows per triangle.`},{name:`triangleBands`,type:`GraphDataView<"uint32">`,description:`Band index per triangle.`},{name:`edges`,type:`GraphDataView<"float32x4">`,description:`Counter-clockwise band boundary edges.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUIsobands, getGPUIsobandsParameterValues} from '@luma-contributors/analysis/raster-analysis';

const graph = new GPUCommandGraph(device, {id: 'isobands'});
const parameters = new GPUParameterBuffer(device, {id: 'band-parameters', format: 'float32', length: 8});

// breaks: float32 view such as [0, 5, 10, 20] minutes of travel time
graph.add(new GPUIsobands({
  grid, values, breaks, parameters: parameters.importToGraph(graph),
  output: {triangles, triangleBands, status: {count, overflow}}
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUIsobandsParameterValues({breakCount: 4, extent: [minX, minY, maxX, maxY]}, grid));
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`contours`,seed:320,caption:`Filled bands between breaks, one color each.`},related:[`isoband-rings`,`isolines`,`cost-distance`],lensScenario:`gpu-raster.isobands`,reference:n,maturity:`stable`},{id:`isoband-rings`,symbol:`GPUIsobandRings`,kind:`class`,importPath:e,title:`Isoband polygon rings`,summary:`Assembles every isoband into closed polygon rings, shells and holes, in GeoArrow layout.`,explanation:`Best when you need vector polygons, not triangles: export, overlay or style by band. Boundary edges are chained into band-pure rings and holes attach to shells of their own band.`,dataTypes:{input:[`raster`,`terrain`],output:[`polygons`]},useCases:[`transform`,`classify`],domains:[`cartography`],params:[a,p,{name:`breaks`,type:`GraphDataView<"float32">`,required:!0,description:`Ascending break values.`},{...c,description:`Packed breakCount, extent and band window.`},{name:`edgeCapacity`,type:`number`,required:!0,description:`Capacity of the intermediate boundary edge list.`},{name:`vertexTolerance`,type:`number | undefined`,description:`Vertex matching distance in world units.`},{name:`splitTouchingRings`,type:`boolean | undefined`,description:`Split rings that touch themselves at a vertex.`},{name:`output`,type:`GPUIsobandRingsOutput`,required:!0,description:`Ring offsets, positions and status views.`}],outputs:[{name:`ringOffsets`,type:`GraphDataView<"uint32">`,description:`CSR offsets of each ring into positions.`},{name:`positions`,type:`GraphDataView<"float32x2">`,description:`Ring vertices.`},{name:`ringIsHole`,type:`GraphDataView<"uint32">`,description:`Optional: 1 for hole rings.`},{name:`status`,type:`GPUCardinalityPort & GPUOverflowStatusPort & GPUIsobandRingsStatus`,description:`Ring count; check final overflow and intermediate edgeOverflow before use.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUIsobandRings, getGPUIsobandsParameterValues} from '@luma-contributors/analysis/raster-analysis';

const graph = new GPUCommandGraph(device, {id: 'rings'});
const parameters = new GPUParameterBuffer(device, {id: 'ring-parameters', format: 'float32', length: 8});

// status holds one-row scalars: ring count, overflow and edge overflow
graph.add(new GPUIsobandRings({
  grid, values, breaks, parameters: parameters.importToGraph(graph),
  edgeCapacity: 100000,
  output: {ringOffsets, positions, ringIsHole, status: {count, overflow, edgeOverflow}}
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUIsobandsParameterValues({breakCount: 4, extent: [minX, minY, maxX, maxY]}, grid));
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`polygons`,seed:321,caption:`Nested shells and holes, one ring set per band.`},related:[`isobands`,`isolines`],lensScenario:`gpu-raster.isoband-rings`,reference:n,maturity:`stable`},{id:`streamlines`,symbol:`GPUStreamlines`,kind:`class`,importPath:e,title:`Evenly spaced streamlines`,summary:`Draws evenly spaced streamlines through a 2D vector field, independent of GPU thread order.`,explanation:`Shows wind, currents or flow direction as a clean line map. Jobard-Lefer seeding prunes crowded lines, so the same field always gives the same lines.`,dataTypes:{input:[`raster`],output:[`lines`]},useCases:[`simulate`,`classify`],domains:[`cartography`,`hydrology`],params:[u,{...d,description:`Vector field width in cells; fieldHeight is the height.`},{name:`gridWidth`,type:`number`,required:!0,description:`Output grid width; gridHeight is the height.`},{name:`seedColumns`,type:`number`,required:!0,description:`Seed lattice columns; seedRows is the row count.`},{name:`stepsPerDirection`,type:`number`,required:!0,description:`Integration steps each way from a seed.`},{name:`parameters`,type:`GraphDataView<"float32">`,required:!0,description:`Extents and step length from getGPUStreamlinesParameterValues.`},{name:`uint32Parameters`,type:`GraphDataView<"uint32">`,required:!0,description:`Per-frame counts from getGPUStreamlinesUint32ParameterValues.`},{name:`output`,type:`GPUStreamlinesOutput`,required:!0,description:`Caller-owned line and position views.`}],moreRequired:[`fieldHeight`,`gridHeight`,`seedRows`],outputs:[{name:`lines`,type:`GPUCompactOutput`,description:`Seed index of each published line, with a count.`},{name:`lineOffsets`,type:`GraphDataView<"uint32">`,description:`CSR offsets of each line into positions.`},{name:`positions`,type:`GraphDataView<"float32x2">`,description:`Polyline points, backward end to forward end.`},{name:`positionCount`,type:`GraphDataView<"uint32">`,description:`One-row count of published positions.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUStreamlines, getGPUStreamlinesParameterValues, getGPUStreamlinesUint32ParameterValues}
  from '@luma-contributors/analysis/raster-analysis';

const graph = new GPUCommandGraph(device, {id: 'streamlines'});
// parameters and uint32Parameters: GPUParameterBuffer views imported into graph
graph.add(new GPUStreamlines({
  velocities, fieldWidth: 128, fieldHeight: 128, gridWidth: 256, gridHeight: 256,
  seedColumns: 32, seedRows: 32, stepsPerDirection: 64,
  parameters, uint32Parameters,
  output: {lines, lineOffsets, positions, positionCount}
}));
const compiled = await graph.compileAsync();
// then write both packers with fieldExtent, gridExtent and stepLength, and encode
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`flows`,seed:322,caption:`Evenly spaced lines follow the vector field.`},related:[`line-integral-convolution`,`particle-advection`,`terrain-flow-field`],lensScenario:`gpu-raster.streamlines`,reference:n,maturity:`stable`},{id:`line-integral-convolution`,symbol:`GPULineIntegralConvolution`,kind:`class`,importPath:e,title:`Line integral convolution`,summary:`Renders a texture that smears white noise along the streamlines of a vector field, optionally animated.`,explanation:"Shows the whole field at once, down to eddies and saddles. Each pixel averages noise along its streamline for `stepCount` steps each way (Cabral and Leedom 1993); advance the phase each frame to animate.",dataTypes:{input:[`raster`],output:[`raster`]},useCases:[`simulate`,`classify`],domains:[`cartography`,`hydrology`],params:[u,d,f,a,{name:`stepCount`,type:`number | undefined`,description:`Streamline steps in each direction, at most 256.`},{...c,description:`Float32 settings from the LIC packer.`},{name:`uint32Parameters`,type:`GraphDataView<"uint32">`,required:!0,description:`Uint32 settings from the LIC packer.`},{name:`output`,type:`GPULineIntegralConvolutionOutput`,required:!0,description:`Values, optional speeds and texture.`}],outputs:[{name:`values`,type:`GraphDataView<"float32">`,description:`LIC value in [0, 1] per pixel; NaN outside the field.`},{name:`speeds`,type:`GraphDataView<"float32">`,description:`Optional field speed per pixel, for color.`},{name:`texture`,type:`GraphTextureView<"r32float">`,description:`Optional r32float texture copy of values.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPULineIntegralConvolution, getGPULineIntegralConvolutionParameterValues}
  from '@luma-contributors/analysis/raster-analysis';

const graph = new GPUCommandGraph(device, {id: 'lic'});

// parameters and uint32Parameters: GPUParameterBuffer views imported into graph
graph.add(new GPULineIntegralConvolution({
  velocities, fieldWidth: 128, fieldHeight: 128, grid: outputGrid,
  stepCount: 24, parameters, uint32Parameters,
  output: {values, speeds}
}));
const compiled = await graph.compileAsync();
// write fieldExtent and outputExtent through the two packers; bump phase per frame
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`flows`,seed:323,caption:`Noise stretched along the flow reveals its structure.`},related:[`streamlines`,`particle-advection`,`terrain-flow-field`],lensScenario:`gpu-raster.line-integral-convolution`,reference:n,maturity:`stable`},{id:`particle-advection`,symbol:`GPUParticleAdvection`,kind:`class`,importPath:e,title:`Particle advection`,summary:`Moves particles through a 2D vector field one frame per encoding, with exact, replayable randomness.`,explanation:`Best for animated wind, current or flow maps. Respawns are counter-based, so the same parameters replay the same frames.`,dataTypes:{input:[`raster`],output:[`points`,`trajectories`]},useCases:[`simulate`],domains:[`hydrology`,`cartography`],params:[u,d,f,{...c,description:`Float32 settings: extent, time step, drop rate.`},{name:`uint32Parameters`,type:`GraphDataView<"uint32">`,required:!0,description:`Seed, frame, maximum age and reset flag.`},{name:`state`,type:`{ positions: GraphDataView<"float32x2">; ages: GraphDataView<"uint32">; generations: GraphDataView<"uint32"> }`,required:!0,description:`Particle state, updated in place.`},{name:`output`,type:`GPUParticleAdvectionOutput | undefined`,description:`Optional previous positions, speeds and trails.`}],outputs:[{name:`previousPositions`,type:`GraphDataView<"float32x2">`,description:`Optional position before this frame's step.`},{name:`speeds`,type:`GraphDataView<"float32">`,description:`Optional speed at the new position, for color.`},{name:`trails`,type:`GPUParticleAdvectionTrails`,description:`Optional trail ring buffer.`}],example:`import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParticleAdvection, getGPUParticleAdvectionParameterValues, getGPUParticleAdvectionUint32ParameterValues}
  from '@luma-contributors/analysis/raster-analysis';

const graph = new GPUCommandGraph(device, {id: 'particles'});

// state: positions, ages, generations views with one row per particle
graph.add(new GPUParticleAdvection({
  velocities, fieldWidth: 128, fieldHeight: 128,
  parameters, uint32Parameters, state,
  output: {previousPositions, speeds}
}));
const compiled = await graph.compileAsync();
// each frame: write {fieldExtent, timeStep, dropRate} and {seed, frame, maximumAge}, then encode
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,preview:{kind:`trajectories`,seed:324,caption:`Particles drift along the field and respawn.`},related:[`terrain-flow-field`,`streamlines`,`line-integral-convolution`],lensScenario:`gpu-raster.particle-advection`,reference:n,maturity:`stable`}];export{h as entries};