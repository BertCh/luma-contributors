# Indirect scene execution paths

Manual resolution of 23 indirect source paths against the frozen wave 22 source manifest. All 91 exact text anchors match. This is potential execution evidence, not runtime or whole-option acceptance.

## GPUCoverageSimplification

Scene: `noding-and-coverage`; owner: `geometry_observation`.

scene.create → createNodingAndCoverage → factories[view] → createGeneraliseView → graph compile → active view encode.

Controls: `view=generalise`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPUSegmentIntersection

Scene: `noding-and-coverage`; owner: `geometry_observation`.

scene.create → createNodingAndCoverage → factories[view] → createCrossingsView → compiled graph → active view encode.

Controls: `view=crossings`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPULineSplit

Scene: `noding-and-coverage`; owner: `geometry_observation`.

scene.create → createNodingAndCoverage → factories[view] → createNodingView → selected line/network graph compile → active view encode.

Controls: `view=noding`, `Show=Network for GPUNetworkNoding`, `Lines to node`, `Snap tolerance`, `Intersection capacity`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPULineMerge

Scene: `noding-and-coverage`; owner: `geometry_observation`.

scene.create → createNodingAndCoverage → factories[view] → createNodingView → selected line/network graph compile → active view encode.

Controls: `view=noding`, `Show=Network for GPUNetworkNoding`, `Lines to node`, `Snap tolerance`, `Intersection capacity`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPUNetworkNoding

Scene: `noding-and-coverage`; owner: `geometry_observation`.

scene.create → createNodingAndCoverage → factories[view] → createNodingView → selected line/network graph compile → active view encode.

Controls: `view=noding`, `Show=Network for GPUNetworkNoding`, `Lines to node`, `Snap tolerance`, `Intersection capacity`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPUSegmentRingAssembly

Scene: `noding-and-coverage`; owner: `geometry_observation`.

scene.create → createNodingAndCoverage → factories[view] → createDissolveView → compile → active view encode.

Controls: `view=dissolve`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPULineSegmentize

Scene: `line-operations`; owner: `atlas_experience`.

line-operations.scene.create → createLineOperations → factories[view] initial/GraphSwap selection → createReshapeView.getTool → selected compiled tool → active view encode → buildDensify.

Controls: `view=reshape`, `lineTool=densify`, `lineSystem`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPULineChunk

Scene: `line-operations`; owner: `atlas_experience`.

line-operations.scene.create → createLineOperations → factories[view] initial/GraphSwap selection → createReshapeView.getTool → selected compiled tool → active view encode → buildChunk.

Controls: `view=reshape`, `lineTool=chunk or substring`, `lineSystem planar/spherical`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPULineLocate

Scene: `line-operations`; owner: `atlas_experience`.

line-operations.scene.create → createLineOperations → factories[view] initial/GraphSwap selection → createReshapeView.getTool → selected compiled tool → active view encode → buildLocate.

Controls: `view=reshape`, `lineTool=locate`, `locateMode distance/fraction`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPULinearReferencing

Scene: `line-operations`; owner: `atlas_experience`.

scene.create → createLineOperations factories.snap → createSnapView → GPULinearReferencing → compile → dirty compiled.encode; options write radius/selected points.

Controls: `view=snap`, `snapRadius`, `placeCategory`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPULineSmooth

Scene: `line-operations`; owner: `atlas_experience`.

scene.create → factories.tracks → createTracksView → prepareTool → buildSmooth → compile → smooth compiled.encode; iteration changes rebuild/cache, ratio updates parameters.

Controls: `view=tracks`, `trackTool=smooth`, `smoothIterations`, `smoothRatio`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPUShapeGenerator

Scene: `buffers-and-shapes`; owner: `geometry_observation`.

scene.create → createBuffersAndShapes factories[view] → createShapesView → build(options) → graph compile → active.compiled.encode; generated GPU positions/offsets feed fill rendering.

Controls: `view=shapes`, `shapeKind`, `shapeSystem`, `ellipseSpacing`, `maxSegments`, `segmentCount`, `shapeRadius`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPUHilbertKeys

Scene: `buffers-and-shapes`; owner: `geometry_observation`.

scene.create → factories.grid → createGridView → grid and real stop graph builders → GPUHilbertKeys sortedRows → addCurvePass → grid/stops compiled.encode.

Controls: `view=grid`, `hilbertOrder`, `grid type`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## addHotSpotAnalysisRecipe

Scene: `hot-spots`; owner: `spatial_statistics`.

createHotSpots → buildWorldFor(source) → buildCellsWorld → returned world.prepare() → buildRecipe() → recipe plus local projection graph.compileAsync → world.encode() recipe.encode; actual centers feed display/readouts.

Controls: `source=nature-cells`, `family quadbin/h3`, `resolution`, `statistic`, `inference`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPULocalFrameProjection

Scene: `hot-spots`; owner: `spatial_statistics`.

createHotSpots → buildWorldFor(source) → buildCellsWorld → returned world.prepare() → buildRecipe() → recipe plus local projection graph.compileAsync → world.encode() recipe.encode; actual centers feed display/readouts.

Controls: `source=nature-cells`, `family quadbin/h3`, `resolution`, `statistic`, `inference`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPUHotSpotAnalysis

Scene: `hot-spots`; owner: `spatial_statistics`.

createHotSpots → buildWorldFor → buildPolygonWorld → returned world.prepare → compileStatGraph with !variant.precomputed and isGi → graph.compile → world.encode statGraphs.encode.

Controls: `statistic=gi-star`, `polygon source (not precomputed nature recipe)`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPULocalMoran

Scene: `hot-spots`; owner: `spatial_statistics`.

createHotSpots → selected world.prepare → compileStatGraph with !variant.precomputed and !isGi → GPULocalMoran graph.compile → world.encode statGraphs.encode.

Controls: `statistic=local-moran`, `source nature-cells/counties/tracts`, `analytic/permutation gating`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPUSpatialTwoStageLeastSquares

Scene: `health-regression`; owner: `spatial_statistics`.

scene.create → createHealthRegression → compileSelection → getVariant(...).addLag(order) → lagGraph compile/cache → commitSelection → SceneController.encode variant.lag.get(activeOrder).encode; actual lag result table/residuals read back.

Controls: `predictor model`, `instruments order1/order2`, `weights`.

Statistics reports parent seven-step health production sweep includes lag/error; 12 OOF numerical checks validate separate held-out OLS, not these lag coefficients. County panel composition also invokes this contributor and pooled/unit fits have independent coefficient checks in county-models-numeric/report.json; that does not certify every standalone option.

## GPUPointHorizonProfile

Scene: `horizon`; owner: `earth_systems`.

horizon.scene.create → createHorizon → requestRebuild → GraphSwap.request({build:buildRebuild}) → buildSkyline → Profile and Peaks(wrap=true) → compile → SceneController.encode skyline.compiled.encode → SummaryReader.

Controls: `projection`, `traversal`, `azimuthCount`, `peakWindow`, `observer position/height`, `maxDistance`, `refraction`, `minProminence`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPUProfilePeaks

Scene: `horizon`; owner: `earth_systems`.

horizon.scene.create → createHorizon → requestRebuild → GraphSwap.request({build:buildRebuild}) → buildSkyline → Profile and Peaks(wrap=true) → compile → SceneController.encode skyline.compiled.encode → SummaryReader.

Controls: `projection`, `traversal`, `azimuthCount`, `peakWindow`, `observer position/height`, `maxDistance`, `refraction`, `minProminence`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPUPointHorizonVisibility

Scene: `horizon`; owner: `earth_systems`.

horizon.scene.create → createHorizon → requestRebuild → named buildRebuild callback → buildVisibility → compile → SceneController.encode visibility.compiled.encode → SummaryReader.

Controls: `observer position/height`, `toleranceDegrees`, `sigmaZ`, `skylineToleranceDegrees`, `targetIgnoreDistance/Fraction`.

No fresh runtime or numerical acceptance supplied for this legacy path; source dispatch is verified only.

## GPUCoverageValidity

Scene: `noding-and-coverage`; owner: `geometry_observation`.

scene.create -> createNodingAndCoverage -> factories[coverage-audit] dynamic import -> createCoverageAuditView -> requestVariant GraphSwap buildVariant -> graph compile -> active.compiled.encode.

Controls: `view=coverage-audit`, `coverageAuditInput`, `coverageAuditGroup`, `coverageAuditGapWidth`.

Eight independent coverage-audit cases passed in wave 13 on bundle 658033867c43774cdb5ca9b9d9fa4aff28575cb70fc8b6d190d343ac32d0660d. This is bounded configured-output acceptance, not every option.

## GPUCoverageDissolve

Scene: `noding-and-coverage`; owner: `geometry_observation`.

scene.create -> createNodingAndCoverage -> factories[coverage-audit] dynamic import -> createCoverageAuditView -> requestVariant GraphSwap buildVariant -> graph compile -> active.compiled.encode.

Controls: `view=coverage-audit`, `coverageAuditInput`, `coverageAuditGroup`, `coverageAuditGapWidth`.

Eight independent coverage-audit cases passed in wave 13 on bundle 658033867c43774cdb5ca9b9d9fa4aff28575cb70fc8b6d190d343ac32d0660d. This is bounded configured-output acceptance, not every option.

