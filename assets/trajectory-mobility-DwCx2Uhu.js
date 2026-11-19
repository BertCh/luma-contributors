var e=`@luma-contributors/analysis/gpu-spatial-analysis`,t=`api/gpu-spatial-analysis/movement-semantics`,n=`api/gpu-spatial-analysis/home-range-and-recursion-mob07`,r=`const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,i={name:`positions`,type:`GraphDataView<"float32x2">`,required:!0,description:`Planar fix positions, one float32x2 per row, sorted by track then time.`},a={name:`timestamps`,type:`GraphDataView<"float32"> | GraphDataView<"uint32x2">`,required:!0,description:`Fix times: float32, or Int64 epoch-millisecond words.`},o={name:`trackOffsets`,type:`GraphDataView<"uint32">`,required:!0,description:`Track-to-row offsets with one terminal entry.`},s={name:`spatialContext`,type:`GPUSpatialContext | undefined`,description:`Planar (default) or longitude/latitude great-circle.`},c=e=>({name:`parameters`,type:`GraphDataView<"float32">`,required:!0,description:`Per-frame settings, ${e}.`}),l=[{id:`trajectory-hampel`,symbol:`GPUTrajectoryHampel`,kind:`class`,importPath:e,title:`Trajectory Hampel filter`,summary:`Replaces spikes in a per-row column by the median of a window that never crosses a track.`,explanation:"A value is a spike when it is more than `numberOfSigmas` scaled median absolute deviations from the median of its window. Use it on a coordinate or speed column; rows within half a window of a track end pass through unchanged.",dataTypes:{input:[`trajectories`,`time-series`],output:[`time-series`]},useCases:[`measure`,`find-patterns`],domains:[`transport`,`urban`],params:[{name:`values`,type:`GraphDataView<"float32">`,required:!0,description:`One value per row, for example a speed or a coordinate.`},o,{name:`windowSize`,type:`number`,required:!0,description:`Full window in rows (compile-time, at most 129).`},c(`[numberOfSigmas, 0, 0, 0]`)],moreRequired:[`output`],outputs:[{name:`filtered`,type:`GraphDataView<"float32">`,description:`The window median at a spike, else the input.`},{name:`flags`,type:`GraphDataView<"uint32"> | undefined`,description:`Per-row 1 when the value was replaced.`},{name:`outlierCount`,type:`GraphDataView<"uint32"> | undefined`,description:`One row: the number of replaced values.`}],example:`import {GPUTrajectoryHampel, GPU_TRAJECTORY_HAMPEL_PARAMETER_LENGTH, getGPUTrajectoryHampelParameterValues} from '${e}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

// graph: GPUCommandGraph; speeds: one float32 per fix
const hampelParameters = new GPUParameterBuffer(device, {
  id: 'hampel', format: 'float32', length: GPU_TRAJECTORY_HAMPEL_PARAMETER_LENGTH
});
graph.add(new GPUTrajectoryHampel({
  values: speeds,
  trackOffsets,
  windowSize: 7,
  parameters: hampelParameters.importToGraph(graph),
  output: {filtered, flags, outlierCount}
}));
hampelParameters.write(getGPUTrajectoryHampelParameterValues({numberOfSigmas: 3}));
${r}`,preview:{kind:`time-series`,caption:`Spikes replaced by the window median.`,seed:701},related:[`trajectory-outlier-flags`,`trajectory-kalman`,`trajectory-thinning`],reference:t,maturity:`stable`},{id:`trajectory-outlier-flags`,symbol:`GPUTrajectoryOutlierFlags`,kind:`class`,importPath:e,title:`Trajectory outlier flags`,summary:`Flags a displaced fix: both steps around it exceed the speed limit, or speed spikes up and down around it.`,explanation:`A displaced fix makes both the step into it and the step out of it too fast, so the good fix after a spike is not blamed. It is a single pass: two displaced fixes in a row are not caught. Rows are flagged, not removed.`,dataTypes:{input:[`trajectories`],output:[`trajectories`]},useCases:[`find-patterns`,`measure`],domains:[`transport`,`urban`],params:[i,a,o,c(`[maximumSpeed, maximumAcceleration, 0, 0]`),s],moreRequired:[`output`],outputs:[{name:`flags`,type:`GraphDataView<"uint32">`,description:`Per-row bits: 1 speed, 2 acceleration.`},{name:`trackFlaggedCounts`,type:`GraphDataView<"uint32"> | undefined`,description:`Flagged rows per track.`}],example:`import {GPUTrajectoryOutlierFlags, getGPUTrajectoryOutlierFlagsParameterValues, GPU_TRAJECTORY_OUTLIER_FLAGS, GPU_TRAJECTORY_OUTLIER_FLAGS_PARAMETER_LENGTH} from '${e}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

// graph: GPUCommandGraph
const limitParameters = new GPUParameterBuffer(device, {
  id: 'limits', format: 'float32', length: GPU_TRAJECTORY_OUTLIER_FLAGS_PARAMETER_LENGTH
});
graph.add(new GPUTrajectoryOutlierFlags({
  positions, timestamps, trackOffsets,
  parameters: limitParameters.importToGraph(graph),
  output: {flags, trackFlaggedCounts}
}));
limitParameters.write(getGPUTrajectoryOutlierFlagsParameterValues({maximumSpeed: 15}));
${r}
// after readback: (flags[row] & GPU_TRAJECTORY_OUTLIER_FLAGS.speed) !== 0`,preview:{kind:`trajectories`,caption:`Displaced fixes flagged.`,seed:702},related:[`trajectory-hampel`,`trajectory-kalman`,`trajectory-split`],reference:t,maturity:`stable`},{id:`trajectory-split`,symbol:`GPUTrajectorySplit`,kind:`class`,importPath:e,title:`Trajectory split`,summary:`Splits tracks into segments at time gaps, at staypoints or where a per-row value changes.`,explanation:"Mode `gap` cuts after any step longer than `maximumGap`, `stop` keeps the runs outside staypoints, and `value-change` cuts where a label changes and repeats the boundary row. Segments of one row are dropped. The output lists source rows, so the next operation needs no readback.",dataTypes:{input:[`trajectories`],output:[`trajectories`]},useCases:[`relate`,`summarize`],domains:[`transport`,`urban`],params:[{name:`mode`,type:`"gap" | "stop" | "value-change"`,required:!0,description:`How tracks are split (compile-time).`},i,a,o,{name:`parameters`,type:`GraphDataView<"float32"> | undefined`,description:`Mode 'gap': [maximumGap, 0, 0, 0], per frame.`},{name:`staypointIds`,type:`GraphDataView<"uint32"> | undefined`,description:`Mode 'stop': per-row staypoint index.`},{name:`values`,type:`GraphDataView<"uint32"> | undefined`,description:`Mode 'value-change': one label per row.`}],moreRequired:[`output`],outputs:[{name:`segmentOffsets`,type:`GraphDataView<"uint32">`,description:`New offsets into rowIds, one more than segments.`},{name:`segmentTracks`,type:`GraphDataView<"uint32">`,description:`Source track of each segment.`},{name:`rowIds`,type:`GraphDataView<"uint32">`,description:`Source row of each output row.`},{name:`status`,type:`GPUCardinalityPort & GPUOverflowStatusPort`,description:`Segment count and overflow.`},{name:`positions`,type:`GraphDataView<"float32x2"> | undefined`,description:`Optional copied positions of the output rows.`}],example:`import {GPUTrajectorySplit, GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH, getGPUTrajectorySplitParameterValues} from '${e}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

// graph: GPUCommandGraph; capacities are row counts
const gapParameters = new GPUParameterBuffer(device, {
  id: 'gap', format: 'float32', length: GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH
});
graph.add(new GPUTrajectorySplit({
  mode: 'gap',
  positions, timestamps, trackOffsets,
  parameters: gapParameters.importToGraph(graph),
  output: {segmentOffsets, segmentTracks, rowIds, status: {count, overflow}}
}));
gapParameters.write(getGPUTrajectorySplitParameterValues({maximumGap: 1800}));
${r}`,preview:{kind:`trajectories`,caption:`Tracks cut into segments.`,seed:703},related:[`trajectory-staypoints`,`trajectory-segments`,`trajectory-thinning`],reference:t,maturity:`stable`},{id:`trajectory-thinning`,symbol:`GPUTrajectoryThinning`,kind:`class`,importPath:e,title:`Trajectory thinning`,summary:`Keeps the first fix, then the first later fix at least a tolerance away, in distance or time.`,explanation:`The MovingPandas minimum-distance and minimum-time generalization: each kept fix is followed by the first later fix at or beyond the tolerance, and the last fix is always kept. The sequential rule runs in a bounded forward search per row plus pointer doubling.`,dataTypes:{input:[`trajectories`],output:[`trajectories`]},useCases:[`summarize`,`measure`],domains:[`transport`,`urban`],params:[{name:`mode`,type:`"minimum-distance" | "minimum-time"`,required:!0,description:`What the tolerance measures.`},i,a,o,c(`[tolerance, 0, 0, 0]`),{name:`maximumSearchRows`,type:`number | undefined`,default:`4096`,description:`Forward-search cap per row (compile-time).`},s],moreRequired:[`output`],outputs:[{name:`rowIds`,type:`GraphDataView<"uint32">`,description:`Kept source rows in order.`},{name:`trackOffsets`,type:`GraphDataView<"uint32">`,description:`New per-track offsets into rowIds.`},{name:`status`,type:`GPUCardinalityPort & GPUOverflowStatusPort & GPUCandidateStatusPort`,description:`Kept count and overflow flags.`},{name:`validity`,type:`GraphDataView<"uint32"> | undefined`,description:`Optional 1/0 validity of the rows: 1 for a kept row.`}],example:`import {GPUTrajectoryThinning, GPU_TRAJECTORY_THINNING_PARAMETER_LENGTH, getGPUTrajectoryThinningParameterValues} from '${e}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

// graph: GPUCommandGraph
const toleranceParameters = new GPUParameterBuffer(device, {
  id: 'tolerance', format: 'float32', length: GPU_TRAJECTORY_THINNING_PARAMETER_LENGTH
});
graph.add(new GPUTrajectoryThinning({
  mode: 'minimum-distance',
  positions, timestamps, trackOffsets,
  parameters: toleranceParameters.importToGraph(graph),
  output: {rowIds, trackOffsets: keptOffsets, status: {count, overflow, candidateOverflow}}
}));
toleranceParameters.write(getGPUTrajectoryThinningParameterValues({tolerance: 500}));
${r}`,preview:{kind:`trajectories`,caption:`Fixes thinned by distance.`,seed:704},related:[`trajectory-resample`,`trajectory-split`,`trajectory-hampel`],reference:t,maturity:`stable`},{id:`trajectory-track-relative-times`,symbol:`GPUTrajectoryTrackRelativeTimes`,kind:`class`,importPath:e,title:`Track-relative times`,summary:`Int64 epoch-millisecond words to float32 ticks after each track's own first fix.`,explanation:`A float32 cannot hold a current epoch in milliseconds; it is off by tens of seconds. Differences taken on the words are exact, so a track is exact while it spans fewer than 2^24 ticks. The output feeds float32 trajectory paths such as time-ratio simplification.`,dataTypes:{input:[`trajectories`,`time-series`],output:[`time-series`]},useCases:[`measure`],domains:[`transport`,`urban`],params:[{name:`timestamps`,type:`GraphDataView<"uint32x2">`,required:!0,description:`Int64 word timestamps, (low, high) per row.`},o,{name:`temporalContext`,type:`GPUTemporalContext | undefined`,description:`Optional declaration; must be int64-words.`}],moreRequired:[`output`],outputs:[{name:`output`,type:`GraphDataView<"float32">`,description:`Ticks after the first row of each track.`}],example:`import {GPUTrajectoryTrackRelativeTimes} from '${e}';

// graph: GPUCommandGraph; timestamps: uint32x2 words of epoch milliseconds
graph.add(new GPUTrajectoryTrackRelativeTimes({
  timestamps,
  trackOffsets,
  output: relativeTimes
}));
${r}
// relativeTimes is the float32 time column other trajectory operations read`,preview:{kind:`time-series`,caption:`Epoch words to exact relative ticks.`,seed:705},related:[`trajectory-resample`,`trajectory-thinning`,`trajectory-playhead`],reference:t,maturity:`stable`},{id:`trajectory-staypoints`,symbol:`GPUTrajectoryStaypoints`,kind:`class`,importPath:e,title:`Trajectory staypoints`,summary:`Finds where a track stayed within a distance of an anchor for at least a minimum time.`,explanation:"The trackintel sliding-window rule: a fix at `distanceThreshold` or farther closes the window and emits a staypoint if `timeThreshold` has elapsed. Each row searches forward independently and pointer doubling reproduces the sequential chain exactly.",dataTypes:{input:[`trajectories`],output:[`table`,`points`]},useCases:[`find-patterns`,`summarize`],domains:[`transport`,`urban`],params:[i,a,o,c(`[distanceThreshold, timeThreshold, maximumGap, includeLast]`),{name:`maximumSearchRows`,type:`number | undefined`,default:`4096`,description:`Forward-search cap per anchor (compile-time).`},s],moreRequired:[`output`],outputs:[{name:`ids`,type:`GraphDataView<"uint32">`,description:`Track of each staypoint, in order.`},{name:`status`,type:`GPUCompactOutput["status"] & GPUCandidateStatusPort`,description:`Staypoint count, overflow and candidate overflow.`},{name:`centroids`,type:`GraphDataView<"float32x2"> | undefined`,description:`Centroid of each staypoint.`},{name:`durations`,type:`GraphDataView<"float32"> | undefined`,description:`finished_at minus started_at.`},{name:`staypointIds`,type:`GraphDataView<"uint32"> | undefined`,description:`Per-row staypoint index, else 0xffffffff.`}],example:`import {GPUTrajectoryStaypoints, GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH, getGPUTrajectoryStaypointsParameterValues} from '${e}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

// graph: GPUCommandGraph; capacity rows for each column
const stayParameters = new GPUParameterBuffer(device, {
  id: 'stays', format: 'float32', length: GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH
});
graph.add(new GPUTrajectoryStaypoints({
  positions, timestamps, trackOffsets,
  parameters: stayParameters.importToGraph(graph),
  output: {ids, status: {count, overflow}, startRows, endRows, finishRows, centroids, durations,
    staypointIds}
}));
stayParameters.write(getGPUTrajectoryStaypointsParameterValues(
  {distanceThreshold: 100, timeThreshold: 1800}));
${r}`,preview:{kind:`trajectories`,caption:`Stays found along each track.`,seed:706},related:[`trajectory-locations`,`trajectory-segments`,`trajectory-metrics`],reference:t,maturity:`stable`},{id:`trajectory-locations`,symbol:`GPUTrajectoryLocations`,kind:`class`,importPath:e,title:`Trajectory locations`,summary:`Groups the staypoints of one track that lie within epsilon of each other into locations.`,explanation:"Connected components of the staypoints of a track within `epsilon` (trackintel `generate_locations`, DBSCAN with one sample). Components are labelled by bounded min-label hooking with path compression. Feed it the output of `GPUTrajectoryStaypoints`.",dataTypes:{input:[`points`],output:[`points`,`table`]},useCases:[`find-patterns`,`summarize`],domains:[`transport`,`urban`],params:[{name:`centroids`,type:`GraphDataView<"float32x2">`,required:!0,description:`Staypoint centroids from GPUTrajectoryStaypoints.`},{name:`tracks`,type:`GraphDataView<"uint32">`,required:!0,description:`Track of each staypoint, sorted.`},{name:`count`,type:`GraphDataView<"uint32">`,required:!0,description:`One row: the active staypoint count.`},c(`[epsilon, 0, 0, 0]`),{name:`maximumIterations`,type:`number | undefined`,default:`32`,description:`Hook-and-compress rounds (compile-time, at most 256).`},s],moreRequired:[`output`],outputs:[{name:`locationIds`,type:`GraphDataView<"uint32">`,description:`Compact location of each staypoint.`},{name:`status`,type:`GPUStatusScalar & GPUIterationStatusPort`,description:`Location count and iteration status.`},{name:`centers`,type:`GraphDataView<"float32x2"> | undefined`,description:`Centroid of each location.`}],example:`import {GPUTrajectoryLocations, GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH, getGPUTrajectoryLocationsParameterValues} from '${e}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

// graph: GPUCommandGraph; the staypoint columns come from GPUTrajectoryStaypoints
const locationParameters = new GPUParameterBuffer(device, {
  id: 'locations', format: 'float32', length: GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH
});
graph.add(new GPUTrajectoryLocations({
  centroids: stayCentroids,
  tracks: stayTracks,
  count: stayCount,
  parameters: locationParameters.importToGraph(graph),
  output: {locationIds, status: {count: locationCount}, centers}
}));
locationParameters.write(getGPUTrajectoryLocationsParameterValues({epsilon: 150}));
${r}`,preview:{kind:`point-clusters`,caption:`Stays merged into places.`,seed:707},related:[`trajectory-staypoints`,`trajectory-mobility-indicators`,`trajectory-segments`],reference:t,maturity:`stable`},{id:`trajectory-segments`,symbol:`GPUTrajectorySegments`,kind:`class`,importPath:e,title:`Triplegs and trips`,summary:`Triplegs are the runs between staypoints; trips join triplegs between two activities.`,explanation:"The trackintel tripleg and trip rules. A staypoint longer than `activityMinimumDuration` is an activity; a trip runs from one activity to the next and breaks at steps longer than `maximumGap`. Needs the per-row staypoint ids and table of `GPUTrajectoryStaypoints`.",dataTypes:{input:[`trajectories`],output:[`trajectories`,`table`]},useCases:[`summarize`,`find-patterns`],domains:[`transport`,`urban`],params:[i,a,o,{name:`staypointIds`,type:`GraphDataView<"uint32">`,required:!0,description:`Per-row staypoint index from GPUTrajectoryStaypoints.`},{name:`staypoints`,type:`object | undefined`,description:`Staypoint table columns, required for trips.`},c(`[maximumGap, activityMinimumDuration, 0, 0]`)],moreRequired:[`output`],outputs:[{name:`triplegs`,type:`GPUTrajectoryTriplegOutput`,description:`Tripleg table: track, start and end rows.`},{name:`trips`,type:`GPUTrajectoryTripOutput | undefined`,description:`Trip table with origin and destination.`}],example:`import {GPUTrajectorySegments, GPU_TRAJECTORY_SEGMENTS_PARAMETER_LENGTH, getGPUTrajectorySegmentsParameterValues} from '${e}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

// graph: GPUCommandGraph; stay* columns come from GPUTrajectoryStaypoints
const segmentParameters = new GPUParameterBuffer(device, {
  id: 'segments', format: 'float32', length: GPU_TRAJECTORY_SEGMENTS_PARAMETER_LENGTH
});
graph.add(new GPUTrajectorySegments({
  positions, timestamps, trackOffsets,
  staypointIds: rowStaypointIds,
  staypoints: {tracks: stayTracks, startRows, finishRows, durations,
    status: {count: stayCount, overflow: stayOverflow, candidateOverflow: stayCandidateOverflow}},
  parameters: segmentParameters.importToGraph(graph),
  output: {
    triplegs: {ids: triplegTracks, status: {count: triplegCount, overflow, candidateOverflow: triplegCandidateOverflow}, startRows: legStarts, endRows: legEnds},
    trips: {ids: tripTracks, status: {count: tripCount, overflow: tripOverflow, candidateOverflow: tripCandidateOverflow}}
  }
}));
segmentParameters.write(getGPUTrajectorySegmentsParameterValues({activityMinimumDuration: 1800}));`,preview:{kind:`trajectories`,caption:`Triplegs between stays.`,seed:708},related:[`trajectory-staypoints`,`trajectory-modal-split`,`trajectory-mobility-indicators`],reference:t,maturity:`stable`},{id:`trajectory-recursions`,symbol:`GPUTrajectoryRecursions`,kind:`class`,importPath:e,title:`Trajectory recursions`,summary:`For a circle around every fix: revisits, residence time and first passage time.`,explanation:`After Bracis, Bildstein and Mueller (recurse) and Fauchald and Tveraa: how often a track returns to a place, how long it stays and how long it takes to leave. Crossings are exact on the straight segments. Work grows with the square of the track length (reference tier).`,dataTypes:{input:[`trajectories`],output:[`trajectories`]},useCases:[`measure`,`find-patterns`],domains:[`ecology`,`transport`],params:[i,a,o,{name:`radii`,type:`GraphDataView<"float32">`,required:!0,description:`Circle radii in position units, one row each (at most 32).`},{name:`maximumTrackSegments`,type:`number | undefined`,default:`4096`,description:`Segments examined per row (compile-time).`},s],moreRequired:[`output`],outputs:[{name:`revisits`,type:`GraphDataView<"uint32"> | undefined`,description:`Times the track re-enters the circle.`},{name:`residenceTimes`,type:`GraphDataView<"float32"> | undefined`,description:`Total time inside the circle.`},{name:`firstPassageTimes`,type:`GraphDataView<"float32"> | undefined`,description:`Time until the track first leaves it.`},{name:`firstPassageCensored`,type:`GraphDataView<"uint32"> | undefined`,description:`1 where the track ended inside.`}],example:`import {GPUTrajectoryRecursions} from '${e}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

// graph: GPUCommandGraph; the radii view holds one float32 per radius, rewritten per frame
const radiiBuffer = new GPUParameterBuffer(device, {id: 'radii', format: 'float32', length: 1});
graph.add(new GPUTrajectoryRecursions({
  positions, timestamps, trackOffsets,
  radii: radiiBuffer.importToGraph(graph),
  output: {revisits, residenceTimes, firstPassageTimes, firstPassageCensored}
}));
radiiBuffer.write(Float32Array.of(500));
${r}
// output rows are fix * radiusCount + radius`,preview:{kind:`trajectories`,caption:`Revisits and residence per fix.`,seed:709},related:[`trajectory-brownian-bridge-utilization`,`trajectory-staypoints`,`trajectory-metrics`],reference:n,maturity:`partial`},{id:`trajectory-brownian-bridge-utilization`,symbol:`GPUBrownianBridgeUtilization`,kind:`class`,importPath:e,title:`Brownian bridge utilization`,summary:`Probability density of where a track was, between its fixes, on a grid.`,explanation:"After Horne et al.: between two fixes the position is a normal whose variance swells mid-gap with `motionVariances` and the location errors at each end. The density of a cell is the time-weighted average over all segments. Reference tier: every cell loops over all fixes.",dataTypes:{input:[`trajectories`],output:[`raster`]},useCases:[`interpolate`,`model`],domains:[`ecology`,`transport`],params:[i,a,o,{name:`motionVariances`,type:`GraphDataView<"float32">`,required:!0,description:`Brownian variance of the segment starting at each fix.`},{name:`locationErrors`,type:`GraphDataView<"float32">`,required:!0,description:`Location error standard deviation of each fix.`},{name:`gridSize`,type:`readonly [number, number]`,required:!0,description:`Columns and rows of the output grid (compile-time).`},c(`[minX, minY, maxX, maxY] of the grid extent`),{name:`subStepCount`,type:`number | undefined`,default:`15`,description:`Midpoint-rule steps per segment (1 to 256).`}],moreRequired:[`output`],outputs:[{name:`density`,type:`GraphDataView<"float32">`,description:`Density per cell, row-major from the smallest y.`}],example:`import {GPUBrownianBridgeUtilization, GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH, getGPUBrownianBridgeUtilizationParameterValues} from '${e}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

// graph: GPUCommandGraph; variances and errors hold one float32 per fix
const gridParameters = new GPUParameterBuffer(device, {
  id: 'grid', format: 'float32', length: GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH
});
graph.add(new GPUBrownianBridgeUtilization({
  positions, timestamps, trackOffsets,
  motionVariances: variances,
  locationErrors: errors,
  gridSize: [128, 128],
  parameters: gridParameters.importToGraph(graph),
  output: {density}
}));
gridParameters.write(getGPUBrownianBridgeUtilizationParameterValues(
  {extent: [0, 0, 20000, 20000]}));
${r}`,preview:{kind:`density-surface`,caption:`Where the track probably was.`,seed:710},related:[`trajectory-utilization-volume-isopleths`,`trajectory-recursions`,`kernel-density`],reference:n,maturity:`partial`},{id:`trajectory-utilization-volume-isopleths`,symbol:`GPUUtilizationVolumeIsopleths`,kind:`class`,importPath:e,title:`Utilization volume isopleths`,summary:`Home-range volumes: the share of probability mass in cells at least as dense as each cell.`,explanation:"Sorts cells by density and sums from the densest down (`getVolumeUD` of move). The `level` isopleth, for example 0.95, is the smallest set of densest cells whose mass reaches it. Reads any non-negative surface, typically a Brownian bridge density.",dataTypes:{input:[`raster`],output:[`raster`]},useCases:[`summarize`,`measure`],domains:[`ecology`,`transport`],params:[{name:`density`,type:`GraphDataView<"float32">`,required:!0,description:`Non-negative density per cell.`},c(`[level, 0, 0, 0] with level in (0, 1]`)],moreRequired:[`output`],outputs:[{name:`volume`,type:`GraphDataView<"float32">`,description:`Cumulative mass share up to each cell.`},{name:`inside`,type:`GraphDataView<"uint32"> | undefined`,description:`1 for cells of the level isopleth.`},{name:`totalMass`,type:`GraphDataView<"float32"> | undefined`,description:`One row: the sum of the cells.`}],example:`import {GPUUtilizationVolumeIsopleths, GPU_UTILIZATION_VOLUME_ISOPLETHS_PARAMETER_LENGTH, getGPUUtilizationVolumeIsoplethsParameterValues} from '${e}';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

// graph: GPUCommandGraph; density: output of GPUBrownianBridgeUtilization
const isoplethParameters = new GPUParameterBuffer(device, {
  id: 'isopleths', format: 'float32', length: GPU_UTILIZATION_VOLUME_ISOPLETHS_PARAMETER_LENGTH
});
graph.add(new GPUUtilizationVolumeIsopleths({
  density,
  parameters: isoplethParameters.importToGraph(graph),
  output: {volume, inside, totalMass}
}));
isoplethParameters.write(getGPUUtilizationVolumeIsoplethsParameterValues({level: 0.95}));
${r}`,preview:{kind:`contours`,caption:`The 95% home range outlined.`,seed:711},related:[`trajectory-brownian-bridge-utilization`,`kernel-density`,`trajectory-recursions`],reference:n,maturity:`stable`},{id:`trajectory-mobility-indicators`,symbol:`GPUMobilityIndicators`,kind:`class`,importPath:e,title:`Mobility indicators`,summary:`Per-track radius of gyration, entropies, home and work locations and tracking quality.`,explanation:`The scikit-mobility individual measures and trackintel home and tracking-quality rules, computed from staypoints and their locations, one thread per track. Meant for people over weeks; on vessels, home is the berth.`,dataTypes:{input:[`table`,`points`],output:[`table`]},useCases:[`measure`,`summarize`],domains:[`transport`,`urban`],params:[{name:`trackCount`,type:`number`,required:!0,description:`Number of tracks, the row count of per-track outputs.`},a,{name:`staypoints`,type:`GPUMobilityStaypointColumns`,required:!0,description:`Staypoint tracks, centroids, start rows, durations, locations, count.`},{name:`triplegs`,type:`object | undefined`,description:`Tripleg table, needed for trackingQuality.`},{name:`maximumEntropySequenceRows`,type:`number | undefined`,default:`1024`,description:`Cap on a track's staypoint sequence for realEntropy.`},s],moreRequired:[`output`],outputs:[{name:`radiusOfGyration`,type:`GraphDataView<"float32"> | undefined`,description:`RMS distance of stays from their mean.`},{name:`realEntropy`,type:`GraphDataView<"float32"> | undefined`,description:`Lempel-Ziv entropy of the location sequence.`},{name:`homeLocations`,type:`GraphDataView<"uint32"> | undefined`,description:`Location with the longest summed stay.`},{name:`trackingQuality`,type:`GraphDataView<"float32"> | undefined`,description:`Tracked duration over extent.`}],example:`import {GPUMobilityIndicators} from '${e}';

// graph: GPUCommandGraph; staypoint columns from GPUTrajectoryStaypoints and Locations
graph.add(new GPUMobilityIndicators({
  trackCount,
  timestamps,
  staypoints: {tracks, centroids, startRows, durations, locationIds,
    status: {count, overflow: staypointOverflow, candidateOverflow: staypointCandidateOverflow}},
  output: {radiusOfGyration, realEntropy, randomEntropy, homeLocations, workLocations}
}));
${r}`,preview:{kind:`trajectories`,caption:`One set of numbers per track.`,seed:712},related:[`trajectory-staypoints`,`trajectory-locations`,`trajectory-modal-split`],reference:t,maturity:`stable`},{id:`trajectory-modal-split`,symbol:`GPUModalSplit`,kind:`class`,importPath:e,title:`Modal split`,summary:`Totals triplegs per track and mode by count, distance or duration; modes may come from speed.`,explanation:"The trackintel `calculate_modal_split`. Give each tripleg a mode, or give ascending `modeSpeedBreaks` and the mode is the speed class of its average speed. One thread per track accumulates the chosen `metric`.",dataTypes:{input:[`trajectories`,`table`],output:[`table`]},useCases:[`summarize`,`measure`],domains:[`transport`,`urban`],params:[i,a,o,{name:`triplegs`,type:`GPUModalSplitTriplegs`,required:!0,description:`Tripleg tracks, start rows, end rows and count.`},{name:`modes`,type:`GraphDataView<"uint32"> | undefined`,description:`Mode of each tripleg, unless modeSpeedBreaks is given.`},{name:`modeSpeedBreaks`,type:`readonly number[] | undefined`,description:`Strictly ascending upper speed bounds per class (compile-time).`},{name:`modeCount`,type:`number`,required:!0,description:`Number of modes; the length of modeSpeedBreaks when given.`},{name:`metric`,type:`"count" | "distance" | "duration"`,required:!0,description:`What is added up per track and mode (compile-time).`}],moreRequired:[`output`],outputs:[{name:`modalSplit`,type:`GraphDataView<"float32">`,description:`Metric per track and mode, track-major.`},{name:`predictedModes`,type:`GraphDataView<"uint32"> | undefined`,description:`Speed class of each tripleg.`}],example:`import {GPUModalSplit} from '${e}';

// graph: GPUCommandGraph; tripleg columns come from GPUTrajectorySegments
graph.add(new GPUModalSplit({
  positions, timestamps, trackOffsets,
  triplegs: {tracks: legTracks, startRows: legStarts, endRows: legEnds,
    status: {count: legCount, overflow: legOverflow, candidateOverflow: legCandidateOverflow}},
  modeSpeedBreaks: [15 / 3.6, 100 / 3.6, Infinity],
  modeCount: 3,
  metric: 'distance',
  output: {modalSplit, predictedModes}
}));
${r}`,preview:{kind:`histogram`,caption:`Distance per track by speed class.`,seed:713},related:[`trajectory-segments`,`trajectory-mobility-indicators`,`trajectory-metrics`],reference:t,maturity:`stable`}];export{l as entries};