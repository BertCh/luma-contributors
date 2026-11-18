// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {addPreparedRecipe} from '../../gpu-spatial-analysis/recipes/prepared-recipe';

import {getRecipeView as getOrCreateView} from '../../gpu-spatial-analysis/recipes/expansion-recipe-utils';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUTrajectoryMetrics,
  type GPUTrajectoryMetricsProps
} from '../../gpu-spatial-analysis/trajectory-analysis/gpu-trajectory-metrics';
import {
  getTrajectoryTimeBindings,
  getTrajectoryTimeMode,
  getTrajectoryTimeSource
} from '../../gpu-spatial-analysis/trajectory-analysis/trajectory-metrics-kernels';
import {
  GPUTrackSimilarity,
  type GPUTrackSimilarityProps
} from '../../gpu-spatial-analysis/track-similarity/gpu-track-similarity';
import {GPUMapMatching, type GPUMapMatchingProps} from '../map-matching/gpu-map-matching';
import {snapshotMapMatchingConstructionProps} from '../map-matching/map-matching-construction';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {resolveGPUTrajectoryTemporalContext} from '../../gpu-spatial-analysis/trajectory-analysis/trajectory-common';
import type {GPURecipeStageStatusPort} from '../../analysis-core/contracts/index';
import {createRecipeKernelNode as createWGSLKernelNode} from '../../gpu-spatial-analysis/recipes/expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {RecipeBuilder} from '../../gpu-spatial-analysis/recipes/recipe-utils';

/** Raw-observation flags. First observations have no elapsed-time/speed/gap test. */
export const GPU_TRACK_QUALITY_FLAGS = {
  invalidPosition: 1,
  invalidTime: 2,
  excessiveSpeed: 4,
  timeGap: 8,
  invalidThreshold: 16
} as const;
export type GPUTrackQualityRecipeProps = {
  id?: string;
  trajectory: Omit<GPUTrajectoryMetricsProps, 'id' | 'output'>;
  /** Two live rows: maximum speed and maximum time gap, in trajectory units. */
  thresholds: GraphDataView<'float32'>;
  /**
   * How `excessiveSpeed` (bit 4) is decided. `'step'` (default, the documented contract) tests the
   * step ending at each row, so the valid fix after an outlier is flagged too and the first row is
   * never tested. `'point-centred'` flags a fix only when the speed test fails on both of its
   * adjacent steps; a track-end fix has one step, and is flagged when that step fails unless its
   * inner neighbour's outgoing step also fails (the neighbour is then the outlier). Two adjacent
   * steps that displace two consecutive fixes together are not caught by a single pass.
   */
  speedOutlierMode?: 'step' | 'point-centred';
  /** Reference-route comparison preserves raw observations; densification is explicitly selectable. */
  similarity?: Omit<GPUTrackSimilarityProps, 'id' | 'linesA' | 'spatialContext' | 'output'>;
  mapMatching?: Omit<
    GPUMapMatchingProps,
    'id' | 'positions' | 'trackOffsets' | 'spatialContext' | 'output'
  >;
};
/** Speed test of the step ending at `row` (a non-first row of its track). */
const POINT_CENTRED_STEP_WGSL = /* wgsl */ `
fn isExcessiveStep(row: u32) -> bool {
  let elapsed = timeDifferenceRows(row, row - 1u);
  let speed = speeds[speedsOffset + row];
  return isFiniteValue(elapsed) && elapsed > 0.0 && (!isFiniteValue(speed) || speed > thresholds[thresholdsOffset]);
}`;

/** Raw quality flags, optional route distances and optional map-matching completeness. */
export function addTrackQualityRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUTrackQualityRecipeProps
) {
  return addPreparedRecipe(graph, props, declareTrackQualityRecipe);
}

function declareTrackQualityRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUTrackQualityRecipeProps
) {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'trajectory',
    'thresholds',
    'speedOutlierMode',
    'similarity',
    'mapMatching'
  ]);
  props = Object.freeze({
    ...snapshot,
    trajectory: snapshotConstructionFields(snapshot.trajectory, [
      'positions',
      'trackOffsets',
      'timestamps',
      'timestampsLow',
      'temporalContext',
      'spatialContext',
      'parameters'
    ]),
    ...(snapshot.mapMatching
      ? {mapMatching: snapshotMapMatchingConstructionProps(snapshot.mapMatching)}
      : {})
  });
  const id = props.id ?? 'track-quality-recipe';
  const builder = new RecipeBuilder(graph);
  const trajectory = props.trajectory;
  resolveGPUTrajectoryTemporalContext(
    `${id}-metrics`,
    trajectory.timestamps,
    trajectory.temporalContext,
    trajectory.timestampsLow
  );
  const count = trajectory.positions.length;
  const trackCount = trajectory.trackOffsets.length - 1;
  const timeMode = getTrajectoryTimeMode(trajectory);
  if (props.thresholds.length !== 2) throw new Error(`${id}: expected speed and gap thresholds`);
  const metrics = {
    stepSpeeds: getOrCreateView(graph, `${id}-speeds`, 'float32', count),
    trackLengths: getOrCreateView(graph, `${id}-lengths`, 'float32', trackCount),
    trackDurations: getOrCreateView(graph, `${id}-durations`, 'float32', trackCount),
    maximumSpeeds: getOrCreateView(graph, `${id}-maximum-speeds`, 'float32', trackCount)
  };
  builder.add(new GPUTrajectoryMetrics({...trajectory, id: `${id}-metrics`, output: metrics}));
  const isPointCentred = props.speedOutlierMode === 'point-centred';
  const flags = getOrCreateView(graph, `${id}-flags`, 'uint32', count);
  const flaggedCounts = getOrCreateView(graph, `${id}-flagged-counts`, 'uint32', trackCount);
  builder.addGlue(
    'reset-flags',
    createWGSLKernelNode(graph, {
      id: `${id}-reset-flags`,
      operation: id,
      invocationCount: count,
      bindings: [{name: 'flags', view: flags, type: 'u32', access: 'read_write'}],
      body: 'flags[flagsOffset + index] = 0xffffffffu;'
    })
  );
  // One invocation per track avoids a row-to-track join and keeps original row IDs unchanged.
  builder.addGlue(
    'quality',
    createWGSLKernelNode(graph, {
      id: `${id}-quality`,
      operation: id,
      ...(isPointCentred ? {variant: 'point-centred'} : {}),
      invocationCount: trackCount,
      declarations: `${IS_FINITE_VALUE_WGSL}\n${getTrajectoryTimeSource(timeMode)}${isPointCentred ? `\n${POINT_CENTRED_STEP_WGSL}` : ''}`,
      bindings: [
        {name: 'positions', view: trajectory.positions, type: 'f32', access: 'read'},
        {name: 'offsets', view: trajectory.trackOffsets, type: 'u32', access: 'read'},
        ...getTrajectoryTimeBindings(trajectory),
        {name: 'thresholds', view: props.thresholds, type: 'f32', access: 'read'},
        {name: 'speeds', view: metrics.stepSpeeds, type: 'f32', access: 'read'},
        {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
      ],
      body: `let begin = min(offsets[offsetsOffset + index], ${count}u); let end = min(offsets[offsetsOffset + index + 1u], ${count}u);
for (var row = begin; row < end; row++) {
  let position = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]); var state = 0u;
  if (!isFiniteValue(position.x) || !isFiniteValue(position.y)) { state |= 1u; }
  ${timeMode === 'words' ? '' : `if (!isFiniteValue(timestamps[timestampsOffset + row])${timeMode === 'double-single' ? ' || !isFiniteValue(timestampsLow[timestampsLowOffset + row])' : ''}) { state |= 2u; }`}
  if (!isFiniteValue(thresholds[thresholdsOffset]) || thresholds[thresholdsOffset] < 0.0 || !isFiniteValue(thresholds[thresholdsOffset + 1u]) || thresholds[thresholdsOffset + 1u] < 0.0) { state |= 16u; }
  if (row > begin) {
    let elapsed = timeDifferenceRows(row, row - 1u);
    if (!isFiniteValue(elapsed) || elapsed <= 0.0) { state |= 2u; }
    if (elapsed > thresholds[thresholdsOffset + 1u]) { state |= 8u; }
    ${
      isPointCentred
        ? ''
        : `let speed = speeds[speedsOffset + row];
    if (isFiniteValue(elapsed) && elapsed > 0.0 && (!isFiniteValue(speed) || speed > thresholds[thresholdsOffset])) { state |= 4u; }`
    }
  }
  ${
    isPointCentred
      ? `let hasIncoming = row > begin; let hasOutgoing = row + 1u < end;
  let incoming = hasIncoming && isExcessiveStep(row); let outgoing = hasOutgoing && isExcessiveStep(row + 1u);
  var isOutlier = false;
  if (hasIncoming && hasOutgoing) { isOutlier = incoming && outgoing; }
  else if (hasOutgoing) { isOutlier = outgoing && !(row + 2u < end && isExcessiveStep(row + 2u)); }
  else if (hasIncoming) { isOutlier = incoming && !(row >= begin + 2u && isExcessiveStep(row - 1u)); }
  if (isOutlier) { state |= 4u; }`
      : ''
  }
  flags[flagsOffset + row] = state;
}`
    })
  );
  builder.addGlue(
    'flag-counts',
    createWGSLKernelNode(graph, {
      id: `${id}-flag-counts`,
      operation: id,
      invocationCount: trackCount,
      bindings: [
        {name: 'offsets', view: trajectory.trackOffsets, type: 'u32', access: 'read'},
        {name: 'flags', view: flags, type: 'u32', access: 'read'},
        {name: 'counts', view: flaggedCounts, type: 'u32', access: 'read_write'}
      ],
      body: `var total = 0u; for (var row = min(offsets[offsetsOffset + index], ${count}u); row < min(offsets[offsetsOffset + index + 1u], ${count}u); row++) { total += select(0u, 1u, flags[flagsOffset + row] != 0u); } counts[countsOffset + index] = total;`
    })
  );
  const stages: GPURecipeStageStatusPort[] = [];
  const similarity = props.similarity
    ? {
        hausdorff: getOrCreateView(
          graph,
          `${id}-hausdorff`,
          'float32',
          props.similarity.pairA.length
        ),
        frechet: getOrCreateView(graph, `${id}-frechet`, 'float32', props.similarity.pairA.length),
        dynamicTimeWarping: getOrCreateView(
          graph,
          `${id}-dtw`,
          'float32',
          props.similarity.pairA.length
        ),
        longestCommonSubsequence: getOrCreateView(
          graph,
          `${id}-lcss`,
          'float32',
          props.similarity.pairA.length
        ),
        frechetWithin: props.similarity.frechetEpsilon
          ? getOrCreateView(graph, `${id}-frechet-within`, 'uint32', props.similarity.pairA.length)
          : undefined,
        pairStatus: getOrCreateView(
          graph,
          `${id}-pair-status`,
          'uint32',
          props.similarity.pairA.length
        )
      }
    : undefined;
  if (props.similarity && similarity)
    builder.add(
      new GPUTrackSimilarity({
        ...props.similarity,
        id: `${id}-similarity`,
        linesA: {
          kind: 'lines',
          positions: trajectory.positions,
          lineOffsets: trajectory.trackOffsets
        },
        spatialContext: trajectory.spatialContext,
        output: similarity
      })
    );
  const matching = props.mapMatching
    ? {
        matchedEdges: getOrCreateView(graph, `${id}-matched-edges`, 'uint32', count),
        snapDistances: getOrCreateView(graph, `${id}-snap-distances`, 'float32', count),
        breaks: getOrCreateView(graph, `${id}-breaks`, 'uint32', count),
        matchedCount: getOrCreateView(graph, `${id}-matched-count`, 'uint32', 1),
        status: {
          candidateOverflow: getOrCreateView(graph, `${id}-match-candidate-overflow`, 'uint32', 1),
          routeOverflow: getOrCreateView(graph, `${id}-route-overflow`, 'uint32', 1)
        }
      }
    : undefined;
  if (props.mapMatching && matching) {
    builder.add(
      new GPUMapMatching({
        ...props.mapMatching,
        id: `${id}-matching`,
        positions: trajectory.positions,
        trackOffsets: trajectory.trackOffsets,
        spatialContext: trajectory.spatialContext,
        output: matching
      })
    );
    stages.push(
      {
        stage: 'map-matching-candidates',
        status: {candidateOverflow: matching.status.candidateOverflow}
      },
      {stage: 'map-matching-routes', status: {candidateOverflow: matching.status.routeOverflow}}
    );
  }
  return {
    contributors: builder.contributors,
    outputs: {metrics, flags, flaggedCounts, similarity, matching},
    intermediates: {
      positions: trajectory.positions,
      trackOffsets: trajectory.trackOffsets,
      spatialContext: trajectory.spatialContext
    },
    status: {stages}
  };
}
