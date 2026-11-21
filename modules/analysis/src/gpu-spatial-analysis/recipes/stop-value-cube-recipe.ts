import {snapshotWeightsPort} from '../spatial-weights/weights-construction-snapshot';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUPointToCell, type GPUPointToCellProps} from '../cell-indexing/gpu-point-to-cell';
import {GPUKeyLookup} from '../../dataframe-analysis/key-join/gpu-key-join';
import {
  GPUPeriodIndex,
  type GPUPeriodIndexProps
} from '../../dataframe-analysis/period-index/gpu-period-index';
import {GPUValueCube} from '../../dataframe-analysis/value-cube/gpu-value-cube';
import {
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createWeightsLossNodes,
  validateWeightsLossSources,
  getWeightsLossSources
} from '../spatial-weights/spatial-weights-loss';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {addPreparedRecipe} from './prepared-recipe';
import {RecipeBuilder, addRecipeCountNodes, resolveRecipeDestinations} from './recipe-utils';
import {getRegressionWeightsLossNodes} from '../spatial-regression/regression-weights-completeness';

export type GPUStopValueCubeRecipeOutput = {
  /** Cell-major exact counts, cell * periods.count + period; suppressed to zero on loss. */
  counts: GraphDataView<'uint32'>;
};

/** Loss evidence for one stop or slice input. */
export type GPUStopValueCubeRecipeStage =
  | {stage: 'stops'; status: GPUStopValueCubeRecipeProps['stops']['status']}
  | {
      stage: `slice-${number}`;
      status: {overflow?: GraphDataView<'uint32'>; candidateOverflow?: GraphDataView<'uint32'>};
    }
  | {
      stage: 'cube';
      status: {
        invalidCount: GraphDataView<'uint32'>;
        overflow: GraphDataView<'uint32'>;
        candidateOverflow: GraphDataView<'uint32'>;
      };
    };

/** Full resident recipe result; GPUStopValueCubeRecipeOutput remains the counts-only destination. */
export type GPUStopValueCubeRecipeResult = {
  contributors: readonly GPUCommandNodeProducer<never>[];
  outputs: GPUStopValueCubeRecipeOutput;
  intermediates: {
    periods: GraphDataView<'uint32'>;
    cellRows: GraphDataView<'uint32'>;
    live: GraphDataView<'uint32'>;
    times: GraphDataView<'uint32x2'>;
  };
  weights: GPUSpatialWeights;
  status: {
    invalidCount: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    candidateOverflow: GraphDataView<'uint32'>;
    stages: GPUStopValueCubeRecipeStage[];
  };
};

/** Event-to-cube composition with explicit time, dictionary row order and live loss evidence. */
export type GPUStopValueCubeRecipeProps = {
  id?: string;
  /**
   * Compact stop centroids in longitude/latitude degrees. Only status.count rows are live.
   * Allocate at least one centroid, startRows and finishRows row, including when status.count is zero.
   */
  stops: {
    centroids: GraphDataView<'float32x2'>;
    startRows: GraphDataView<'uint32'>;
    finishRows: GraphDataView<'uint32'>;
    status: {
      count: GraphDataView<'uint32'>;
      overflow?: GraphDataView<'uint32'>;
      candidateOverflow?: GraphDataView<'uint32'>;
    };
  };
  time: 'start' | 'finish';
  /** Original trajectory timestamps, signed Int64 ticks from Unix epoch, not relative float times. */
  timestamps: GraphDataView<'uint32x2'>;
  temporalContext: GPUPeriodIndexProps['temporalContext'];
  /** Fixed periods in the explicitly declared timestamp unit. */
  periods: Extract<GPUPeriodIndexProps['periods'], {kind: 'fixed'}>;
  family: GPUPointToCellProps['family'];
  resolution: number;
  /** Fixed cell keys; their row order is the cube cell axis and the weights row order. */
  dictionary: GraphDataView<'uint32x2'>;
  weights: GPUSpatialWeights;
  /** Optional loss from per-slice tables used to prepare the input. */
  sliceStatuses?: readonly {
    overflow?: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
  }[];
  /** Cell-major counts: cell * periods.count + period. Unknown bins on loss are suppressed to zero; inspect status. */
  output?: Partial<GPUStopValueCubeRecipeOutput>;
};

/** Bins live stops against a stable dictionary. Empty observed bins are zero. Loss suppresses the cube. */
export function addStopValueCubeRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUStopValueCubeRecipeProps
): GPUStopValueCubeRecipeResult {
  return addPreparedRecipe(graph, props, declareStopValueCubeRecipe);
}

function declareStopValueCubeRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUStopValueCubeRecipeProps
): GPUStopValueCubeRecipeResult {
  const id = props.id ?? 'stop-value-cube';
  const output = resolveRecipeDestinations(id, props);
  const stopStatus = {
    count: props.stops.status.count,
    overflow: props.stops.status.overflow,
    candidateOverflow: props.stops.status.candidateOverflow
  };
  const sliceStatuses =
    props.sliceStatuses?.map(status => ({
      overflow: status.overflow,
      candidateOverflow: status.candidateOverflow
    })) ?? [];
  const lossInputs = [
    {status: stopStatus},
    ...sliceStatuses.map(status => ({status})),
    props.weights
  ];
  validateWeightsLossSources(id, lossInputs);
  for (const name of ['count', 'requiredCount'] as const) {
    const count = props.weights.status?.[name];
    if (count) {
      validateSinglePackedView(count, ['uint32'], `${id} weights.status.${name}`);
      validateRequirement(
        id,
        count.length === 1,
        `weights.status.${name} must contain one word; received ${count.length}`
      );
    }
  }
  validateSinglePackedView(props.timestamps, ['uint32x2'], `${id} timestamps`);
  validateSinglePackedView(props.stops.startRows, ['uint32'], `${id} stops.startRows`);
  validateSinglePackedView(props.stops.finishRows, ['uint32'], `${id} stops.finishRows`);
  validateRequirement(
    id,
    props.time === 'start' || props.time === 'finish',
    `time must be start or finish; received ${props.time}`
  );
  validateRequirement(
    id,
    props.weights.offsets.length === props.dictionary.length + 1,
    'weights rows must equal dictionary rows'
  );
  validateSinglePackedView(stopStatus.count, ['uint32'], `${id} stops.status.count`);
  validateRequirement(id, stopStatus.count.length >= 1, 'stops.status.count must contain a scalar');
  const rowCount = props.stops.centroids.length;
  validateRequirement(
    id,
    props.stops.startRows.length === rowCount && props.stops.finishRows.length === rowCount,
    'stop startRows and finishRows must align with centroids'
  );
  const builder = new RecipeBuilder(graph);
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${id}-${name}`, format, length);
  const live = view('live', 'uint32', rowCount);
  const times = view('times', 'uint32x2', rowCount);
  const ones = view('ones', 'float32', rowCount);
  builder.addGlue(
    'stop-time',
    createWGSLKernelNode(graph, {
      id: `${id}-stop-time`,
      operation: id,
      invocationCount: rowCount,
      bindings: [
        {
          name: 'rows',
          view: props.time === 'start' ? props.stops.startRows : props.stops.finishRows,
          type: 'u32',
          access: 'read'
        },
        {name: 'count', view: stopStatus.count, type: 'u32', access: 'read'},
        {name: 'timestamps', view: props.timestamps, type: 'u32', access: 'read'},
        {name: 'times', view: times, type: 'u32', access: 'read_write'},
        {name: 'live', view: live, type: 'u32', access: 'read_write'},
        {name: 'ones', view: ones, type: 'f32', access: 'read_write'}
      ],
      body: `let row = rows[rowsOffset + index]; let liveStop = index < count[countOffset] && row < ${props.timestamps.length}u;
live[liveOffset + index] = u32(liveStop); ones[onesOffset + index] = 1.0;
var time = vec2<u32>(0u); if (liveStop) { time = vec2<u32>(timestamps[timestampsOffset + 2u * row], timestamps[timestampsOffset + 2u * row + 1u]); }
times[timesOffset + 2u * index] = time.x; times[timesOffset + 2u * index + 1u] = time.y;`
    })
  );
  const invalidCount = view('invalid-count', 'uint32', 1);
  addRecipeCountNodes(builder, {
    id: `${id}-invalid`,
    operation: id,
    rowCount,
    bindings: [
      {
        name: 'rows',
        view: props.time === 'start' ? props.stops.startRows : props.stops.finishRows,
        type: 'u32',
        access: 'read'
      },
      {name: 'count', view: stopStatus.count, type: 'u32', access: 'read'}
    ],
    counts: [
      {
        name: 'invalid',
        output: invalidCount,
        value: `u32(index < count[countOffset] && rows[rowsOffset + index] >= ${props.timestamps.length}u) + select(0u, count[countOffset] - min(count[countOffset], ${rowCount}u), index == 0u)`
      }
    ]
  });
  const periods = view('periods', 'uint32', rowCount);
  builder.add(
    new GPUPeriodIndex({
      id: `${id}-period`,
      temporalContext: props.temporalContext,
      periods: props.periods,
      timestamps: times,
      mask: live,
      output: {periods}
    })
  );
  const keys = view('keys', 'uint32x2', rowCount);
  builder.add(
    new GPUPointToCell({
      id: `${id}-cells`,
      family: props.family,
      resolution: props.resolution,
      positions: props.stops.centroids,
      mask: live,
      output: {cells: keys}
    })
  );
  const cellRows = view('cell-rows', 'uint32', rowCount);
  builder.add(
    new GPUKeyLookup({
      id: `${id}-join`,
      leftKeys: keys,
      rightKeys: props.dictionary,
      leftMask: live,
      output: {rightRows: cellRows}
    })
  );
  const counts =
    output?.counts ?? view('counts', 'uint32', props.dictionary.length * props.periods.count);
  validateSinglePackedView(counts, ['uint32'], `${id} output.counts`);
  validateViewLength(id, 'output.counts', counts, props.dictionary.length * props.periods.count);
  validateSinglePackedView(stopStatus.count, ['uint32'], `${id} stops.status.count`);
  validateRequirement(
    id,
    stopStatus.count.length >= 1,
    'stops.status.count must contain at least one word'
  );
  const inputViews = [
    props.timestamps,
    props.stops.centroids,
    props.stops.startRows,
    props.stops.finishRows,
    stopStatus.count,
    props.dictionary,
    props.weights.offsets,
    props.weights.neighbors,
    props.weights.weights,
    ...(props.weights.status?.count ? [props.weights.status.count] : []),
    ...(props.weights.status?.requiredCount ? [props.weights.status.requiredCount] : []),
    ...validateWeightsLossSources(id, lossInputs)
  ];
  validateGraphViewsBelongToGraph(id, graph, [counts, ...inputViews]);
  validateGraphOutputsDisjointFromInputs(id, [counts], inputViews);
  builder.add(
    new GPUValueCube({
      id: `${id}-cube`,
      cells: cellRows,
      periods,
      values: ones,
      mask: live,
      cellCount: props.dictionary.length,
      periodCount: props.periods.count,
      output: {counts}
    })
  );
  const losses = getWeightsLossSources(lossInputs);
  const status = {
    invalidCount,
    overflow: view('overflow', 'uint32', 1),
    candidateOverflow: view('candidate-overflow', 'uint32', 1)
  };
  for (const name of ['overflow', 'candidateOverflow'] as const)
    builder.addGlue(
      name,
      createWeightsLossNodes(graph, `${id}-${name}`, losses[name], status[name])
    );
  if (props.weights.status?.requiredCount) {
    // Cardinality is final topology loss even when the optional producer flags are absent.
    // Reuse the shared count/terminal-offset/capacity checks without relabelling candidate loss.
    const cardinality = getRegressionWeightsLossNodes(graph, `${id}-cardinality`, {
      ...props.weights,
      status: {count: props.weights.status.count, requiredCount: props.weights.status.requiredCount}
    });
    builder.addGlue('cardinality', [
      ...cardinality.nodes,
      ...createWeightsLossNodes(
        graph,
        `${id}-cardinality-final`,
        [cardinality.loss!],
        status.overflow,
        true
      )
    ]);
  }
  builder.addGlue(
    'suppress',
    createWGSLKernelNode(graph, {
      id: `${id}-suppress`,
      operation: id,
      invocationCount: counts.length,
      bindings: [
        {name: 'overflow', view: status.overflow, type: 'u32', access: 'read'},
        {name: 'candidate', view: status.candidateOverflow, type: 'u32', access: 'read'},
        {name: 'invalid', view: invalidCount, type: 'u32', access: 'read'},
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
      ],
      body: 'if (overflow[overflowOffset] != 0u || candidate[candidateOffset] != 0u || invalid[invalidOffset] != 0u) { counts[countsOffset + index] = 0u; }'
    })
  );
  return {
    contributors: builder.contributors,
    outputs: {counts},
    intermediates: {periods, cellRows, live, times},
    weights: snapshotWeightsPort(props.weights),
    status: {
      ...status,
      stages: [
        {stage: 'stops', status: stopStatus},
        ...sliceStatuses.map((status, index) => ({
          stage: `slice-${index}` as const,
          status: {...status}
        })),
        {stage: 'cube', status}
      ]
    }
  };
}
