// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUGraphPrimitive, GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateRequirement,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createFillNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPU_LINE_SIMPLIFICATION_DEFAULT_MAXIMUM_ITERATIONS,
  GPU_LINE_SIMPLIFICATION_DEFAULT_VISVALINGAM_MAXIMUM_ITERATIONS,
  GPU_LINE_SIMPLIFICATION_DEFAULT_FINISH_SPAN_LIMIT,
  GPU_LINE_SIMPLIFICATION_DEFAULT_NEIGHBORHOOD_RADIUS,
  GPU_LINE_SIMPLIFICATION_MAXIMUM_ITERATIONS,
  GPU_LINE_SIMPLIFICATION_MAXIMUM_FINISH_SPAN_LIMIT,
  GPU_LINE_SIMPLIFICATION_MAXIMUM_NEIGHBORHOOD_RADIUS
} from './gpu-line-simplification';
import type {LineSimplificationOperationProps} from './gpu-line-simplification-operation';
import {LINE_SIMPLIFICATION_STATUS_FIELDS} from './line-simplification-snapshot';

/** Publishes empty-work results for zero vertex rows or zero lines. */
export function createEmptyLineSimplification(
  props: LineSimplificationOperationProps,
  getView: (port: string) => GraphDataView
): GPUGraphPrimitive {
  const id = props.id ?? 'line-simplification';
  const method = props.method ?? 'douglas-peucker';
  const metric = props.metric ?? 'segment';
  validateRequirement(
    id,
    ['douglas-peucker', 'visvalingam'].includes(method),
    `method must be douglas-peucker or visvalingam; received ${method}`
  );
  validateRequirement(
    id,
    ['segment', 'time-ratio'].includes(metric),
    `metric must be segment or time-ratio; received ${metric}`
  );
  const context = resolveGPUSpatialContext(id, props.spatialContext, {
    ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
    measure: method === 'visvalingam' ? 'area' : 'distance'
  });
  validateRequirement(
    id,
    method !== 'visvalingam' || (context.coordinateSpace === 'planar' && metric === 'segment'),
    'method visvalingam requires planar coordinates and metric segment'
  );
  validateCapacity(
    id,
    'maximumIterations',
    props.maximumIterations ??
      (method === 'visvalingam'
        ? GPU_LINE_SIMPLIFICATION_DEFAULT_VISVALINGAM_MAXIMUM_ITERATIONS
        : GPU_LINE_SIMPLIFICATION_DEFAULT_MAXIMUM_ITERATIONS),
    1,
    GPU_LINE_SIMPLIFICATION_MAXIMUM_ITERATIONS
  );
  validateCapacity(
    id,
    'finishSpanLimit',
    props.finishSpanLimit ?? GPU_LINE_SIMPLIFICATION_DEFAULT_FINISH_SPAN_LIMIT,
    0,
    GPU_LINE_SIMPLIFICATION_MAXIMUM_FINISH_SPAN_LIMIT
  );
  if (method === 'visvalingam')
    validateCapacity(
      id,
      'neighborhoodRadius',
      props.neighborhoodRadius ?? GPU_LINE_SIMPLIFICATION_DEFAULT_NEIGHBORHOOD_RADIUS,
      1,
      GPU_LINE_SIMPLIFICATION_MAXIMUM_NEIGHBORHOOD_RADIUS
    );
  validateRequirement(
    id,
    metric !== 'time-ratio' || props.computeImportance === false || props.timestamps !== undefined,
    'metric time-ratio requires timestamps'
  );
  const positions = getView('positions');
  const offsets = getView('lineOffsets');
  const importance = getView('importance');
  validateSinglePackedView(positions, ['float32x2'], `${id} positions`);
  validateSinglePackedView(offsets, ['uint32'], `${id} lineOffsets`);
  validateSinglePackedView(importance, ['float32'], `${id} importance`);
  validateRequirement(
    id,
    positions.length === props.rowCount && importance.length === props.rowCount,
    `positions and importance length must equal rowCount ${props.rowCount}; received ${positions.length} and ${importance.length}`
  );
  validateRequirement(
    id,
    offsets.length === props.lineCount + 1,
    `lineOffsets length must equal lineCount + 1 (${props.lineCount + 1}); received ${offsets.length}`
  );
  const inputs: GraphDataView[] = [positions, offsets];
  if (props.computeImportance === false) inputs.push(importance);
  if (props.timestamps !== undefined) {
    const timestamps = getView('timestamps');
    validateSinglePackedView(timestamps, ['float32'], `${id} timestamps`);
    validateRequirement(
      id,
      timestamps.length === props.rowCount,
      `timestamps length must equal rowCount ${props.rowCount}; received ${timestamps.length}`
    );
    inputs.push(timestamps);
  }
  if (props.parameters !== undefined) {
    const parameters = getView('parameters');
    validateSinglePackedView(parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      parameters.length >= 4,
      `parameters must contain at least 4 elements; received ${parameters.length}`
    );
    inputs.push(parameters);
  }
  const outputViews: GraphDataView[] = [];
  const destinations: {port: string; view: GraphDataView; value: string}[] = [];
  if (props.computeImportance !== false) {
    outputViews.push(importance);
    destinations.push({port: 'importance', view: importance, value: '0.0'});
  }
  for (const port of ['ids', 'outputMask', 'lineCounts', 'lineStarts'] as const) {
    if (props.output?.[port] === undefined) continue;
    const view = getView(port);
    const length =
      port === 'ids'
        ? (props.capacity ?? props.rowCount)
        : port === 'outputMask'
          ? props.rowCount
          : props.lineCount;
    validateSinglePackedView(view, ['uint32'], `${id} ${port}`);
    validateRequirement(
      id,
      view.length === length,
      `${port} length must equal ${length}; received ${view.length}`
    );
    outputViews.push(view);
    // The ids tail remains unspecified, like a nonempty selection's unused capacity.
    if (port !== 'ids') destinations.push({port, view, value: '0u'});
  }
  for (const port of LINE_SIMPLIFICATION_STATUS_FIELDS) {
    if (props.output?.status?.[port] === undefined) continue;
    const view = getView(port);
    validateSinglePackedView(view, ['uint32'], `${id} ${port}`);
    validateRequirement(id, view.length >= 1, `${port} must contain at least one scalar`);
    outputViews.push(view);
    destinations.push({
      port,
      view,
      value: port === 'converged' || port === 'iterationCount' ? '1u' : '0u'
    });
  }
  validateGraphOutputsDisjointFromInputs(id, outputViews, inputs);
  return {
    getCommandNodes: <Parameters>(graph: GPUCommandGraph<Parameters>) =>
      destinations
        .filter(destination => destination.view.length > 0)
        .map(({port, view, value}) =>
          createFillNode<Parameters>(graph, {
            id: `${id}-empty-${port}`,
            operation: 'GPULineSimplification',
            view,
            type: port === 'importance' ? 'f32' : 'u32',
            value,
            componentCount: LINE_SIMPLIFICATION_STATUS_FIELDS.includes(
              port as (typeof LINE_SIMPLIFICATION_STATUS_FIELDS)[number]
            )
              ? 1
              : view.length
          })
        )
  };
}
