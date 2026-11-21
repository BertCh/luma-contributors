// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateOverlayScratchByteLengthsForDevice,
  getOverlayOutputViews
} from './overlay-storage-validation';

import type {Device} from '@luma.gl/core';
import {validateRingAssemblyDimensions} from '../../analysis-core/ring-assembly/ring-assembly-dimensions';
import {validateOverlayOptions} from './overlay-input-validation';
import {
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {type GPUCommandGraph} from '@luma.gl/graph/gpu-core';

import {
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateCapacity,
  validatePowerOfTwo,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {getGPUSnapRoundedNodingMaximumVertexCount} from '../geometry-topology/gpu-snap-rounded-noding';
import {getGPUSegmentIntersectionMaximumVertexCount} from '../segment-intersection/index';

import type {GPUPolygonOverlayProps, GPUPolygonOverlayPlan} from './gpu-polygon-overlay';

/** Side-effect-free device, binding and scratch preflight. @internal */
export function getPolygonOverlayPlan<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUPolygonOverlayProps
): GPUPolygonOverlayPlan {
  const {left, right, output} = props;
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(
      [
        left.positions,
        left.featureOffsets,
        left.polygonOffsets,
        left.ringOffsets,
        left.sourceIds,
        right?.positions,
        right?.featureOffsets,
        right?.polygonOffsets,
        right?.ringOffsets,
        right?.sourceIds,
        ...getOverlayOutputViews(output)
      ].map((view, index) => [`binding.${index}`, view])
    )
  );
  validateGraphViewsBelongToGraph(id, graph, [
    left.positions,
    left.featureOffsets,
    left.polygonOffsets,
    left.ringOffsets,
    right?.positions,
    right?.featureOffsets,
    right?.polygonOffsets,
    right?.ringOffsets,
    output.geometry.positions,
    output.geometry.featureOffsets,
    output.geometry.polygonOffsets,
    output.geometry.ringOffsets,
    output.boundary.endpoints,
    output.boundary.operandIds,
    output.boundary.sourceFeatureIds,
    ...Object.values(output.boundary.status),
    ...Object.values(output.status)
  ]);
  return planGPUPolygonOverlay(graph.device, {
    id,
    operation: props.operation,
    fillRule: props.fillRule,
    vertexTolerance: props.vertexTolerance,
    spatialContext: props.spatialContext,
    rightSpatialContext: props.rightSpatialContext,
    left: {vertexCount: left.positions.length, ringCount: left.ringOffsets.length - 1},
    right: right && {vertexCount: right.positions.length, ringCount: right.ringOffsets.length - 1},
    capacity: props.capacity,
    leafCapacity: props.leafCapacity,
    output: {
      vertexCapacity: output.geometry.positions.length,
      ringCapacity: output.geometry.ringOffsets.length - 1,
      boundaryCapacity: output.boundary.endpoints.length
    }
  });
}

/** Operand allocation dimensions, including padded vertices/rings rather than live counts. */
export type GPUPolygonOverlayOperandDimensions = Readonly<{vertexCount: number; ringCount: number}>;

/** Device-only planning input; no graph or resident resource is required. Counts are total rows. */
export type GPUPolygonOverlayPlanningProps = Pick<
  GPUPolygonOverlayProps,
  | 'operation'
  | 'fillRule'
  | 'vertexTolerance'
  | 'capacity'
  | 'leafCapacity'
  | 'spatialContext'
  | 'rightSpatialContext'
> & {
  id?: string;
  left: GPUPolygonOverlayOperandDimensions;
  right?: GPUPolygonOverlayOperandDimensions;
  output: Readonly<{vertexCapacity: number; ringCapacity: number; boundaryCapacity: number}>;
};

/** Validates dimensions/options and every derived device scratch binding before allocating views. */
export function planGPUPolygonOverlay(
  device: Device,
  props: GPUPolygonOverlayPlanningProps
): GPUPolygonOverlayPlan {
  const id = props.id ?? 'polygon-overlay';
  validateOverlayOptions(id, props.operation, props.fillRule);
  validateRequirement(
    id,
    props.operation === 'dissolve' ? !props.right : Boolean(props.right),
    props.operation === 'dissolve'
      ? 'dissolve accepts only the left operand'
      : `${props.operation} requires the right operand`
  );
  const context = resolveGPUSpatialContext(id, props.spatialContext);
  validateRequirement(
    id,
    props.rightSpatialContext === undefined || Boolean(props.right),
    'rightSpatialContext needs right'
  );
  resolveGPUSecondarySpatialContext(id, 'right', props.rightSpatialContext, context);
  validateFiniteNumber(id, 'vertexTolerance', props.vertexTolerance, 0, 2 ** 127);
  validateFiniteNumber(
    id,
    'vertexTolerance converted to float32',
    Math.fround(props.vertexTolerance),
    2 ** -126,
    2 ** 127
  );
  for (const [name, operand] of [
    ['left', props.left],
    ['right', props.right]
  ] as const) {
    if (!operand) continue;
    validateCapacity(id, `${name}.vertexCount`, operand.vertexCount, 0);
    validateCapacity(id, `${name}.ringCount`, operand.ringCount, 1);
  }
  validateCapacity(id, 'capacity.intersections', props.capacity.intersections);
  validateCapacity(id, 'capacity.nodedSegments', props.capacity.nodedSegments);
  for (const [name, count] of Object.entries(props.output))
    validateCapacity(id, `output.${name}`, count, 0);
  if (props.leafCapacity !== undefined)
    validatePowerOfTwo(`${id}-noding`, 'leafCapacity', props.leafCapacity);
  validateRingAssemblyDimensions(
    `${id}-assemble`,
    props.output.boundaryCapacity,
    props.output.ringCapacity,
    props.output.vertexCapacity
  );
  const inputRingCount = props.left.ringCount + (props.right?.ringCount ?? 0);
  const expandedVertexCount =
    props.left.vertexCount + (props.right?.vertexCount ?? 0) + inputRingCount;
  validateOverlayScratchByteLengthsForDevice(device, id, {
    expandedVertexCount,
    inputRingCount,
    intersections: props.capacity.intersections,
    nodedSegments: props.capacity.nodedSegments,
    boundaryCapacity: props.output.boundaryCapacity,
    outputRingCapacity: props.output.ringCapacity,
    outputVertexCapacity: props.output.vertexCapacity,
    leafCapacity: props.leafCapacity
  });
  validateMaximumCount(
    id,
    'left + right positions rows + rings',
    expandedVertexCount,
    getGPUSegmentIntersectionMaximumVertexCount(device)
  );
  validateMaximumCount(
    id,
    'left + right positions rows + rings + capacity.intersections',
    expandedVertexCount + props.capacity.intersections,
    getGPUSnapRoundedNodingMaximumVertexCount(device, 0)
  );
  return Object.freeze({
    expandedVertexCount,
    inputRingCount,
    hotPixelCapacity: expandedVertexCount + props.capacity.intersections,
    nodedSegmentCapacity: props.capacity.nodedSegments,
    boundaryCapacity: props.output.boundaryCapacity,
    outputRingCapacity: props.output.ringCapacity,
    outputVertexCapacity: props.output.vertexCapacity
  });
}
