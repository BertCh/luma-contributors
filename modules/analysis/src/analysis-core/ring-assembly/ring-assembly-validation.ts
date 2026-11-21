// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRingAssemblyDimensions} from './ring-assembly-dimensions';
import type {GPUSegmentRingAssemblyProps} from './segment-ring-assembly-types';

import {validateRingOutputDispatches, getOutputViews} from './ring-output-validation';

import {validatePackedUint32View} from '@luma.gl/graph/gpu-core';

import {
  validateRequirement,
  validateFiniteNumber,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView
} from '../kernels/gpu-contributor-utils';

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../kernels/construction-snapshot';

const GPU_SEGMENT_RING_ASSEMBLY_MINIMUM_VERTEX_TOLERANCE = 2 ** -149;
/** Captures caller option maps before validation or lowering. @internal */
export function captureRingAssemblyProps(
  props: GPUSegmentRingAssemblyProps
): GPUSegmentRingAssemblyProps {
  validateRequirement(
    props.id ?? 'segment-ring-assembly',
    !props.count || !props.segmentStatus || props.count === props.segmentStatus.count,
    'count and segmentStatus.count must identify the same view'
  );
  validateRequirement(
    props.id ?? 'segment-ring-assembly',
    !props.output.status.polygonCount || Boolean(props.output.polygons),
    'output.status.polygonCount requires output.polygons'
  );
  validateRequirement(
    props.id ?? 'segment-ring-assembly',
    !props.segmentStatus || Boolean(props.segmentStatus.count && props.segmentStatus.overflow),
    'segmentStatus must supply count and overflow views'
  );
  props = Object.freeze({
    ...snapshotConstructionFields(props, [
      'id',
      'endpoints',
      'count',
      'segmentStatus',
      'groups',
      'pairedOpposites',
      'vertexTolerance',
      'interiorSide',
      'normalizeWinding',
      'spatialContext',
      'cancelOpposingSegments',
      'splitTouchingRings',
      'output'
    ]),
    count: props.count ?? props.segmentStatus?.count,
    segmentStatus:
      props.segmentStatus &&
      snapshotConstructionFields(props.segmentStatus, ['count', 'overflow', 'candidateOverflow']),
    spatialContext:
      props.spatialContext && snapshotConstructionSpatialContext(props.spatialContext),
    output: Object.freeze({
      ...snapshotConstructionFields(props.output, [
        'ringOffsets',
        'positions',
        'ringAreas',
        'ringIsHole',
        'ringShells',
        'ringGroups',
        'segmentRings',
        'segmentVertices',
        'segmentFlags',
        'polygons',
        'status'
      ]),
      status: snapshotConstructionFields(props.output.status, [
        'count',
        'overflow',
        'requiredCount',
        'candidateOverflow',
        'complete',
        'polygonCount',
        'openSegmentCount',
        'touchingSegmentCount'
      ]),
      polygons:
        props.output.polygons &&
        snapshotConstructionFields(props.output.polygons, [
          'kind',
          'positions',
          'ringOffsets',
          'polygonOffsets',
          'featureOffsets',
          'sourceIds'
        ])
    })
  });
  return props;
}

/** Construction-time segment and output layout checks. @internal */
export function validateRingAssemblyProps(id: string, props: GPUSegmentRingAssemblyProps): void {
  const {output} = props;
  validateSinglePackedView(props.endpoints, ['float32x4'], `${id} endpoints`);
  const segments = props.endpoints.length;
  validateRingAssemblyDimensions(
    id,
    segments,
    output.ringOffsets.length - 1,
    output.positions.length
  );
  if (props.vertexTolerance !== undefined) {
    validateFiniteNumber(
      id,
      'vertexTolerance',
      props.vertexTolerance,
      GPU_SEGMENT_RING_ASSEMBLY_MINIMUM_VERTEX_TOLERANCE
    );
  }
  if (props.interiorSide !== undefined && !['left', 'right'].includes(props.interiorSide)) {
    throw new Error(`${id} interiorSide must be 'left' or 'right'`);
  }
  for (const [name, view] of [
    ['count', props.count],
    ['segmentStatus.count', props.segmentStatus?.count],
    ['segmentStatus.overflow', props.segmentStatus?.overflow],
    ['segmentStatus.candidateOverflow', props.segmentStatus?.candidateOverflow],
    ['output.status.candidateOverflow', output.status.candidateOverflow],
    ['output.status.complete', output.status.complete],
    ['output.status.polygonCount', output.status.polygonCount],
    ['output.status.count', output.status.count],
    ['output.status.overflow', output.status.overflow],
    ['output.status.requiredCount', output.status.requiredCount],
    ['output.status.openSegmentCount', output.status.openSegmentCount],
    ['output.status.touchingSegmentCount', output.status.touchingSegmentCount]
  ] as const) {
    if (view) {
      validatePackedUint32View(view, `${id} ${name}`);
      if (view.length < 1) {
        throw new Error(`${id} ${name} must contain one uint32 row`);
      }
    }
  }
  if (props.groups) {
    validatePackedUint32View(props.groups, `${id} groups`);
    if (props.groups.length !== segments) {
      throw new Error(`${id} groups length must equal the segment row count`);
    }
  }
  if (output.ringGroups && !props.groups) {
    throw new Error(`${id} output.ringGroups needs the groups input`);
  }
  validatePackedUint32View(output.ringOffsets, `${id} output.ringOffsets`);

  validateSinglePackedView(output.positions, ['float32x2'], `${id} output.positions`);

  const ringCapacity = output.ringOffsets.length - 1;
  for (const [name, view] of [
    ['ringIsHole', output.ringIsHole],
    ['ringShells', output.ringShells],
    ['ringGroups', output.ringGroups]
  ] as const) {
    if (view) {
      validatePackedUint32View(view, `${id} output.${name}`);
      if (view.length !== ringCapacity) {
        throw new Error(`${id} output.${name} length must equal the ring capacity`);
      }
    }
  }
  if (output.ringAreas) {
    validateSinglePackedView(output.ringAreas, ['float32'], `${id} output.ringAreas`);
    if (output.ringAreas.length !== ringCapacity) {
      throw new Error(`${id} output.ringAreas length must equal the ring capacity`);
    }
  }
  if (output.polygons) {
    const {polygons} = output;
    validateSinglePackedView(polygons.positions, ['float32x2'], `${id} output.polygons.positions`);
    if (polygons.positions.length !== output.positions.length) {
      throw new Error(`${id} output.polygons.positions must match output.positions length`);
    }
    if (polygons.sourceIds) {
      if (!props.groups) {
        throw new Error(`${id} output.polygons.sourceIds needs the groups input`);
      }
      validatePackedUint32View(polygons.sourceIds, `${id} output.polygons.sourceIds`);
      if (polygons.sourceIds.length !== ringCapacity) {
        throw new Error(
          `${id} output.polygons.sourceIds must have ${ringCapacity} rows (the ring capacity)`
        );
      }
    }
    for (const [name, view, length] of [
      ['ringOffsets', polygons.ringOffsets, ringCapacity + 1],
      ['polygonOffsets', polygons.polygonOffsets, ringCapacity + 1],
      ['featureOffsets', polygons.featureOffsets, ringCapacity + 1]
    ] as const) {
      validatePackedUint32View(view, `${id} output.polygons.${name}`);
      if (view.length !== length) {
        throw new Error(`${id} output.polygons.${name} must have ${length} rows`);
      }
    }
  }
  for (const [name, view] of [
    ['segmentRings', output.segmentRings],
    ['segmentVertices', output.segmentVertices],
    ['segmentFlags', output.segmentFlags]
  ] as const) {
    if (view) {
      validatePackedUint32View(view, `${id} output.${name}`);
      if (view.length !== segments) {
        throw new Error(`${id} output.${name} length must equal the segment row count`);
      }
    }
  }
  validateRingOutputDispatches(id, output, Boolean(props.pairedOpposites));
  validateGraphOutputsDisjointFromInputs(id, getOutputViews(output), [
    props.endpoints,
    props.count,
    props.segmentStatus?.overflow,
    props.segmentStatus?.candidateOverflow,
    props.groups
  ]);
}
