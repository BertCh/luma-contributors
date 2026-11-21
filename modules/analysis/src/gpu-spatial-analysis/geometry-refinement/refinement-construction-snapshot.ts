// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';

/** Capture frame accessors before delegating nested copies to the shared metadata helper. */
export function snapshotRefinementSpatialContext(context: GPUSpatialContext): GPUSpatialContext {
  const snapshot = snapshotConstructionFields(context, [
    'coordinateSpace',
    'metric',
    'units',
    'sphereRadius',
    'ellipsoid',
    'edgeModel',
    'datum',
    'gridFamily',
    'gridResolution',
    'frame'
  ]);
  return snapshotConstructionSpatialContext({
    ...snapshot,
    ...(snapshot.frame
      ? {
          frame: snapshotConstructionFields(snapshot.frame, [
            'id',
            'kind',
            'origin',
            'crs',
            'signature',
            'unit',
            'axisOrder',
            'areaOfUse',
            'anchor',
            'precision'
          ])
        }
      : {})
  });
}

const CIRCLE_OPERATION_FIELDS = {
  props: [
    'spatialContext',
    'ringOffsets',
    'polygonOffsets',
    'featureOffsets',
    'featureCount',
    'tolerance',
    'cellCapacity',
    'maximumIterations'
  ],
  output: [
    'circles',
    'witnesses',
    'status',
    'publishedCenterClearance',
    'precisionLimited',
    'witnessOffsets'
  ]
} as const;

const OPERATION_FIELDS = {
  circle: CIRCLE_OPERATION_FIELDS,
  'cpu-circle': {...CIRCLE_OPERATION_FIELDS, props: [...CIRCLE_OPERATION_FIELDS.props, 'storage']},
  convex: {
    props: [
      'spatialContext',
      'ringOffsets',
      'polygonCount',
      'maximumVertices',
      'maximumIterations',
      'offsetDistance',
      'eventTolerance',
      'segmentCapacity'
    ],
    output: ['segments', 'sourceEdges', 'offsetPositions', 'status']
  },
  vector: {
    props: [
      'ringOffsets',
      'polygonOffsets',
      'featureOffsets',
      'polygonCount',
      'maximumVertices',
      'maximumEvents',
      'offsetDistance',
      'eventTolerance',
      'segmentCapacity'
    ],
    output: ['segments', 'sourceEdges', 'offsetPositions', 'offsetLinks', 'status']
  },
  hull: {
    props: ['spatialContext', 'pointCount', 'ratio', 'allowHoles', 'maximumIterations'],
    output: ['triangles', 'triangleValidity', 'boundaryEdges', 'edgeValidity', 'status']
  }
} as const;

/**
 * Capture only the calling operation's documented options and ports, exactly once.
 * `spatialContext` is listed per operation: vector skeleton props do not declare it.
 */
export function snapshotRefinementOperationProps<
  Props extends {output: object; spatialContext?: GPUSpatialContext}
>(props: Props, operation: keyof typeof OPERATION_FIELDS): Props {
  const fields: {readonly props: readonly string[]; readonly output: readonly string[]} =
    OPERATION_FIELDS[operation];
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'positions',
    'output',
    ...fields.props
  ] as (keyof Props)[]);
  const output = snapshotConstructionFields<Props['output']>(
    snapshot.output,
    fields.output as readonly (keyof Props['output'])[]
  );
  const status = (output as {status?: unknown}).status;
  return Object.freeze({
    ...snapshot,
    ...(snapshot.spatialContext
      ? {spatialContext: snapshotRefinementSpatialContext(snapshot.spatialContext)}
      : {}),
    output: Object.freeze({
      ...output,
      ...(operation === 'hull' && status && typeof status === 'object'
        ? {status: snapshotConstructionFields(status, ['converged'] as (keyof typeof status)[])}
        : {})
    })
  });
}
