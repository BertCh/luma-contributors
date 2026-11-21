// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  isGraphDataView,
  isGraphVectorView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';

/** Option wrappers may expose inherited or non-enumerable getters; capture each once. */
function getInterpolationObjectFields(value: object): string[] {
  const fields = new Set<string>();
  for (
    let prototype: object | null = value;
    prototype && prototype !== Object.prototype;
    prototype = Object.getPrototypeOf(prototype)
  ) {
    for (const field of Object.getOwnPropertyNames(prototype)) {
      if (field !== 'constructor') fields.add(field);
    }
  }
  return [...fields];
}

/** Capture plain option metadata recursively; this is never applied to a graph view or buffer. */
function snapshotInterpolationMetadata<Value>(value: Value): Value {
  if (Array.isArray(value)) {
    return Object.freeze(value.map(item => snapshotInterpolationMetadata(item))) as Value;
  }
  if (ArrayBuffer.isView(value)) return structuredClone(value);
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    const captured = snapshotConstructionFields(record, getInterpolationObjectFields(record));
    return Object.freeze(
      Object.fromEntries(
        Object.entries(captured).map(([key, item]) => [key, snapshotInterpolationMetadata(item)])
      )
    ) as Value;
  }
  return value;
}

/** @internal Snapshot compile-time options while retaining live graph views and caller buffers. */
export function snapshotInterpolationConstructionProps<Props extends object>(
  props: Props,
  fields: readonly (keyof Props)[],
  outputFields: readonly string[],
  statusFields: readonly string[] = []
): Props {
  const captured = snapshotConstructionFields(props, fields);
  const result = {...captured} as Record<string, unknown>;
  for (const field of [
    'covariance',
    'coregionalization',
    'coregionalizationModels',
    'means',
    'grid',
    'indexBounds',
    'indexGridSize',
    'temporalContext',
    'queryTemporalContext'
  ]) {
    if (field in result) result[field] = snapshotInterpolationMetadata(result[field]);
  }
  const spatialContext = result['spatialContext'] as GPUSpatialContext | undefined;
  if (spatialContext) result['spatialContext'] = snapshotConstructionSpatialContext(spatialContext);
  const neighborhoodWeights = result['neighborhoodWeights'];
  if (neighborhoodWeights && typeof neighborhoodWeights === 'object') {
    const weights = {
      ...snapshotConstructionFields(neighborhoodWeights as Record<string, unknown>, [
        'offsets',
        'neighbors',
        'status'
      ])
    };
    validateRequirement(
      String(result['id'] ?? 'Interpolation'),
      isGraphDataView(weights['offsets']) && isGraphDataView(weights['neighbors']),
      'neighborhoodWeights must supply packed offsets and neighbors graph views'
    );
    if (weights['status'])
      weights['status'] = snapshotConstructionFields(weights['status'] as Record<string, unknown>, [
        'overflow',
        'candidateOverflow',
        'complete'
      ]);
    for (const [legacy, field] of [
      ['neighborhoods', 'neighbors'],
      ['neighborhoodOffsets', 'offsets']
    ] as const) {
      validateRequirement(
        String(result['id'] ?? 'Interpolation'),
        !result[legacy] || result[legacy] === weights[field],
        `${legacy} must agree with neighborhoodWeights.${field}`
      );
      result[legacy] = weights[field];
    }
    result['neighborhoodWeights'] = Object.freeze(weights);
  }
  if (result['neighborhoodStatus'])
    result['neighborhoodStatus'] = snapshotConstructionFields(
      result['neighborhoodStatus'] as Record<string, unknown>,
      ['overflow', 'candidateOverflow', 'complete']
    );
  // Capture declared ports only; arbitrary prototype methods are not graph views.
  const output = result['output'];
  if (output && typeof output === 'object') {
    const record = output as Record<string, unknown>;
    const copy = {...snapshotConstructionFields(record, outputFields)};
    const status = copy['status'];
    // Scalar status views stay live; wrapper objects (including initially empty ones) are captured.
    if (
      statusFields.length > 0 &&
      status &&
      typeof status === 'object' &&
      !isGraphDataView(status) &&
      !isGraphVectorView(status)
    ) {
      copy['status'] = snapshotConstructionFields(status as Record<string, unknown>, statusFields);
    }
    result['output'] = Object.freeze(copy);
  }
  const conditioning = result['conditioning'];
  if (conditioning && typeof conditioning === 'object') {
    const copy = {
      ...snapshotConstructionFields(conditioning as Record<string, unknown>, [
        'indices',
        'values',
        'mask',
        'neighborhoods',
        'neighborCount',
        'neighborhoodOffsets',
        'status'
      ])
    };
    if (copy['status'])
      copy['status'] = snapshotConstructionFields(copy['status'] as Record<string, unknown>, [
        'overflow',
        'candidateOverflow',
        'complete'
      ]);
    result['conditioning'] = Object.freeze(copy);
  }
  return Object.freeze(result) as Props;
}
