// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';

/** @internal Capture only declared weights-family fields, including inherited getters. */
export function snapshotWeightsFields<T extends object>(value: T, fields: readonly string[]): T {
  return snapshotConstructionFields(value, fields as readonly (keyof T)[]);
}

/** @internal Freeze port maps and topology metadata, preserving per-submission GPU contents. */
export function snapshotWeightsPort<T extends object>(value: T): T {
  type Port = {
    status?: object;
    information?: object;
    spatialContext?: GPUSpatialContext;
    chunks?: readonly {weights: object}[];
  };
  const port = snapshotWeightsFields(value, [
    'offsets',
    'neighbors',
    'weights',
    'distances',
    'status',
    'information',
    'spatialContext',
    'rowCount',
    'chunks',
    'asymmetricSlots'
  ]) as T & Port;
  return Object.freeze({
    ...port,
    ...(port.status
      ? {
          status: snapshotWeightsFields(port.status, [
            'overflow',
            'candidateOverflow',
            'requiredCount',
            'count',
            'invalidCount',
            'approximation',
            'uncertainCount',
            'iterationCount',
            'converged',
            'iterationLimitReached'
          ])
        }
      : {}),
    ...(port.information
      ? {
          information: snapshotWeightsFields(port.information, [
            'rowCount',
            'transform',
            'symmetric',
            'hasSelfWeights',
            'provenance'
          ])
        }
      : {}),
    ...(port.spatialContext
      ? {spatialContext: snapshotConstructionSpatialContext(port.spatialContext)}
      : {}),
    ...(port.chunks
      ? {
          chunks: Object.freeze(
            port.chunks.map(chunk => {
              const captured = snapshotWeightsFields(chunk, ['rowStart', 'rowCount', 'weights']);
              return Object.freeze({...captured, weights: snapshotWeightsPort(captured.weights)});
            })
          )
        }
      : {})
  }) as T;
}

/** @internal Snapshot the packed/chunked contributor and named-operation discriminated props. */
export function snapshotWeightsAlgebraProps<T extends object>(value: T): T {
  type Maps = {
    weights?: object;
    left?: object;
    right?: object;
    output: object;
    lags?: readonly object[];
  };
  const props = snapshotWeightsFields(value, [
    'id',
    'operation',
    'output',
    'weights',
    'left',
    'right',
    'selfWeight',
    'mask',
    'groupIds',
    'groupCount',
    'order',
    'cumulative',
    'workCapacity',
    'periodCount',
    'layout',
    'lags',
    'weightRule',
    'rowCount',
    'capacity'
  ]) as T & Maps;
  return Object.freeze({
    ...props,
    output: snapshotWeightsPort(props.output),
    ...Object.fromEntries(
      (['weights', 'left', 'right'] as const).flatMap(name =>
        props[name] ? [[name, snapshotWeightsPort(props[name]!)]] : []
      )
    ),
    ...(props.lags
      ? {
          lags: Object.freeze(
            props.lags.map(lag => snapshotWeightsFields(lag, ['lag', 'weight', 'includeSelf']))
          )
        }
      : {})
  });
}

/** @internal Contiguity maps retain opaque graph value names or views, with the same frozen fields. */
export function snapshotContiguityOperationProps<T extends object>(value: T): T {
  const props = snapshotWeightsFields(value, [
    'id',
    'criterion',
    'polygons',
    'featureCount',
    'capacity',
    'snapTolerance',
    'coordinateExtent',
    'pairCapacity',
    'output'
  ]) as T & {polygons: object; output: object};
  return Object.freeze({
    ...props,
    polygons: snapshotWeightsFields(props.polygons, [
      'kind',
      'positions',
      'ringOffsets',
      'polygonOffsets',
      'featureOffsets'
    ]),
    output: snapshotWeightsPort(props.output)
  });
}
