// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';

function captureWeightsPublicFields(
  object: object,
  declared: readonly string[] = []
): Record<PropertyKey, unknown> {
  const fields = new Set<PropertyKey>(declared);
  for (
    let prototype: object | null = object;
    prototype && prototype !== Object.prototype;
    prototype = Object.getPrototypeOf(prototype)
  )
    for (const field of Reflect.ownKeys(prototype)) if (field !== 'constructor') fields.add(field);
  return Object.fromEntries(
    [...fields].flatMap(field => {
      const value = Reflect.get(object, field);
      return value !== undefined || field in object ? [[field, value]] : [];
    })
  );
}

/** Capture the public shape, including prototype getters, without traversing opaque graph values. @internal */
export function snapshotWeightsDescription<T extends object>(value: T): T {
  const captured = captureWeightsPublicFields(value);
  const status = captured['status']
    ? captureWeightsPublicFields(captured['status'] as object)
    : undefined;
  const information = captured['information']
    ? captureWeightsPublicFields(captured['information'] as object)
    : undefined;
  const provenance = information?.['provenance']
    ? captureWeightsPublicFields(information['provenance'] as object)
    : undefined;
  const chunks = (captured['chunks'] as readonly object[] | undefined)?.map(chunk => {
    const fields = captureWeightsPublicFields(chunk);
    return Object.freeze({
      ...fields,
      weights: snapshotWeightsDescription(fields['weights'] as object)
    });
  });
  const fields = {
    ...captured,
    ...(status ? {status} : {}),
    ...(information ? {information: {...information, ...(provenance ? {provenance} : {})}} : {}),
    ...(chunks ? {chunks} : {})
  };
  const core = snapshotWeightsPort(fields);
  const coreInformation = core.information as typeof information;
  return Object.freeze({
    ...fields,
    ...core,
    ...(information
      ? {
          information: Object.freeze({
            ...information,
            ...coreInformation,
            ...(provenance
              ? {
                  provenance: Object.freeze({
                    ...provenance,
                    ...(coreInformation?.['provenance'] as object)
                  })
                }
              : {})
          })
        }
      : {}),
    ...(status ? {status: Object.freeze({...status, ...(core.status as object)})} : {}),
    ...(chunks ? {chunks: Object.freeze(chunks)} : {})
  }) as T;
}

/** @internal Capture context/frame getters once before the shared tuple-copying snapshot. */
export function snapshotWeightsSpatialContext(context: GPUSpatialContext): GPUSpatialContext {
  const captured = captureWeightsPublicFields(context, [
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
  ]) as unknown as GPUSpatialContext;
  const frame = captured.frame
    ? (captureWeightsPublicFields(captured.frame, [
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
      ]) as unknown as NonNullable<GPUSpatialContext['frame']>)
    : undefined;
  const ellipsoid = captured.ellipsoid
    ? (captureWeightsPublicFields(captured.ellipsoid, [
        'semiMajorAxis',
        'flattening'
      ]) as unknown as GPUSpatialContext['ellipsoid'])
    : undefined;
  const unit = frame?.unit
    ? (captureWeightsPublicFields(frame.unit, [
        'name',
        'metersPerUnit',
        'degreesPerUnit'
      ]) as unknown as NonNullable<typeof frame>['unit'])
    : undefined;
  const normalized = snapshotConstructionSpatialContext({
    ...captured,
    ...(ellipsoid ? {ellipsoid} : {}),
    ...(frame ? {frame: {...frame, ...(unit ? {unit} : {})}} : {})
  });
  return Object.freeze({
    ...captured,
    ...normalized,
    ...(ellipsoid ? {ellipsoid: Object.freeze({...ellipsoid, ...normalized.ellipsoid})} : {}),
    ...(frame
      ? {
          frame: Object.freeze({
            ...frame,
            ...normalized.frame,
            ...(unit ? {unit: Object.freeze({...unit, ...normalized.frame?.unit})} : {})
          })
        }
      : {})
  });
}

/** @internal Capture only declared weights-family fields, including inherited getters. */
export function snapshotWeightsFields<T extends object>(value: T, fields: readonly string[]): T {
  return snapshotConstructionFields(value, fields as readonly (keyof T)[]);
}

/** Capture one coherent set of named validation inputs and outputs. @internal */
export function snapshotWeightsValidationOperationProps<T extends object>(value: T): T {
  const props = snapshotWeightsFields(value, ['id', 'rowCount', 'weights', 'output']) as T & {
    weights: object;
    output: {status: object};
  };
  const weights = snapshotWeightsFields(props.weights, ['offsets', 'neighbors', 'weights']);
  const output = snapshotWeightsFields(props.output, ['flags', 'status']);
  return Object.freeze({
    ...props,
    weights,
    output: Object.freeze({
      ...output,
      status: snapshotWeightsFields(output.status, ['invalidCount'])
    })
  });
}

/** @internal Freeze port maps and topology metadata, preserving per-submission GPU contents. */
export function snapshotWeightsPort<T extends object>(value: T): T {
  type Port = {
    status?: object;
    information?: {provenance?: object};
    spatialContext?: GPUSpatialContext;
    chunks?: readonly {weights: object}[];
  };
  const port = snapshotWeightsFields(value, [
    'offsets',
    'neighbors',
    'weights',
    // Named regression HAC kernels expose loss graph-value names directly on the kernel.
    'overflow',
    'candidateOverflow',
    'distances',
    'neighborCounts',
    'status',
    'information',
    'spatialContext',
    'rowCount',
    'chunks',
    'asymmetricSlots'
  ]) as T & Port;
  const information = port.information
    ? snapshotWeightsFields(port.information, [
        'rowCount',
        'transform',
        'symmetric',
        'hasSelfWeights',
        'provenance'
      ])
    : undefined;
  return Object.freeze({
    ...port,
    ...(port.status
      ? {
          status: snapshotWeightsFields(port.status, [
            'overflow',
            'candidateOverflow',
            'complete',
            'requiredCount',
            'count',
            'invalidCount',
            'approximation',
            'uncertainCount',
            'unresolvedCount',
            'iterationCount',
            'converged',
            'iterationLimitReached'
          ])
        }
      : {}),
    ...(information
      ? {
          information: Object.freeze({
            ...information,
            ...(information.provenance
              ? {
                  provenance: snapshotWeightsFields(information.provenance, [
                    'source',
                    'criterion',
                    'k',
                    'radius',
                    'order'
                  ])
                }
              : {})
          })
        }
      : {}),
    ...(port.spatialContext
      ? {spatialContext: snapshotWeightsSpatialContext(port.spatialContext)}
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
