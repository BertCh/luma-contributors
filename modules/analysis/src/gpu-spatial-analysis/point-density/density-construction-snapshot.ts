// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  isGraphDataView,
  isGraphVectorView
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

type DensityMetadataSchema = {
  readonly [field: string]: DensityMetadataSchema | 'tuple' | null;
};

const SPATIAL_CONTEXT_SCHEMA: DensityMetadataSchema = {
  coordinateSpace: null,
  metric: null,
  units: null,
  sphereRadius: null,
  ellipsoid: {semiMajorAxis: null, flattening: null},
  edgeModel: null,
  datum: null,
  gridFamily: null,
  gridResolution: null,
  frame: {
    id: null,
    kind: null,
    origin: 'tuple',
    crs: null,
    signature: null,
    unit: {name: null, metersPerUnit: null, degreesPerUnit: null},
    axisOrder: null,
    areaOfUse: 'tuple',
    anchor: null,
    precision: null
  }
};
const TEMPORAL_CONTEXT_SCHEMA: DensityMetadataSchema = {
  unit: null,
  epoch: null,
  encoding: null,
  timeZone: null,
  axis: {kind: null, step: {calendarUnit: null, interval: null}}
};
const GRID_SCHEMA: DensityMetadataSchema = {
  width: null,
  height: null,
  affine: 'tuple',
  pixelInterpretation: null,
  coordinateReferenceSystem: {authority: null, wellKnownText: null, projectionJson: null},
  level: null,
  levelZeroOrigin: 'tuple'
};

/** Copy only declared tuple elements; application properties and back-references are not metadata. */
function snapshotDensityTuple(value: unknown): unknown {
  if (Array.isArray(value)) return Object.freeze([...value]);
  if (ArrayBuffer.isView(value)) return structuredClone(value);
  return value;
}

/** Explicit schemas avoid reading unrelated getters or following application-layer cycles. */
function snapshotDensityMetadata(value: unknown, schema: DensityMetadataSchema): unknown {
  if (!value || typeof value !== 'object') return value;
  const captured = snapshotConstructionFields(
    value as Record<string, unknown>,
    Object.keys(schema)
  );
  return Object.freeze(
    Object.fromEntries(
      Object.entries(captured).map(([field, item]) => {
        const nested = schema[field];
        return [
          field,
          nested === 'tuple'
            ? snapshotDensityTuple(item)
            : nested
              ? snapshotDensityMetadata(item, nested)
              : item
        ];
      })
    )
  );
}

/** @internal Snapshot compile-time options while retaining live graph views and caller buffers. */
export function snapshotDensityConstructionProps<Props extends object>(
  props: Props,
  fields: readonly (keyof Props)[],
  outputFields: readonly string[],
  statusFields: readonly string[] = []
): Props {
  const captured = snapshotConstructionFields(props, fields);
  const result = {...captured} as Record<string, unknown>;
  for (const [field, schema] of Object.entries({
    grid: GRID_SCHEMA,
    spatialContext: SPATIAL_CONTEXT_SCHEMA,
    temporalContext: TEMPORAL_CONTEXT_SCHEMA
  })) {
    if (field in result) result[field] = snapshotDensityMetadata(result[field], schema);
  }
  for (const field of ['indexBounds', 'indexGridSize', 'bounds', 'gridSize']) {
    if (field in result) result[field] = snapshotDensityTuple(result[field]);
  }
  const smoothing = result['smoothing'];
  if (smoothing && typeof smoothing === 'object') {
    const copy = {
      ...snapshotConstructionFields(smoothing as Record<string, unknown>, [
        'kernel',
        'separableKernel',
        'kernelWidth',
        'kernelHeight',
        'strategy'
      ])
    };
    const separable = copy['separableKernel'];
    if (separable && typeof separable === 'object') {
      copy['separableKernel'] = snapshotConstructionFields(separable as Record<string, unknown>, [
        'horizontal',
        'vertical'
      ]);
    }
    result['smoothing'] = Object.freeze(copy);
  }
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
  const window = result['studyWindow'];
  if (window && typeof window === 'object') {
    const record = window as {
      kind: string;
      id: string;
      spatialContext: object;
      getViews: (...parameters: unknown[]) => unknown;
    };
    const capturedWindow = snapshotConstructionFields(record, [
      'kind',
      'id',
      'spatialContext',
      'getViews'
    ]);
    result['studyWindow'] = Object.freeze({
      ...capturedWindow,
      spatialContext: snapshotDensityMetadata(
        capturedWindow.spatialContext,
        SPATIAL_CONTEXT_SCHEMA
      ),
      getViews:
        typeof capturedWindow.getViews === 'function'
          ? capturedWindow.getViews.bind(window)
          : capturedWindow.getViews
    });
  }
  return Object.freeze(result) as Props;
}
