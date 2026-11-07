// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {
  CatalogCoordinateSystem,
  CatalogDiagnostic,
  CatalogFeatureCollection,
  CatalogOutputTable
} from './catalog-geojson';
import type {CatalogNetworkParameters} from './catalog-network';

/** Engine that produced a result. */
export type CatalogEngine = 'cpu-reference' | 'gpu';

/** Engines a binding supports. */
export type CatalogEngineSupport = 'cpu-reference' | 'gpu-or-cpu' | 'gpu';

/** Device handed to GPU runners; the caller owns it. */
export type CatalogGpuContext = {device: Device};

/** Tables, diagnostics and summary of one run. */
export type CatalogRunOutput = {
  tables: CatalogOutputTable[];
  diagnostics: CatalogDiagnostic[];
  summary: Record<string, string | number>;
  /** GPU contributors the run dispatched (GPU runners only). */
  dispatchedContributors?: readonly string[];
};

export type CatalogParameterValue = string | number | boolean | readonly number[];

export type CatalogParameterSpec = {
  name: string;
  label: string;
  kind: 'number' | 'integer' | 'boolean' | 'enum' | 'string' | 'number-list';
  /** Physical unit of the value, `crs` when it depends on the declared coordinate system. */
  unit?: string;
  minimum?: number;
  maximum?: number;
  options?: readonly string[];
  maximumLength?: number;
  defaultValue: CatalogParameterValue;
};

export type CatalogInputSpec = {
  name: string;
  label: string;
  accepts: readonly ('point' | 'line' | 'polygon')[];
  maximumFeatures: number;
  maximumVertices: number;
};

export type CatalogOutputSpec = {
  tableId: string;
  label: string;
  geometry: 'point' | 'line' | 'polygon' | 'none';
  /** Columns that tie each output row to the source feature and to its new identity. */
  identityColumns: readonly string[];
  maximumRows: number;
};

export type CatalogOperationBinding = {
  id: string;
  title: string;
  domain:
    | 'geometry'
    | 'network'
    | 'aggregation'
    | 'interpolation'
    | 'density'
    | 'clustering'
    | 'raster'
    | 'analysis';
  summary: string;
  /** Ontology capability families this operation belongs to. */
  capabilityIds: readonly string[];
  /** GPU contributor exports whose semantics this binding mirrors on the CPU. */
  contributorNames: readonly string[];
  /** GPU contributor exports the GPU path dispatches. Empty when only the CPU reference exists. */
  gpuContributorNames: readonly string[];
  /**
   * Execution support: 'cpu-reference' runs only on the CPU; 'gpu-or-cpu' dispatches the GPU
   * contributors when a device is supplied and keeps the CPU engine as oracle and fallback;
   * 'gpu' needs a device and has no CPU engine.
   */
  engine: CatalogEngineSupport;
  inputs: readonly CatalogInputSpec[];
  parameters: readonly CatalogParameterSpec[];
  outputs: readonly CatalogOutputSpec[];
  /** Resource ownership: results are plain arrays; publication goes through native Sources. */
  cleanup: string;
  /** Behavior for invalid, degenerate or capacity-exceeding input. */
  declaredBehavior: readonly string[];
  limitation: string;
  /** CPU reference engine; absent for 'gpu' operations. */
  run?(
    inputs: Readonly<Record<string, CatalogFeatureCollection>>,
    parameters: Readonly<Record<string, CatalogParameterValue>>
  ): CatalogRunOutput;
  /** GPU engine; dispatches real graph contributors on the supplied device. */
  runGpu?(
    context: CatalogGpuContext,
    inputs: Readonly<Record<string, CatalogFeatureCollection>>,
    parameters: Readonly<Record<string, CatalogParameterValue>>
  ): Promise<CatalogRunOutput>;
};

export type CatalogRunResult = {
  operationId: string;
  /** Engine that actually produced the tables. */
  engine: CatalogEngine;
  /** GPU contributor classes dispatched; empty for the CPU engine. */
  dispatchedContributors: readonly string[];
  /** Why the GPU engine did not run (no device, unsupported inputs or a GPU failure). */
  fallbackReason?: string;
  parameters: Record<string, CatalogParameterValue>;
  tables: CatalogOutputTable[];
  diagnostics: CatalogDiagnostic[];
  summary: Record<string, string | number>;
};

export const COORDINATE_SYSTEM_PARAMETER: CatalogParameterSpec = {
  name: 'coordinateSystem',
  label: 'Coordinate system',
  kind: 'enum',
  options: ['EPSG:4326', 'planar'],
  defaultValue: 'EPSG:4326'
};

export const NETWORK_PARAMETERS: readonly CatalogParameterSpec[] = [
  COORDINATE_SYSTEM_PARAMETER,
  {
    name: 'snapTolerance',
    label: 'Vertex snap tolerance',
    kind: 'number',
    unit: 'crs',
    minimum: 0,
    maximum: 1e6,
    defaultValue: 0
  },
  {
    name: 'splitAtSharedVertices',
    label: 'Split at shared vertices',
    kind: 'boolean',
    defaultValue: true
  },
  {
    name: 'costMode',
    label: 'Cost',
    kind: 'enum',
    options: ['length', 'property'],
    defaultValue: 'length'
  },
  {
    name: 'forwardCostProperty',
    label: 'Forward cost property',
    kind: 'string',
    maximumLength: 64,
    defaultValue: ''
  },
  {
    name: 'reverseCostProperty',
    label: 'Reverse cost property',
    kind: 'string',
    maximumLength: 64,
    defaultValue: ''
  },
  {
    name: 'onewayProperty',
    label: 'One-way property',
    kind: 'string',
    maximumLength: 64,
    defaultValue: ''
  },
  {
    name: 'crossings',
    label: 'Crossing lines',
    kind: 'enum',
    options: ['not-noded', 'noded'],
    defaultValue: 'not-noded'
  }
];

export const SNAP_DISTANCE_PARAMETER: CatalogParameterSpec = {
  name: 'maximumSnapDistance',
  label: 'Maximum snap distance',
  kind: 'number',
  unit: 'crs',
  minimum: 0,
  maximum: 1e6,
  defaultValue: 100
};

export const LINE_INPUT: CatalogInputSpec = {
  name: 'lines',
  label: 'Network lines',
  accepts: ['line'],
  maximumFeatures: 20000,
  maximumVertices: 200000
};

export const MAXIMUM_NETWORK_EDGES = 50000;

export function getNetworkParameters(
  values: Readonly<Record<string, CatalogParameterValue>>
): CatalogNetworkParameters {
  return {
    coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem,
    snapTolerance: values['snapTolerance'] as number,
    splitAtSharedVertices: values['splitAtSharedVertices'] as boolean,
    costMode: values['costMode'] as 'length' | 'property',
    forwardCostProperty: values['forwardCostProperty'] as string,
    reverseCostProperty: values['reverseCostProperty'] as string,
    onewayProperty: values['onewayProperty'] as string
  };
}

export const NETWORK_BEHAVIOR = [
  'Non-line features and lines with fewer than two distinct vertices are skipped and counted.',
  'Arcs with negative or non-finite cost are removed and counted; one-way values forward/reverse restrict direction.',
  'Lines that cross without a shared vertex are not connected by the CPU engine; crossings=noded cuts them with GPUNetworkNoding when a GPU device is supplied.',
  'More than 50,000 edges throws before any result is produced.'
];
