// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {AnalysisSensitivityProvenance} from './analysis-sensitivity';
import type {Device} from '@luma.gl/core';
import type {
  CatalogCoordinateSystem,
  CatalogDiagnostic,
  CatalogFeatureCollection,
  CatalogOutputTable
} from './catalog-geojson';
import type {CatalogDataInput} from './catalog-data-input';
import type {CatalogNetworkParameters} from './catalog-network';

/** Engine that produced a result. */
export type CatalogEngine = 'cpu-reference' | 'gpu';

/** Engines a binding supports. */
export type CatalogEngineSupport = 'cpu-reference' | 'gpu-or-cpu' | 'gpu';

/** Ordered analytical stage, including host work between GPU submissions. */
export type CatalogExecutionStage = {
  identity: {package: string; exportName: string};
  execution: 'cpu' | 'gpu';
};

/** Device handed to GPU runners; the caller owns it. */
export type CatalogGpuContext = {device: Device; signal?: AbortSignal};

/** A binding-owned compiled executor. Execute calls are serialized by the preview session. */
export type CatalogPreviewExecutor = {
  /** Fallible allocation/compilation, called once after ownership and admission are retained. */
  initialize?: (signal: AbortSignal) => Promise<void>;
  execute: (
    parameters: Readonly<Record<string, CatalogParameterValue>>,
    signal: AbortSignal
  ) => Promise<CatalogRunOutput>;
  /** Must release every resource after GPU completion, including failed initialization/cancellation. */
  destroy: () => Promise<void>;
};

/** Explicit peak, including compiled scratch, readback and two CPU result copies for replacement. */
export type CatalogPreviewPlan = {
  /** Compile-time settings only; the session additionally keys on the caller's input revision. */
  key: string;
  gpuByteLength: number;
  cpuByteLength: number;
  /**
   * Return the cleanup owner before fallible resource allocation; perform that work in initialize.
   * A rejection here must leave no resources, because no cleanup owner has been returned yet.
   */
  create: (context: CatalogGpuContext & {signal: AbortSignal}) => Promise<CatalogPreviewExecutor>;
};

/** Tables, diagnostics and summary of one run. */
export type CatalogRunOutput = {
  stages?: readonly CatalogExecutionStage[];
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
  coordinateModeUnits?: {geographic: string; planar: string};
  nonzero?: boolean;
  /** The unscaled value and its product with this scale must each remain finite and nonzero as Float32. */
  nonzeroFloat32Scale?: number;
  /** Canonical parameter named by a retained deprecated spelling. */
  aliasFor?: string;
  /** Retired name of this parameter, used in validation messages. */
  renamedFrom?: string;
  minimum?: number;
  maximum?: number;
  options?: readonly string[];
  maximumLength?: number;
  /** Omitted when the binding derives the default from its input data. */
  defaultValue?: CatalogParameterValue;
};

export type CatalogInputSpec = {
  name: string;
  label: string;
  /** Absent means features, preserving existing bindings. Non-feature ports use accepts: []. */
  dataKind?: 'features' | 'table' | 'raster';
  /** Additive table route for a feature port whose calculation only reads numeric columns. */
  acceptsTable?: boolean;
  accepts: readonly ('point' | 'line' | 'polygon')[];
  /** Feature cap; for table routes (including acceptsTable) this is the row cap. */
  maximumFeatures: number;
  /** Required positive bound for dataKind=raster; cells are never counted as features. */
  maximumCells?: number;
  maximumVertices: number;
};

export type CatalogOutputSpec = {
  /** Computational representation, independent of the display table's geometry. */
  dataKind?: 'raster';
  tableId: string;
  label: string;
  geometry: 'point' | 'line' | 'polygon' | 'mixed' | 'none';
  /** Selected input determines geometry, including no geometry for a table route. */
  geometryFromInput?: string;
  /** Columns that tie each output row to the source feature and to its new identity. */
  identityColumns: readonly string[];
  maximumRows: number;
};

export type CatalogOperationBinding = {
  /** Qualified contributor declarations for exports whose names overlap between packages. */
  contributorIdentities?: readonly {package: string; exportName: string}[];
  stages?: readonly CatalogExecutionStage[];
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
  /** Analytical contributor/helper exports, including CPU stages in a mixed execution. */
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
  /** Opt in only when the binding can retain one compilation and declare its entire byte peak. */
  planPreview?(
    inputs: Readonly<Record<string, CatalogDataInput>>,
    parameters: Readonly<Record<string, CatalogParameterValue>>
  ): CatalogPreviewPlan;
  /** Typed engine for bindings with table or raster ports (including mixed feature inputs). */
  runData?(
    inputs: Readonly<Record<string, CatalogDataInput>>,
    parameters: Readonly<Record<string, CatalogParameterValue>>
  ): CatalogRunOutput;
  runDataGpu?(
    context: CatalogGpuContext,
    inputs: Readonly<Record<string, CatalogDataInput>>,
    parameters: Readonly<Record<string, CatalogParameterValue>>
  ): Promise<CatalogRunOutput>;
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
  stages?: readonly CatalogExecutionStage[];
  sensitivity?: AnalysisSensitivityProvenance;
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

/** Geometry-only operations do not read costs or one-way restrictions. */
export const NETWORK_TOPOLOGY_PARAMETERS = NETWORK_PARAMETERS.filter(
  parameter =>
    !['costMode', 'forwardCostProperty', 'reverseCostProperty', 'onewayProperty'].includes(
      parameter.name
    )
);

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
    costMode: (values['costMode'] ?? 'length') as 'length' | 'property',
    forwardCostProperty: (values['forwardCostProperty'] ?? '') as string,
    reverseCostProperty: (values['reverseCostProperty'] ?? '') as string,
    onewayProperty: (values['onewayProperty'] ?? '') as string
  };
}

export const NETWORK_TOPOLOGY_BEHAVIOR = [
  'Non-line features and lines with fewer than two distinct vertices are skipped and counted.',
  'Lines that cross without a shared vertex are not connected by the CPU engine; crossings=noded cuts them with GPUNetworkNoding when a GPU device is supplied.',
  'More than 50,000 edges throws before any result is produced.'
];

export const NETWORK_BEHAVIOR = [
  ...NETWORK_TOPOLOGY_BEHAVIOR,
  'Missing, blank, non-numeric or non-finite property costs remove the affected arcs and are counted in diagnostics; negative costs throw a named validation error. One-way values forward/reverse restrict direction.',
  'Length mode ignores named forward/reverse cost columns with a visible warning; choose costMode=property to use those columns.'
];
