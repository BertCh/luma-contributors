// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import type {Device} from '@luma.gl/core';
import {
  GPU_SPATIAL_ANALYSIS_CAPABILITIES,
  GPU_SPATIAL_ANALYSIS_OPERATORS
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {
  addCatalogDiagnostic,
  countCollectionVertices,
  hashCatalogText,
  validateCatalogCoordinateSystem,
  type CatalogCoordinateSystem,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {
  runConvexClip,
  runGeometryMeasures,
  runGeometryValidity,
  runLineSegmentize,
  runPointBuffer
} from './catalog-geometry';
import {runNetworkBuild, runNetworkServiceArea, runNetworkShortestPaths} from './catalog-network';
import {
  COORDINATE_SYSTEM_PARAMETER,
  LINE_INPUT,
  MAXIMUM_NETWORK_EDGES,
  NETWORK_BEHAVIOR,
  NETWORK_PARAMETERS,
  SNAP_DISTANCE_PARAMETER,
  getNetworkParameters,
  type CatalogEngine,
  type CatalogOperationBinding,
  type CatalogParameterValue,
  type CatalogRunResult
} from './catalog-binding-types';
import {validateGeometryOutputCapacity} from './catalog-extra-geometry';
import {EXTRA_CPU_BINDINGS} from './catalog-extra-bindings';
import {listPointCloudCatalogEntries} from './catalog-pointcloud-entries';
import {POINT_CLOUD_BINDINGS} from './catalog-pointcloud-bindings';
import {PLATFORM3_A_BINDINGS} from './catalog-platform3-a-bindings';
import {PLATFORM3_B_BINDINGS} from './catalog-platform3-b-bindings';
import {PLATFORM4_GEOMETRY_GPU_RUNNERS} from './catalog-platform4-geometry-gpu';
import {RASTER_TIMESERIES_PANEL_OPERATORS} from './catalog-raster-timeseries-bindings';
import {PLATFORM4_NEW_BINDINGS} from './catalog-platform4-new-bindings';
import {SPACETIME_BINDINGS} from './catalog-spacetime-bindings';
import {appendValidationCatalogEntries} from './catalog-validation-bindings';
import {RASTER_EMBEDDING_BINDINGS} from './catalog-raster-embeddings-bindings';
import {PLATFORM5_NEW_BINDINGS} from './catalog-platform5-bindings';
import {NETWORK_CATEGORICAL_BINDINGS} from './catalog-network-categorical-bindings';
import {KEPLER_CURATED_BINDINGS} from '../curated/bindings/index';
import {TRAJECTORY_BINDINGS} from './catalog-trajectory-bindings';
import {PLATFORM5_POINT_STATISTICS_GPU_RUNNERS} from './catalog-platform5-point-statistics-gpu';
import {PLATFORM5_STATISTICS_GPU_RUNNERS} from './catalog-platform5-statistics-gpu';
import {PLATFORM4_STATISTICS_GPU_RUNNERS} from './catalog-platform4-statistics-gpu';
import {SNAP_POINTS_CPU_BINDING} from './catalog-snap-binding';
import {CATALOG_GPU_ONLY_BINDINGS, CATALOG_GPU_RUNNERS} from './catalog-gpu-bindings';
import {listCatalogLensScenarios, type CatalogLensScenario} from './catalog-lens';

export type {
  CatalogEngine,
  CatalogEngineSupport,
  CatalogGpuContext,
  CatalogInputSpec,
  CatalogOperationBinding,
  CatalogOutputSpec,
  CatalogParameterSpec,
  CatalogParameterValue,
  CatalogRunOutput,
  CatalogRunResult
} from './catalog-binding-types';

/** CPU-bound operation before the GPU runner (if any) is attached. */
type CpuBinding = Omit<CatalogOperationBinding, 'gpuContributorNames'>;

const CPU_BINDINGS: readonly CpuBinding[] = [
  {
    id: 'geometry-validity',
    title: 'Polygon validity report',
    domain: 'geometry',
    summary:
      'Reports closure, length, zero area, self-intersection, ring crossings and hole containment per polygon.',
    capabilityIds: ['geometry-validation'],
    contributorNames: ['GPUGeometryValidity'],
    engine: 'cpu-reference',
    inputs: [
      {
        name: 'polygons',
        label: 'Polygons',
        accepts: ['polygon'],
        maximumFeatures: 20000,
        maximumVertices: 200000
      }
    ],
    parameters: [],
    outputs: [
      {
        tableId: 'validity',
        label: 'Validity report',
        geometry: 'polygon',
        identityColumns: ['source_index', 'source_id'],
        maximumRows: 20000
      }
    ],
    cleanup:
      'CPU only; no retained GPU resources. Published results are removed through native Sources.',
    declaredBehavior: [
      'Invalid features are reported unchanged; nothing is repaired or dropped.',
      'Features over 2000 vertices skip pairwise intersection checks and report skipped.',
      'Hole containment tests the first hole vertex; boundary touching is accepted.'
    ],
    limitation:
      'Detection only and planar; touching rings are accepted and no repair is attempted.',
    run: inputs => runGeometryValidity(inputs['polygons'])
  },
  {
    id: 'geometry-measures',
    title: 'Geometry measures',
    domain: 'geometry',
    summary:
      'Area, length or perimeter, bounds and vertex count per feature with an explicit coordinate system.',
    capabilityIds: ['geometry-measurement'],
    contributorNames: ['GPUGeometryMeasures'],
    engine: 'cpu-reference',
    inputs: [
      {
        name: 'features',
        label: 'Features',
        accepts: ['point', 'line', 'polygon'],
        maximumFeatures: 20000,
        maximumVertices: 200000
      }
    ],
    parameters: [COORDINATE_SYSTEM_PARAMETER],
    outputs: [
      {
        tableId: 'measures',
        label: 'Measures',
        geometry: 'polygon',
        identityColumns: ['source_index', 'source_id'],
        maximumRows: 20000
      }
    ],
    cleanup:
      'CPU only; no retained GPU resources. Published results are removed through native Sources.',
    declaredBehavior: [
      'EPSG:4326 reports meters and square meters (haversine, spherical excess); planar reports coordinate units.',
      'Holes subtract from polygon area; empty geometries produce null measures.',
      'Centroid is a planar vertex average.'
    ],
    limitation: 'Spherical, not ellipsoidal; no projection or datum transformation.',
    run: (inputs, parameters) =>
      runGeometryMeasures(
        inputs['features'],
        parameters['coordinateSystem'] as CatalogCoordinateSystem
      )
  },
  {
    id: 'geometry-point-buffer',
    title: 'Point buffer',
    domain: 'geometry',
    summary:
      'Inscribed circle polygons around points with new buffer identities tied to source rows.',
    capabilityIds: ['geometry-construction'],
    contributorNames: ['GPUPointBuffer'],
    engine: 'cpu-reference',
    inputs: [
      {
        name: 'points',
        label: 'Points',
        accepts: ['point'],
        maximumFeatures: 20000,
        maximumVertices: 20000
      }
    ],
    parameters: [
      {
        name: 'radius',
        label: 'Radius',
        kind: 'number',
        unit: 'crs',
        minimum: 1e-9,
        maximum: 1e7,
        defaultValue: 100
      },
      {
        name: 'segmentsPerQuarter',
        label: 'Segments per quarter',
        kind: 'integer',
        minimum: 1,
        maximum: 64,
        defaultValue: 8
      },
      COORDINATE_SYSTEM_PARAMETER
    ],
    outputs: [
      {
        tableId: 'buffers',
        label: 'Buffers',
        geometry: 'polygon',
        identityColumns: ['buffer_id', 'source_index', 'source_id', 'point_index'],
        maximumRows: 20000
      }
    ],
    cleanup:
      'CPU only; no retained GPU resources. Published results are removed through native Sources.',
    declaredBehavior: [
      'Radius is meters for EPSG:4326 and coordinate units for planar.',
      'Buffers reaching a pole are dropped; antimeridian crossings are not split.',
      'Overlapping buffers stay separate features; no dissolve or union.'
    ],
    limitation: 'Points only; line and polygon buffering with union is not bound.',
    run: (inputs, parameters) =>
      runPointBuffer(inputs['points'], {
        radius: parameters['radius'] as number,
        segmentsPerQuarter: parameters['segmentsPerQuarter'] as number,
        coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
      })
  },
  {
    id: 'geometry-line-segmentize',
    title: 'Line refinement (segmentize)',
    domain: 'geometry',
    summary:
      'Densifies lines so no segment exceeds a maximum length; source identity is preserved.',
    capabilityIds: ['line-processing'],
    contributorNames: ['GPULineSegmentize'],
    engine: 'cpu-reference',
    inputs: [
      {
        name: 'lines',
        label: 'Lines',
        accepts: ['line'],
        maximumFeatures: 20000,
        maximumVertices: 200000
      }
    ],
    parameters: [
      {
        name: 'maximumSegmentLength',
        label: 'Maximum segment length',
        kind: 'number',
        unit: 'crs',
        minimum: 1e-6,
        maximum: 1e7,
        defaultValue: 100
      },
      COORDINATE_SYSTEM_PARAMETER
    ],
    outputs: [
      {
        tableId: 'segmentized',
        label: 'Segmentized lines',
        geometry: 'line',
        identityColumns: ['output_id', 'source_index', 'source_id'],
        maximumRows: 20000
      }
    ],
    cleanup:
      'CPU only; no retained GPU resources. Published results are removed through native Sources.',
    declaredBehavior: [
      'Output over 500,000 vertices throws instead of truncating.',
      'New vertices are linear in coordinate space; segment length is measured in the declared system.',
      'Non-line features produce no output and are counted.'
    ],
    limitation:
      'At most 500,000 output vertices, checked before generation. Linear interpolation is not a great-circle path.',
    run: (inputs, parameters) =>
      runLineSegmentize(
        inputs['lines'],
        {
          maximumSegmentLength: parameters['maximumSegmentLength'] as number,
          coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
        },
        500000
      )
  },
  {
    id: 'geometry-convex-clip',
    title: 'Clip by convex polygon (overlay)',
    domain: 'geometry',
    summary:
      'Intersection of polygons or lines with one convex clip polygon; outputs get new part identities.',
    capabilityIds: ['geometry-overlay', 'geometry-editing'],
    contributorNames: ['GPURectangleClip'],
    engine: 'cpu-reference',
    inputs: [
      {
        name: 'features',
        label: 'Features to clip',
        accepts: ['line', 'polygon'],
        maximumFeatures: 20000,
        maximumVertices: 200000
      },
      {
        name: 'clip',
        label: 'Convex clip polygon',
        accepts: ['polygon'],
        maximumFeatures: 1,
        maximumVertices: 1000
      }
    ],
    parameters: [],
    outputs: [
      {
        tableId: 'clipped',
        label: 'Clipped features',
        geometry: 'polygon',
        identityColumns: ['output_id', 'source_index', 'source_id', 'part_index'],
        maximumRows: 40000
      }
    ],
    cleanup:
      'CPU only; no retained GPU resources. Published results are removed through native Sources.',
    declaredBehavior: [
      'A non-convex or invalid clip polygon throws; the clip polygon is oriented counterclockwise.',
      'Features entirely outside the clip produce no row and are counted.',
      'Holes clip independently; a hole outside the clip is dropped; concave shells are flagged.'
    ],
    limitation:
      'Intersection only (no union, difference or general polygon overlay); planar edges.',
    run: (inputs, _parameters) => runConvexClip(inputs['features'], inputs['clip'])
  },
  {
    id: 'network-build-graph',
    title: 'Build network from lines',
    domain: 'network',
    summary:
      'Clusters vertices into nodes, splits at shared vertices and publishes directed edges with source edge IDs.',
    capabilityIds: [],
    contributorNames: ['GPUNetworkNoding'],
    engine: 'cpu-reference',
    inputs: [LINE_INPUT],
    parameters: NETWORK_PARAMETERS,
    outputs: [
      {
        tableId: 'network-edges',
        label: 'Network edges',
        geometry: 'line',
        identityColumns: ['edge_id', 'source_index', 'source_id', 'part_index'],
        maximumRows: MAXIMUM_NETWORK_EDGES
      }
    ],
    cleanup:
      'CPU only; no retained GPU resources. Published results are removed through native Sources.',
    declaredBehavior: NETWORK_BEHAVIOR,
    limitation: 'Vertex-level noding only; property costs are prorated by length over split edges.',
    run: (inputs, parameters) =>
      runNetworkBuild(inputs['lines'], getNetworkParameters(parameters), MAXIMUM_NETWORK_EDGES)
  },
  {
    id: 'network-shortest-path',
    title: 'Shortest paths from one origin',
    domain: 'network',
    summary:
      'Snaps points to the network and returns directed least-cost paths with edge IDs and unreachable status.',
    capabilityIds: [],
    contributorNames: ['GPUNetworkSnapping', 'GPUNetworkPathExtraction'],
    engine: 'cpu-reference',
    inputs: [
      LINE_INPUT,
      {
        name: 'origin',
        label: 'Origin point',
        accepts: ['point'],
        maximumFeatures: 1,
        maximumVertices: 1
      },
      {
        name: 'destinations',
        label: 'Destination points',
        accepts: ['point'],
        maximumFeatures: 500,
        maximumVertices: 500
      }
    ],
    parameters: [...NETWORK_PARAMETERS, SNAP_DISTANCE_PARAMETER],
    outputs: [
      {
        tableId: 'network-paths',
        label: 'Shortest paths',
        geometry: 'line',
        identityColumns: ['path_id', 'destination_index', 'edge_ids', 'source_indices'],
        maximumRows: 500
      }
    ],
    cleanup:
      'CPU only; no retained GPU resources. Published results are removed through native Sources.',
    declaredBehavior: [
      ...NETWORK_BEHAVIOR,
      'Statuses: ok, origin-not-snapped, destination-not-snapped, unreachable-directed, unreachable-disconnected.',
      'Partial edge cost is proportional to length along the snapped edge.'
    ],
    limitation:
      'Single origin; exhaustive O(edges) snapping per point; not a time-dependent or turn-restricted router.',
    run: (inputs, parameters) =>
      runNetworkShortestPaths(
        inputs['lines'],
        inputs['origin'],
        inputs['destinations'],
        {
          ...getNetworkParameters(parameters),
          maximumSnapDistance: parameters['maximumSnapDistance'] as number
        },
        {maximumEdges: MAXIMUM_NETWORK_EDGES, maximumDestinations: 500}
      )
  },
  {
    id: 'network-service-area',
    title: 'Service area and cost bands',
    domain: 'network',
    summary:
      'Edges reached within cost budgets with band, reached fraction and separated unreachable diagnostics.',
    capabilityIds: [],
    contributorNames: ['GPUNetworkServiceAreas'],
    engine: 'cpu-reference',
    inputs: [
      LINE_INPUT,
      {
        name: 'origin',
        label: 'Origin point',
        accepts: ['point'],
        maximumFeatures: 1,
        maximumVertices: 1
      }
    ],
    parameters: [
      ...NETWORK_PARAMETERS,
      SNAP_DISTANCE_PARAMETER,
      {
        name: 'budgets',
        label: 'Cost budgets',
        kind: 'number-list',
        unit: 'cost',
        minimum: 0,
        maximum: 1e9,
        maximumLength: 8,
        defaultValue: [500, 1000]
      },
      {
        name: 'includeUnreachable',
        label: 'Keep unreachable edges',
        kind: 'boolean',
        defaultValue: true
      }
    ],
    outputs: [
      {
        tableId: 'network-service-area',
        label: 'Service area edges',
        geometry: 'line',
        identityColumns: ['edge_id', 'source_index', 'source_id'],
        maximumRows: MAXIMUM_NETWORK_EDGES
      }
    ],
    cleanup:
      'CPU only; no retained GPU resources. Published results are removed through native Sources.',
    declaredBehavior: [
      ...NETWORK_BEHAVIOR,
      'Unreachable statuses: unreachable-beyond-budget, unreachable-directed, unreachable-disconnected.',
      'An unsnappable origin returns an error diagnostic and no reached edges.'
    ],
    limitation: 'Edge bands only; no isochrone polygon boundaries; at most 8 budgets.',
    run: (inputs, parameters) =>
      runNetworkServiceArea(
        inputs['lines'],
        inputs['origin'],
        {
          ...getNetworkParameters(parameters),
          maximumSnapDistance: parameters['maximumSnapDistance'] as number,
          budgets: parameters['budgets'] as readonly number[],
          includeUnreachable: parameters['includeUnreachable'] as boolean
        },
        {maximumEdges: MAXIMUM_NETWORK_EDGES}
      )
  }
];

const GPU_OR_CPU_CLEANUP =
  'CPU engine retains nothing. The GPU engine creates storage buffers and compiled graphs per run and destroys them before the result returns; results are plain arrays and publication goes through native Sources.';

const PLATFORM4_GPU_RUNNERS = {
  ...PLATFORM4_GEOMETRY_GPU_RUNNERS,
  ...PLATFORM4_STATISTICS_GPU_RUNNERS,
  ...PLATFORM5_STATISTICS_GPU_RUNNERS,
  ...PLATFORM5_POINT_STATISTICS_GPU_RUNNERS
};

/** Upgrades a CPU-reference binding to dispatch its GPU contributors when PLATFORM4 registered a runner. */
function attachPlatform4Runner(binding: CatalogOperationBinding): CatalogOperationBinding {
  const runner = PLATFORM4_GPU_RUNNERS[binding.id];
  if (!runner || binding.runGpu) return binding;
  return {
    ...binding,
    engine: 'gpu-or-cpu',
    cleanup: GPU_OR_CPU_CLEANUP,
    gpuContributorNames: runner.contributorNames,
    runGpu: runner.run
  };
}

const BINDINGS: readonly CatalogOperationBinding[] = [
  ...[...CPU_BINDINGS, ...EXTRA_CPU_BINDINGS, SNAP_POINTS_CPU_BINDING].map(
    (binding): CatalogOperationBinding => {
      const runner = CATALOG_GPU_RUNNERS[binding.id];
      return runner
        ? {
            ...binding,
            engine: 'gpu-or-cpu',
            cleanup: GPU_OR_CPU_CLEANUP,
            gpuContributorNames: runner.contributorNames,
            runGpu: runner.run
          }
        : {...binding, gpuContributorNames: []};
    }
  ),
  ...POINT_CLOUD_BINDINGS,
  ...CATALOG_GPU_ONLY_BINDINGS,
  ...NETWORK_CATEGORICAL_BINDINGS,
  ...PLATFORM3_A_BINDINGS,
  ...PLATFORM3_B_BINDINGS
]
  .map(attachPlatform4Runner)
  .concat([
    ...PLATFORM4_NEW_BINDINGS,
    ...PLATFORM5_NEW_BINDINGS,
    ...SPACETIME_BINDINGS,
    ...TRAJECTORY_BINDINGS,
    ...RASTER_EMBEDDING_BINDINGS,
    ...KEPLER_CURATED_BINDINGS
  ]);

/** Registered, executable operations. Anything not here is shown as not runnable. */
export const KEPLER_CATALOG_BINDINGS: readonly CatalogOperationBinding[] = BINDINGS;

export function getCatalogBinding(operationId: string): CatalogOperationBinding | undefined {
  return BINDINGS.find(binding => binding.id === operationId);
}

export function getCatalogDefaultParameters(
  binding: Pick<CatalogOperationBinding, 'parameters'>
): Record<string, CatalogParameterValue> {
  return Object.fromEntries(
    binding.parameters.flatMap(spec =>
      spec.defaultValue === undefined ? [] : [[spec.name, spec.defaultValue]]
    )
  );
}

/** Coerces and range-checks raw values; obsolete topology parameters in saved workflows are ignored. */
export function normalizeCatalogParameters(
  binding: Pick<CatalogOperationBinding, 'parameters'> &
    Partial<Pick<CatalogOperationBinding, 'id'>>,
  raw: Readonly<Record<string, unknown>>
): Record<string, CatalogParameterValue> {
  for (const name of Object.keys(raw)) {
    const obsoleteTopologyParameter =
      ['network-structure', 'network-line-graph', 'network-snap-points'].includes(
        binding.id ?? ''
      ) &&
      ['costMode', 'forwardCostProperty', 'reverseCostProperty', 'onewayProperty'].includes(name);
    if (!binding.parameters.some(spec => spec.name === name) && !obsoleteTopologyParameter)
      throw new Error(
        `${binding.id ?? 'catalog'}: ${name} is an unknown parameter; use a declared parameter.`
      );
  }
  const result: Record<string, CatalogParameterValue> = {};
  for (const spec of binding.parameters) {
    const value = raw[spec.name] === undefined ? spec.defaultValue : raw[spec.name];
    if (value === undefined) continue;
    // Parameter value outside its declared type, range or options.
    const reject = (): never => {
      throw new Error('Parameter out of range');
    };
    if (spec.kind === 'boolean') {
      if (typeof value !== 'boolean') reject();
      result[spec.name] = value as boolean;
    } else if (spec.kind === 'enum') {
      if (typeof value !== 'string' || !spec.options!.includes(value)) reject();
      result[spec.name] = value as string;
    } else if (spec.kind === 'string') {
      if (typeof value !== 'string' || value.length > (spec.maximumLength ?? 64)) reject();
      result[spec.name] = value as string;
    } else if (spec.kind === 'number-list') {
      const list =
        typeof value === 'string'
          ? value
              .split(',')
              .filter(item => item.trim())
              .map(Number)
          : value;
      if (
        !Array.isArray(list) ||
        !list.length ||
        list.length > (spec.maximumLength ?? 8) ||
        list.some(
          item =>
            typeof item !== 'number' ||
            !Number.isFinite(item) ||
            item < (spec.minimum ?? -Infinity) ||
            item > (spec.maximum ?? Infinity)
        )
      )
        reject();
      result[spec.name] = list as number[];
    } else {
      const number = typeof value === 'string' && value.trim() ? Number(value) : value;
      if (
        typeof number !== 'number' ||
        !Number.isFinite(number) ||
        number < (spec.minimum ?? -Infinity) ||
        number > (spec.maximum ?? Infinity) ||
        (spec.kind === 'integer' && !Number.isInteger(number))
      )
        reject();
      result[spec.name] = number as number;
    }
  }
  return result;
}

function describeGeometryKind(type: string | undefined): 'point' | 'line' | 'polygon' | null {
  if (type === 'Point' || type === 'MultiPoint') return 'point';
  if (type === 'LineString' || type === 'MultiLineString') return 'line';
  if (type === 'Polygon' || type === 'MultiPolygon') return 'polygon';
  return null;
}

/** Enforces declared geometry kinds and capacities before any work starts. */
export function validateCatalogInputs(
  binding: CatalogOperationBinding,
  inputs: Readonly<Record<string, CatalogFeatureCollection | undefined>>
): void {
  for (const spec of binding.inputs) {
    const collection = inputs[spec.name];
    // Required input missing, empty, over capacity or of an unaccepted geometry kind.
    if (!collection || !collection.features.length) throw new Error('Input required');
    if (collection.features.length > spec.maximumFeatures)
      throw new KeplerAllocationError('capacity', {message: 'Input feature capacity exceeded'});
    if (countCollectionVertices(collection) > spec.maximumVertices)
      throw new KeplerAllocationError('capacity', {message: 'Input vertex capacity exceeded'});
    for (const feature of collection.features) {
      if (!feature.geometry) continue;
      const kind = describeGeometryKind(feature.geometry.type);
      if (!kind || !spec.accepts.includes(kind))
        throw new Error('Input geometry kind not accepted');
    }
  }
}

function runCpuEngine(
  binding: CatalogOperationBinding,
  inputs: Readonly<Record<string, CatalogFeatureCollection | undefined>>,
  parameters: Record<string, CatalogParameterValue>
) {
  // Operations without a CPU reference engine need a GPU device.
  if (!binding.run) throw new Error('GPU device required');
  const result = binding.run(inputs as Record<string, CatalogFeatureCollection>, parameters);
  if (parameters['crossings'] === 'noded')
    addCatalogDiagnostic(
      result.diagnostics,
      'crossings-not-noded-cpu',
      'warning',
      'crossings=noded needs the GPU engine; the CPU engine did not node crossing lines.'
    );
  return result;
}

function checkOutputCapacity(
  binding: CatalogOperationBinding,
  tables: readonly {id: string; rowCount: number}[]
): void {
  for (const table of tables) {
    const spec = binding.outputs.find(output => output.tableId === table.id);
    // Output exceeded its declared row capacity.
    if (!spec || table.rowCount > spec.maximumRows)
      throw new KeplerAllocationError('capacity', {message: 'Output capacity exceeded'});
  }
}

/** Validates, runs on the CPU reference engine and enforces output row capacity. */
export function runCatalogOperation(
  binding: CatalogOperationBinding,
  rawParameters: Readonly<Record<string, unknown>>,
  inputs: Readonly<Record<string, CatalogFeatureCollection | undefined>>
): CatalogRunResult {
  const parameters = normalizeCatalogParameters(binding, rawParameters);
  validateCatalogInputs(binding, inputs);
  validateCatalogCoordinateSystem(
    binding.id,
    parameters['coordinateSystem'],
    inputs as Record<string, CatalogFeatureCollection>
  );
  validateGeometryOutputCapacity(
    binding.id,
    inputs as Record<string, CatalogFeatureCollection>,
    parameters
  );
  const result = runCpuEngine(binding, inputs, parameters);
  checkOutputCapacity(binding, result.tables);
  return {
    operationId: binding.id,
    engine: 'cpu-reference',
    dispatchedContributors: [],
    parameters,
    ...result
  };
}

export type CatalogRunOptions = {
  /** GPU device. Without one, operations with a CPU engine run there and GPU-only operations throw. */
  device?: Device;
  /** `cpu` forces the reference engine (the test oracle); `auto` prefers the GPU. */
  engine?: 'auto' | 'cpu';
};

/**
 * Validates, then runs on the GPU contributors when a device is supplied and the operation has a
 * GPU path, otherwise on the CPU reference engine. A GPU failure on an operation that also has a
 * CPU engine falls back to it with a visible `gpu-fallback` warning and `fallbackReason`; GPU-only
 * operations rethrow. The returned `engine` names what actually produced the tables.
 */
export async function runCatalogOperationAsync(
  binding: CatalogOperationBinding,
  rawParameters: Readonly<Record<string, unknown>>,
  inputs: Readonly<Record<string, CatalogFeatureCollection | undefined>>,
  options: CatalogRunOptions = {}
): Promise<CatalogRunResult> {
  const parameters = normalizeCatalogParameters(binding, rawParameters);
  validateCatalogInputs(binding, inputs);
  validateCatalogCoordinateSystem(
    binding.id,
    parameters['coordinateSystem'],
    inputs as Record<string, CatalogFeatureCollection>
  );
  validateGeometryOutputCapacity(
    binding.id,
    inputs as Record<string, CatalogFeatureCollection>,
    parameters
  );
  let output: ReturnType<typeof runCpuEngine> | undefined;
  let engine: CatalogEngine = 'cpu-reference';
  let dispatchedContributors: readonly string[] = [];
  let fallbackReason: string | undefined;
  if (options.engine === 'cpu') fallbackReason = 'CPU engine requested';
  else if (!binding.runGpu) fallbackReason = 'This operation has no GPU path';
  else if (!options.device) fallbackReason = 'No GPU device supplied';
  else {
    try {
      const gpu = await binding.runGpu(
        {device: options.device},
        inputs as Record<string, CatalogFeatureCollection>,
        parameters
      );
      output = gpu;
      engine = 'gpu';
      dispatchedContributors = gpu.dispatchedContributors ?? binding.gpuContributorNames;
    } catch (failure) {
      fallbackReason = failure instanceof Error ? failure.message : String(failure);
      // Operations without a CPU engine have nothing to fall back to.
      if (!binding.run) throw failure;
    }
  }
  if (!output) {
    output = runCpuEngine(binding, inputs, parameters);
    if (fallbackReason && options.device && options.engine !== 'cpu' && binding.runGpu)
      addCatalogDiagnostic(
        output.diagnostics,
        'gpu-fallback',
        'warning',
        `The GPU engine did not run: ${fallbackReason}. Results come from the CPU reference engine.`
      );
  }
  checkOutputCapacity(binding, output.tables);
  return {
    operationId: binding.id,
    engine,
    dispatchedContributors,
    ...(fallbackReason && engine === 'cpu-reference' ? {fallbackReason} : {}),
    parameters,
    tables: output.tables,
    diagnostics: output.diagnostics,
    summary: output.summary
  };
}

export type CatalogInvocation = {
  schemaVersion: 1;
  operationId: string;
  parameters: Record<string, CatalogParameterValue>;
  /** Inputs are referenced by content hash, never embedded. */
  inputs: Record<string, {featureCount: number; vertexCount: number; contentHash: string}>;
};

export function hashCatalogCollection(collection: CatalogFeatureCollection): string {
  return hashCatalogText(JSON.stringify(collection));
}

export function serializeCatalogInvocation(
  binding: CatalogOperationBinding,
  rawParameters: Readonly<Record<string, unknown>>,
  inputs: Readonly<Record<string, CatalogFeatureCollection>>
): string {
  const invocation: CatalogInvocation = {
    schemaVersion: 1,
    operationId: binding.id,
    parameters: normalizeCatalogParameters(binding, rawParameters),
    inputs: Object.fromEntries(
      binding.inputs.map(spec => [
        spec.name,
        {
          featureCount: inputs[spec.name].features.length,
          vertexCount: countCollectionVertices(inputs[spec.name]),
          contentHash: hashCatalogCollection(inputs[spec.name])
        }
      ])
    )
  };
  return JSON.stringify(invocation);
}

export function parseCatalogInvocation(text: string): {
  binding: CatalogOperationBinding;
  invocation: CatalogInvocation;
} {
  const invocation = JSON.parse(text) as CatalogInvocation;
  const binding = getCatalogBinding(invocation.operationId);
  // Unknown schema version or operation; the invocation cannot be restored.
  if (invocation.schemaVersion !== 1 || !binding) throw new Error('Unsupported invocation');
  return {
    binding,
    invocation: {
      ...invocation,
      parameters: normalizeCatalogParameters(binding, invocation.parameters)
    }
  };
}

/** Whether supplied inputs are the exact data a serialized invocation was created from. */
export function matchCatalogInvocationInputs(
  invocation: CatalogInvocation,
  inputs: Readonly<Record<string, CatalogFeatureCollection>>
): boolean {
  return Object.entries(invocation.inputs).every(
    ([name, reference]) =>
      inputs[name] && hashCatalogCollection(inputs[name]) === reference.contentHash
  );
}

/** Network contributors are exported by the graph package but are outside the ontology catalog. */
export const KEPLER_NETWORK_CONTRIBUTOR_NAMES = [
  'GPUNetworkNoding',
  'GPUNetworkSnapping',
  'GPUNetworkReachability',
  'GPUNetworkCostMatrix',
  'GPUNetworkCostTable',
  'GPUNetworkCostWeights',
  'GPUNetworkAccessibility',
  'GPUNetworkServiceAreas',
  'GPUNetworkIsochrones',
  'GPUNetworkPathExtraction',
  'GPUNetworkNeighborhood',
  'GPUNetworkSubgraphFilter',
  'GPUNetworkLineGraph',
  'GPUNetworkStructure',
  'GPUNetworkCentrality',
  'GPUNetworkCoarsening',
  'GPUAlternativeRoutes',
  'GPUMapMatching',
  'GPUFlowAggregation',
  'GPUEdgeBundling',
  'GPUNetworkKFunction',
  'GPUNetworkStatistics'
] as const;

/** Host panels that already run an operator natively (not through these catalog bindings). */
const NATIVE_PANEL_OPERATORS: Readonly<Record<string, string>> = {
  ...RASTER_TIMESERIES_PANEL_OPERATORS,
  GPUVariableStandardization: 'Analysis panel · Standardize',
  GPUClassBreaks: 'Analysis panel · Classify',
  GPUDorlingCartogram: 'Analysis panel · Cartogram'
};

export type CatalogEntry = {
  contributorName: string;
  capabilityId: string;
  capabilityTitle: string;
  domain: string;
  runnable: 'catalog-binding' | 'native-panel' | 'lens-scenario' | 'no';
  /** Engine behind a runnable entry: what the binding dispatches, or a CPU engine that mirrors the contributor. */
  engine?: 'gpu' | 'gpu-or-cpu' | 'cpu-reference' | 'lens-scenario';
  bindingIds: readonly string[];
  /** Registered lens scenarios that run this contributor on an embedded dataset. */
  lensScenarioIds?: readonly string[];
  location?: string;
  /** Present when not runnable; says exactly what is missing. */
  reason?: string;
};

const NOT_BOUND_REASON =
  'No executable host binding: declared inputs, units, capacities, output identity, cleanup and serialization are not defined for this contributor. Computation-pack operator and formula entries describe it but do not execute it.';

const LENS_NOT_LOADED_REASON =
  ' The lens scenario registry is empty here; loading it exposes embedded-dataset scenarios for contributors that have one.';

export type CatalogEntryOptions = {
  /** Registered lens scenarios; defaults to whatever the registry holds now. */
  lensScenarios?: readonly CatalogLensScenario[];
};

/** One entry per ontology operator and network contributor, with runnability and the reason when not. */
export function listCatalogEntries(options: CatalogEntryOptions = {}): CatalogEntry[] {
  const entries: CatalogEntry[] = [];
  const lensScenarios = options.lensScenarios ?? listCatalogLensScenarios();
  const bindingsFor = (name: string) =>
    BINDINGS.filter(
      binding =>
        binding.contributorNames.includes(name) || binding.gpuContributorNames.includes(name)
    );
  const describe = (
    contributorName: string,
    capabilityId: string,
    capabilityTitle: string,
    domain: string
  ): CatalogEntry => {
    const bindings = bindingsFor(contributorName);
    const lensScenarioIds = lensScenarios
      .filter(scenario => scenario.element === contributorName)
      .map(scenario => scenario.id);
    const base = {
      contributorName,
      capabilityId,
      capabilityTitle,
      domain,
      bindingIds: bindings.map(binding => binding.id),
      ...(lensScenarioIds.length ? {lensScenarioIds} : {})
    };
    if (bindings.length) {
      const dispatching = bindings.filter(binding =>
        binding.gpuContributorNames.includes(contributorName)
      );
      return {
        ...base,
        runnable: 'catalog-binding',
        engine: dispatching.length
          ? dispatching.some(binding => binding.engine === 'gpu')
            ? 'gpu'
            : 'gpu-or-cpu'
          : 'cpu-reference'
      };
    }
    const location = NATIVE_PANEL_OPERATORS[contributorName];
    if (location) return {...base, runnable: 'native-panel', location};
    if (lensScenarioIds.length)
      return {...base, runnable: 'lens-scenario', engine: 'lens-scenario'};
    return {
      ...base,
      runnable: 'no',
      reason: lensScenarios.length ? NOT_BOUND_REASON : NOT_BOUND_REASON + LENS_NOT_LOADED_REASON
    };
  };
  const seen = new Set<string>();
  for (const operator of GPU_SPATIAL_ANALYSIS_OPERATORS) {
    const capability = GPU_SPATIAL_ANALYSIS_CAPABILITIES.find(
      candidate => candidate.id === operator.primaryCapability
    );
    seen.add(operator.exportName);
    entries.push(
      describe(
        operator.exportName,
        operator.primaryCapability,
        capability?.title ?? operator.primaryCapability,
        capability?.domain ?? 'workflow'
      )
    );
  }
  // Register domain ontology records before the fallback for newer bound contributors.
  for (const entry of listPointCloudCatalogEntries(BINDINGS))
    if (!seen.has(entry.contributorName)) {
      seen.add(entry.contributorName);
      entries.push(
        describe(entry.contributorName, entry.capabilityId, entry.capabilityTitle, entry.domain)
      );
    }
  // Newer bound contributors may be outside the original spatial ontology table.
  for (const binding of BINDINGS)
    for (const name of binding.contributorNames)
      if (!seen.has(name)) {
        seen.add(name);
        entries.push(
          describe(name, binding.capabilityIds[0] ?? binding.domain, binding.title, binding.domain)
        );
      }
  for (const name of KEPLER_NETWORK_CONTRIBUTOR_NAMES)
    if (!seen.has(name)) entries.push(describe(name, 'gpu-network', 'GPU network', 'network'));
  appendValidationCatalogEntries(entries, seen, describe);
  return entries;
}
