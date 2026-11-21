// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {listQualifiedCatalogEntries} from './catalog-qualified-entries';
import {KeplerAllocationError} from '../runtime/dataset-registry';
import {
  getCatalogNumberProblem,
  getCatalogParameterRenameMessage,
  getCatalogValidationIdentity,
  parseCatalogParameterNumber,
  resolveCatalogParameterAliases
} from './catalog-parameter-constraints';
export {resolveCatalogParameterUnit} from './catalog-parameter-constraints';
import type {Device} from '@luma.gl/core';
import {planGpuPointBufferPreview} from './catalog-gpu-point-buffer';
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
import {POINT_CLOUD_BINDINGS} from './catalog-pointcloud-bindings';
import {
  POINT_CLOUD_ICP_ID,
  POINT_CLOUD_ICP_MAXIMUM_ITERATIONS_PARAMETER,
  REMOVED_POINT_CLOUD_ICP_ITERATIONS_MESSAGE
} from './catalog-pointcloud-icp-parameters';
import {PLATFORM3_A_BINDINGS} from './catalog-platform3-a-bindings';
import {PLATFORM3_B_BINDINGS} from './catalog-platform3-b-bindings';
import {PLATFORM4_GEOMETRY_GPU_RUNNERS} from './catalog-platform4-geometry-gpu';
import {RASTER_TIMESERIES_PANEL_OPERATORS} from './catalog-raster-timeseries-bindings';
import {PLATFORM4_NEW_BINDINGS} from './catalog-platform4-new-bindings';
import {SPACETIME_BINDINGS} from './catalog-spacetime-bindings';
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
import {
  getCatalogFeatureInputs,
  isCatalogFeatureInput,
  validateCatalogTableInput,
  type CatalogDataInput
} from './catalog-data-input';
import {validateCatalogRasterInput} from './catalog-raster-input';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {listCatalogLensScenarios, type CatalogLensScenario} from './catalog-lens';

export type {
  CatalogEngine,
  CatalogEngineSupport,
  CatalogExecutionStage,
  CatalogGpuContext,
  CatalogInputSpec,
  CatalogOperationBinding,
  CatalogOutputSpec,
  CatalogParameterSpec,
  CatalogParameterValue,
  CatalogPreviewExecutor,
  CatalogPreviewPlan,
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
      'Intersection of subject polygons or lines with one simple convex clip shell with no holes; outputs get new part identities.',
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
        label: 'Clip operand: one simple convex shell, no holes',
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
        geometryFromInput: 'features',
        identityColumns: ['output_id', 'source_index', 'source_id', 'part_index'],
        maximumRows: 40000
      }
    ],
    cleanup:
      'CPU only; no retained GPU resources. Published results are removed through native Sources.',
    declaredBehavior: [
      'The clip operand must contain exactly one polygon and exactly one ring: a simple convex shell with no holes. Multipart, holed, non-convex or invalid clip operands throw; the accepted shell is oriented counterclockwise.',
      'Features entirely outside the clip produce no row and are counted.',
      'Holes in the subject geometry clip independently; a subject hole outside the clip is dropped; concave subject shells are flagged.'
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
            ...(binding.id === 'geometry-point-buffer'
              ? {planPreview: planGpuPointBufferPreview}
              : {}),
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
  const defaults: Record<string, CatalogParameterValue> = Object.fromEntries(
    binding.parameters.flatMap(spec => {
      const value = spec.defaultValue;
      return value === undefined ? [] : [[spec.name, Array.isArray(value) ? [...value] : value]];
    })
  );
  return resolveCatalogParameterAliases(binding.parameters, defaults, (name, problem) => {
    validateRequirement(getCatalogValidationIdentity('catalog'), false, `${name} ${problem}`);
    throw new Error('unreachable');
  });
}

/** Coerces and range-checks raw values; obsolete topology parameters in saved workflows are ignored. */
export function normalizeCatalogParameters(
  binding: Pick<CatalogOperationBinding, 'parameters'> &
    Partial<Pick<CatalogOperationBinding, 'id'>>,
  raw: Readonly<Record<string, unknown>>
): Record<string, CatalogParameterValue> {
  if (
    binding.id === 'trajectory-modal-split' ||
    (binding.id === undefined &&
      binding.parameters.some(parameter => parameter.name === 'modeSpeedUpperBoundsFloat32'))
  ) {
    validateRequirement(
      getCatalogValidationIdentity('trajectory-modal-split'),
      !('modeSpeedBreaks' in raw),
      'modeSpeedBreaks was removed; use modeSpeedUpperBoundsFloat32'
    );
  }
  if (
    binding.id === POINT_CLOUD_ICP_ID ||
    (binding.id === undefined &&
      binding.parameters.includes(POINT_CLOUD_ICP_MAXIMUM_ITERATIONS_PARAMETER))
  ) {
    validateRequirement(
      getCatalogValidationIdentity(POINT_CLOUD_ICP_ID),
      !('iterations' in raw),
      REMOVED_POINT_CLOUD_ICP_ITERATIONS_MESSAGE
    );
  }
  const explicit = resolveCatalogParameterAliases(binding.parameters, raw, (name, problem) => {
    const spec = binding.parameters.find(candidate => candidate.name === name);
    validateRequirement(
      getCatalogValidationIdentity(binding.id ?? 'catalog'),
      false,
      `${name} ${problem}`,
      {
        code: spec?.kind === 'integer' ? 'INTEGER_RANGE' : 'NUMBER_RANGE',
        path: name,
        received: raw[name],
        expected: problem
      }
    );
    throw new Error('unreachable');
  });
  for (const name of Object.keys(raw)) {
    const obsoleteTopologyParameter =
      ['network-structure', 'network-line-graph', 'network-snap-points'].includes(
        binding.id ?? ''
      ) &&
      ['costMode', 'forwardCostProperty', 'reverseCostProperty', 'onewayProperty'].includes(name);
    if (!binding.parameters.some(spec => spec.name === name) && !obsoleteTopologyParameter) {
      const problem =
        getCatalogParameterRenameMessage(binding.parameters, name) ??
        `${name} is an unknown parameter; use a declared parameter.`;
      throw new Error(`${binding.id ?? 'catalog'}: ${problem}`);
    }
  }
  const result: Record<string, CatalogParameterValue> = {};
  const defaults = getCatalogDefaultParameters(binding);
  for (const spec of binding.parameters) {
    const value = explicit[spec.name] === undefined ? defaults[spec.name] : explicit[spec.name];
    if (value === undefined) continue;
    // Parameter value outside its declared type, range or options.
    const reject = (problem: string): never => {
      validateRequirement(
        getCatalogValidationIdentity(binding.id ?? 'catalog'),
        false,
        `${spec.name} ${problem}`,
        {
          code:
            spec.kind === 'enum'
              ? 'INVALID_OPTION'
              : spec.kind === 'integer'
                ? 'INTEGER_RANGE'
                : spec.kind === 'number'
                  ? 'NUMBER_RANGE'
                  : 'REQUIREMENT',
          path: spec.name,
          received: value,
          expected: problem
        }
      );
      throw new Error('unreachable');
    };
    if (spec.kind === 'boolean') {
      if (typeof value !== 'boolean') reject(`must be a boolean; received ${String(value)}`);
      result[spec.name] = value as boolean;
    } else if (spec.kind === 'enum') {
      if (typeof value !== 'string' || !spec.options!.includes(value))
        reject(`must be one of ${spec.options!.join(', ')}; received ${String(value)}`);
      result[spec.name] = value as string;
    } else if (spec.kind === 'string') {
      if (typeof value !== 'string' || value.length > (spec.maximumLength ?? 64))
        reject(
          `must be a string of at most ${spec.maximumLength ?? 64} characters; received ${String(value)}`
        );
      result[spec.name] = value as string;
    } else if (spec.kind === 'number-list') {
      const list =
        typeof value === 'string' ? value.split(',').map(parseCatalogParameterNumber) : value;
      if (
        !Array.isArray(list) ||
        !list.length ||
        list.length > (spec.maximumLength ?? 8) ||
        (typeof value === 'string' && value.split(',').some(item => !item.trim()))
      )
        reject(`must have 1 to ${spec.maximumLength ?? 8} numbers; received ${String(value)}`);
      for (const item of list as unknown[]) {
        if (typeof item !== 'number')
          reject(`must contain finite numbers; received ${String(item)}`);
        const problem = getCatalogNumberProblem(spec, item as number);
        if (problem) reject(problem);
      }
      result[spec.name] = [...(list as number[])];
    } else {
      const number = parseCatalogParameterNumber(value);
      if (typeof number !== 'number') reject(`must be a finite number; received ${String(value)}`);
      const problem = getCatalogNumberProblem(spec, number as number);
      if (problem) reject(problem);
      if (spec.kind === 'integer' && !Number.isInteger(number))
        reject(`must be a whole number; received ${String(number)}`);
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

/** Enforces feature, table and raster shapes and declared capacities before dispatch. */
export function validateCatalogInputs(
  binding: CatalogOperationBinding,
  inputs: Readonly<Record<string, CatalogDataInput | undefined>>
): void {
  const hasDataPorts = binding.inputs.some(
    spec => spec.dataKind === 'table' || spec.dataKind === 'raster' || spec.acceptsTable
  );
  const hasExclusiveDataPorts = binding.inputs.some(
    spec => spec.dataKind === 'table' || spec.dataKind === 'raster'
  );
  validateRequirement(
    binding.id,
    !hasDataPorts || Boolean(binding.runData || binding.runDataGpu),
    'table and raster inputs require a typed runData or runDataGpu engine'
  );
  validateRequirement(
    binding.id,
    !hasExclusiveDataPorts || (!binding.run && !binding.runGpu),
    'table and raster inputs must use typed engines exclusively'
  );
  for (const spec of binding.inputs) {
    const input = inputs[spec.name];
    if (!input) throw new Error('Input required');
    const inputId = `${binding.id}: ${spec.name}`;
    if (spec.dataKind === 'raster') {
      validateRequirement(inputId, input.type === 'raster', 'input must be a raster band/grid');
      validateCatalogRasterInput(inputId, input, spec.maximumCells ?? 0);
      continue;
    }
    if (spec.dataKind === 'table' || (spec.acceptsTable && input.type === 'table')) {
      validateRequirement(inputId, input.type === 'table', 'input must be a geometry-free table');
      validateCatalogTableInput(inputId, input, spec.maximumFeatures);
      if (!input.table.rowCount) throw new Error('Input required');
      continue;
    }
    validateRequirement(inputId, isCatalogFeatureInput(input), 'input must be a FeatureCollection');
    const collection = input;
    // Required input missing, empty, over capacity or of an unaccepted geometry kind.
    if (!collection.features.length) throw new Error('Input required');
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
  inputs: Readonly<Record<string, CatalogDataInput | undefined>>,
  parameters: Record<string, CatalogParameterValue>
) {
  // Operations without a CPU reference engine need a GPU device.
  if (!binding.runData && !binding.run) throw new Error('GPU device required');
  const result = binding.runData
    ? binding.runData(inputs as Record<string, CatalogDataInput>, parameters)
    : binding.run!(getCatalogFeatureInputs(inputs), parameters);
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
  inputs: Readonly<Record<string, CatalogDataInput | undefined>>
): CatalogRunResult {
  const parameters = normalizeCatalogParameters(binding, rawParameters);
  validateCatalogInputs(binding, inputs);
  validateCatalogCoordinateSystem(
    binding.id,
    parameters['coordinateSystem'],
    getCatalogFeatureInputs(inputs)
  );
  validateGeometryOutputCapacity(binding.id, getCatalogFeatureInputs(inputs), parameters);
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
  /** Stops scheduling and rejects obsolete results; submitted work still completes before cleanup. */
  signal?: AbortSignal;
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
  inputs: Readonly<Record<string, CatalogDataInput | undefined>>,
  options: CatalogRunOptions = {}
): Promise<CatalogRunResult> {
  options.signal?.throwIfAborted();
  const parameters = normalizeCatalogParameters(binding, rawParameters);
  validateCatalogInputs(binding, inputs);
  validateCatalogCoordinateSystem(
    binding.id,
    parameters['coordinateSystem'],
    getCatalogFeatureInputs(inputs)
  );
  validateGeometryOutputCapacity(binding.id, getCatalogFeatureInputs(inputs), parameters);
  let output: ReturnType<typeof runCpuEngine> | undefined;
  let engine: CatalogEngine = 'cpu-reference';
  let dispatchedContributors: readonly string[] = [];
  let fallbackReason: string | undefined;
  if (options.engine === 'cpu') fallbackReason = 'CPU engine requested';
  else if (!binding.runDataGpu && !binding.runGpu)
    fallbackReason = 'This operation has no GPU path';
  else if (!options.device) fallbackReason = 'No GPU device supplied';
  else {
    try {
      const gpu = binding.runDataGpu
        ? await binding.runDataGpu(
            {device: options.device, signal: options.signal},
            inputs as Record<string, CatalogDataInput>,
            parameters
          )
        : await binding.runGpu!(
            {device: options.device, signal: options.signal},
            getCatalogFeatureInputs(inputs),
            parameters
          );
      output = gpu;
      engine = 'engine' in gpu && gpu.engine === 'cpu-reference' ? 'cpu-reference' : 'gpu';
      dispatchedContributors = gpu.dispatchedContributors ?? binding.gpuContributorNames;
    } catch (failure) {
      options.signal?.throwIfAborted();
      if (failure instanceof KeplerAllocationError) throw failure;
      fallbackReason = failure instanceof Error ? failure.message : String(failure);
      // Operations without a CPU engine have nothing to fall back to.
      if (!binding.runData && !binding.run) throw failure;
    }
  }
  if (!output) {
    options.signal?.throwIfAborted();
    output = runCpuEngine(binding, inputs, parameters);
    if (
      fallbackReason &&
      options.device &&
      options.engine !== 'cpu' &&
      (binding.runDataGpu || binding.runGpu)
    )
      addCatalogDiagnostic(
        output.diagnostics,
        'gpu-fallback',
        'warning',
        `The GPU engine did not run: ${fallbackReason}. Results come from the CPU reference engine.`
      );
  }
  checkOutputCapacity(binding, output.tables);
  options.signal?.throwIfAborted();
  return {
    operationId: binding.id,
    engine,
    dispatchedContributors,
    ...(fallbackReason && engine === 'cpu-reference' ? {fallbackReason} : {}),
    parameters,
    tables: output.tables,
    diagnostics: output.diagnostics,
    summary: output.summary,
    ...(output.stages ? {stages: output.stages} : {})
  };
}

export type CatalogInvocation = {
  schemaVersion: 1;
  operationId: string;
  parameters: Record<string, CatalogParameterValue>;
  /** Inputs are referenced by content hash, never embedded. */
  inputs: Record<
    string,
    {
      featureCount: number;
      vertexCount: number;
      contentHash: string;
      dataKind?: 'table' | 'raster';
      rowCount?: number;
      cellCount?: number;
    }
  >;
};

export function hashCatalogCollection(collection: CatalogFeatureCollection): string {
  return hashCatalogText(JSON.stringify(collection));
}

/** Content hash including decoded raster samples, validity and georeference, or exact table keys. */
export function hashCatalogDataInput(input: CatalogDataInput): string {
  if (isCatalogFeatureInput(input)) return hashCatalogCollection(input);
  // JSON alone collapses NaN, infinities and negative zero. Preserve them for decoded bands
  // and scalar tables, where a caller may change a value without changing its validity mask.
  return hashCatalogText(
    JSON.stringify(input, (_name, value: unknown) => {
      if (typeof value !== 'number') return value;
      if (!Number.isFinite(value)) return {catalogNumber: String(value)};
      return Object.is(value, -0) ? {catalogNumber: '-0'} : value;
    })
  );
}

export function serializeCatalogInvocation(
  binding: CatalogOperationBinding,
  rawParameters: Readonly<Record<string, unknown>>,
  inputs: Readonly<Record<string, CatalogDataInput>>
): string {
  const invocation: CatalogInvocation = {
    schemaVersion: 1,
    operationId: binding.id,
    parameters: normalizeCatalogParameters(binding, rawParameters),
    inputs: Object.fromEntries(
      binding.inputs.map(spec => {
        const input = inputs[spec.name];
        const counts = isCatalogFeatureInput(input)
          ? {featureCount: input.features.length, vertexCount: countCollectionVertices(input)}
          : input.type === 'table'
            ? {
                featureCount: 0,
                vertexCount: 0,
                dataKind: 'table' as const,
                rowCount: input.table.rowCount
              }
            : {
                featureCount: 0,
                vertexCount: 0,
                dataKind: 'raster' as const,
                cellCount: input.grid.width * input.grid.height
              };
        return [spec.name, {...counts, contentHash: hashCatalogDataInput(input)}];
      })
    )
  };
  return JSON.stringify(invocation);
}

export function parseCatalogInvocation(
  text: string,
  options: {deferParameterValidation?: boolean} = {}
): {
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
      parameters: options.deferParameterValidation
        ? {...invocation.parameters}
        : normalizeCatalogParameters(binding, invocation.parameters)
    }
  };
}

/** Whether supplied inputs are the exact data a serialized invocation was created from. */
export function matchCatalogInvocationInputs(
  invocation: CatalogInvocation,
  inputs: Readonly<Record<string, CatalogDataInput>>
): boolean {
  return Object.entries(invocation.inputs).every(
    ([name, reference]) =>
      inputs[name] && hashCatalogDataInput(inputs[name]) === reference.contentHash
  );
}

/** Host panels that already run an operator natively (not through these catalog bindings). */
const NATIVE_PANEL_OPERATORS = [
  ...Object.entries(RASTER_TIMESERIES_PANEL_OPERATORS).map(([exportName, location]) => ({
    package: '@luma-contributors/analysis/raster-analysis',
    exportName,
    location
  })),
  {
    package: '@luma-contributors/analysis/dataframe-analysis',
    exportName: 'GPUVariableStandardization',
    location: 'Analysis panel · Standardize'
  },
  {
    package: '@luma-contributors/analysis/dataframe-analysis',
    exportName: 'GPUClassBreaks',
    location: 'Analysis panel · Classify'
  },
  {
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    exportName: 'GPUDorlingCartogram',
    location: 'Analysis panel · Cartogram'
  }
];

export type CatalogEntry = {
  /** Public package entry point and export name. */
  identity: {package: string; exportName: string};
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

export type CatalogEntryOptions = {
  /** Registered lens scenarios; defaults to whatever the registry holds now. */
  lensScenarios?: readonly CatalogLensScenario[];
};

/** Lists the full analysis catalog and qualified binding extensions, with host runnability. */
export function listCatalogEntries(options: CatalogEntryOptions = {}): CatalogEntry[] {
  return listQualifiedCatalogEntries(
    BINDINGS,
    options.lensScenarios ?? listCatalogLensScenarios(),
    NATIVE_PANEL_OPERATORS
  );
}
