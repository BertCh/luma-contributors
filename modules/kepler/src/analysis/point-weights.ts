// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {Kernel} from '@luma.gl/graph';
import {
  createTransientView,
  doGraphDataViewsOverlap,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUNeighborSearch,
  getGPUNeighborSearchParameterValues,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  type GPUChunkedSpatialWeights,
  type GPUNeighborSearchOutput,
  type GPUNeighborSearchStructure,
  type GPUSpatialWeights
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  parseKeplerPointWeightsSettings,
  DEFAULT_KEPLER_PERMUTATION_DEGREE,
  MAXIMUM_KEPLER_DISTANCE_BAND_ROWS,
  type KeplerDatasetReference,
  type KeplerPointWeightsSettings
} from '../recipes/analysis-recipe';

import {
  assertKeplerStorageBindingFits,
  getKeplerNeighborGridSize,
  KEPLER_ANALYSIS_ROW_TILE_SIZE
} from './analysis-scaling';
import {addKeplerNeighborTotalGuard} from './point-weights-guard';
import {
  getKeplerPointWeightBounds,
  getKeplerPointWeightSpatialContext
} from './point-weight-coordinates';

export type KeplerPointWeightsStatus = {
  overflow: GraphDataView<'uint32'>;
  requiredCount: GraphDataView<'uint32'>;
  invalidCount: GraphDataView<'uint32'>;
};

/** Caller-owned point topology imported into the current consumer graph. */
export type KeplerPointWeightsBinding = {
  source: KeplerDatasetReference;
  settings: KeplerPointWeightsSettings;
  /**
   * One CSR, or row chunks when the single-CSR storage exceeded the device binding limit (see
   * `buildKeplerStreamedPointWeights`); only consumers that accept `GPUChunkedSpatialWeights`
   * can use the chunked form.
   */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  coordinateMask: GraphDataView<'uint32'>;
  status: KeplerPointWeightsStatus;
};

export type KeplerPointWeightsProps = {
  id?: string;
  source: KeplerDatasetReference;

  /** Combined source coordinate null validity, independent of numeric validity and selection. */
  coordinateValidityMask?: GraphDataView<'uint32'>;
  /** Optional caller-owned coordinate inclusion mask, retained with persistent CSR outputs. */
  coordinateMask?: GraphDataView<'uint32'>;
  /** Optional persistent CSR. Otherwise outputs remain owned by the compiled graph. */
  output?: GPUNeighborSearchOutput;
  /** Optional persistent diagnostics when the CSR itself is graph-owned. */
  status?: KeplerPointWeightsStatus;
  /**
   * Search structure; results never depend on it. Defaults to the grid. `'bvh'` (planar settings
   * only) is an opt-in for strongly clustered projected points, where one grid cell can hold most
   * targets.
   */
  searchStructure?: GPUNeighborSearchStructure;
} & (
  | {
      settings: Extract<KeplerPointWeightsSettings, {coordinateSpace: 'longitude-latitude'}>;
      longitude: GraphDataView<'float32'>;
      latitude: GraphDataView<'float32'>;
    }
  | {
      settings: Extract<KeplerPointWeightsSettings, {coordinateSpace: 'planar'}>;
      x: GraphDataView<'float32'>;
      y: GraphDataView<'float32'>;
    }
);

/** Physical CSR slots, including one storage slot for a neighborless singleton. */
export function getKeplerPointWeightsCapacity(
  settings: KeplerPointWeightsSettings,
  rowCount: number
): number {
  const normalized = parseKeplerPointWeightsSettings(settings);
  if (!Number.isSafeInteger(rowCount) || rowCount < 1 || rowCount > 2 ** 24)
    throw new KeplerAllocationError('capacity', {
      message: 'Point weights require source rows within capacity'
    });
  if (normalized.kind === 'distance-band' && rowCount > MAXIMUM_KEPLER_DISTANCE_BAND_ROWS)
    throw new Error('Distance-band source rows exceed uint32 neighbor-count capacity');
  const capacity =
    normalized.kind === 'knn'
      ? Math.max(1, rowCount * Math.min(normalized.k, rowCount - 1))
      : normalized.neighborCapacity;
  if (capacity > 2 ** 31 - 1)
    throw new KeplerAllocationError('capacity', {
      message: 'Point weights exceed neighbor capacity'
    });
  return capacity;
}

/**
 * Permutation scratch limit, not a degree cutoff: band rows above the declared permutation degree
 * (64 unless the settings opt in to up to 512) report inference overflow.
 */
export function getKeplerPointWeightsMaximumNeighbors(
  settings: KeplerPointWeightsSettings
): number {
  return settings.kind === 'knn'
    ? settings.k
    : (settings.permutationDegree ?? DEFAULT_KEPLER_PERMUTATION_DEGREE);
}

/**
 * Build reusable binary kNN or distance-band weights across all valid source points. Coordinates explicitly declare geographic degrees with spherical distance or projected XY meters with Euclidean distance. No selection or rate-input mask
 * changes this graph: consumers filter its existing neighbors without refilling excluded slots.
 * Invalid-coordinate diagnostics include source nulls, nonfinite values and coordinates outside the declared bounds.
 *
 * The caller compiles, encodes and submits. Keep parameters alive through completion and destroy
 * them with destroyParameters(); source data is borrowed and never repacked in place.
 */
export function addKeplerPointWeights<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerPointWeightsProps
) {
  const id = props.id ?? 'kepler-point-weights';
  const settings = parseKeplerPointWeightsSettings(props.settings);
  if ((settings.coordinateSpace === 'planar') !== 'x' in props)
    throw new Error('Point coordinate ports must match the declared coordinate space');
  const firstCoordinate = 'x' in props ? props.x : props.longitude;
  const secondCoordinate = 'y' in props ? props.y : props.latitude;
  const rowCount = firstCoordinate.length;
  validatePackedView(firstCoordinate, ['float32'], `${id} first coordinate`);
  validatePackedView(secondCoordinate, ['float32'], `${id} second coordinate`);
  if (
    !props.source.datasetId ||
    !Number.isSafeInteger(props.source.revision) ||
    props.source.revision < 1 ||
    rowCount < 1 ||
    rowCount > 2 ** 24 ||
    secondCoordinate.length !== rowCount
  )
    throw new Error('Point weights require matching source coordinates and a current source');
  if (props.coordinateValidityMask) {
    validatePackedView(props.coordinateValidityMask, ['uint32'], `${id} coordinate validity`);
    if (props.coordinateValidityMask.length !== rowCount)
      throw new Error('Coordinate validity must match point rows');
  }
  // CSR storage needs one physical slot even when a singleton has no logical neighbors.
  const neighborCapacity = getKeplerPointWeightsCapacity(settings, rowCount);
  if (
    settings.kind === 'distance-band' &&
    props.output &&
    props.output.neighbors.length !== neighborCapacity
  )
    throw new Error('Distance-band output must match declared neighbor capacity');
  if (props.output && props.status) throw new Error('Provide output or status, not both');
  // Two neighbor-slot columns (ids and weights) and the CSR offsets must each fit one binding.
  assertKeplerStorageBindingFits(graph.device.limits, neighborCapacity, 4, 'neighbor slots');
  assertKeplerStorageBindingFits(graph.device.limits, rowCount + 1, 4, 'neighbor offsets');
  const positions = createTransientView(graph, `${id}-positions`, 'float32x2', rowCount);
  const coordinateMask =
    props.coordinateMask ?? createTransientView(graph, `${id}-coordinate-mask`, 'uint32', rowCount);
  validatePackedView(coordinateMask, ['uint32'], `${id} coordinate mask`);
  if (coordinateMask.length !== rowCount) throw new Error('Coordinate mask must match point rows');
  if (
    [firstCoordinate, secondCoordinate, props.coordinateValidityMask].some(
      input => input && doGraphDataViewsOverlap(coordinateMask, input)
    )
  )
    throw new Error('Coordinate output must not overlap source inputs');
  addCoordinatePacking(
    graph,
    id,
    {
      firstCoordinate,
      secondCoordinate,
      coordinateValidityMask: props.coordinateValidityMask,
      bounds: getKeplerPointWeightBounds(settings),
      planar: settings.coordinateSpace === 'planar'
    },
    positions,
    coordinateMask
  );
  const status = props.output?.status ??
    props.status ?? {
      overflow: createTransientView(graph, `${id}-overflow`, 'uint32', 1),
      requiredCount: createTransientView(graph, `${id}-required-count`, 'uint32', 1),
      invalidCount: createTransientView(graph, `${id}-invalid-count`, 'uint32', 1)
    };
  // Above 65536 rows the 32-bit total slot count could wrap, so per-row counts are summed exactly.
  const guardTotal = settings.kind === 'distance-band' && rowCount > 65536;
  const output: GPUNeighborSearchOutput = props.output ?? {
    offsets: createTransientView(graph, `${id}-offsets`, 'uint32', rowCount + 1),
    neighbors: createTransientView(graph, `${id}-neighbors`, 'uint32', neighborCapacity),
    weights: createTransientView(graph, `${id}-weights`, 'float32', neighborCapacity),
    ...(guardTotal
      ? {neighborCounts: createTransientView(graph, `${id}-neighbor-counts`, 'uint32', rowCount)}
      : {}),
    status
  };
  // The guard needs unclamped per-row counts: a caller-owned CSR must supply them.
  if (guardTotal && !output.neighborCounts)
    throw new Error('Large distance-band output requires neighborCounts');
  const spatialContext = getKeplerPointWeightSpatialContext(settings);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: `${id}-parameters`,
    format: 'float32',
    length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
    values: getGPUNeighborSearchParameterValues({
      spatialContext,
      bounds: getKeplerPointWeightBounds(settings),
      ...(settings.kind === 'distance-band' ? {radius: settings.radiusMeters} : {}),
      weightKind: 'binary',
      rowStandardize: false
    })
  });
  try {
    const search = new GPUNeighborSearch({
      id,
      spatialContext,
      mode: settings.kind === 'knn' ? 'knn' : 'radius',
      positions,
      parameters: parameters.importToGraph(graph),
      gridSize: getKeplerNeighborGridSize(
        rowCount,
        settings.coordinateSpace === 'planar' ? settings.bounds : undefined
      ),
      ...(props.searchStructure ? {searchStructure: props.searchStructure} : {}),
      rowTileSize: KEPLER_ANALYSIS_ROW_TILE_SIZE,
      ...(settings.kind === 'knn' ? {k: settings.k} : {}),
      weightTransform: 'binary',
      rowStandardize: false,
      output
    });
    graph.add(search);
    if (guardTotal && output.neighborCounts)
      addKeplerNeighborTotalGuard(graph, id, output.neighborCounts, status);
    return {
      source: Object.freeze({...props.source}),
      settings,
      weights: search.outputWeights,
      status,
      coordinateMask,
      parameterByteLength: parameters.buffer.byteLength,
      destroyParameters() {
        parameters.destroy();
      }
    };
  } catch (error) {
    parameters.destroy();
    throw error;
  }
}

export function addCoordinatePacking<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: {
    firstCoordinate: GraphDataView<'float32'>;
    secondCoordinate: GraphDataView<'float32'>;
    coordinateValidityMask?: GraphDataView<'uint32'>;
    bounds: readonly [number, number, number, number];
    planar: boolean;
  },
  positions: GraphDataView<'float32x2'>,
  coordinateMask: GraphDataView<'uint32'>
): void {
  const {firstCoordinate, secondCoordinate, coordinateValidityMask, bounds, planar} = props;
  const dispatch = getBoundedDispatchLayout(
    id,
    firstCoordinate.length,
    64,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  graph.addComputePass({
    id: `${id}-pack-coordinates`,
    resources: [
      {buffer: firstCoordinate, usage: 'storage-read'},
      {buffer: secondCoordinate, usage: 'storage-read'},
      ...(coordinateValidityMask
        ? [{buffer: coordinateValidityMask, usage: 'storage-read' as const}]
        : []),
      {buffer: positions, usage: 'storage-read-write'},
      {buffer: coordinateMask, usage: 'storage-read-write'}
    ],
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: `${id}-pack-coordinates`,
        source: /* wgsl */ `
@group(0) @binding(0) var<storage, read> firstCoordinate: array<f32>;
@group(0) @binding(1) var<storage, read> secondCoordinate: array<f32>;
@group(0) @binding(2) var<storage, read_write> positions: array<f32>;
@group(0) @binding(3) var<storage, read_write> coordinateMask: array<u32>;
${coordinateValidityMask ? '@group(0) @binding(4) var<storage, read> validity: array<u32>;' : ''}
@compute @workgroup_size(64) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatch, 64)}
  if (index >= ${firstCoordinate.length}u) { return; }
  let firstCoordinateValue = firstCoordinate[${getViewElementOffset(firstCoordinate)}u + index];
  let secondCoordinateValue = secondCoordinate[${getViewElementOffset(secondCoordinate)}u + index];
  let included = ${coordinateValidityMask ? `validity[${getViewElementOffset(coordinateValidityMask)}u + index] != 0u` : 'true'};
  // Missing coordinates become invalid positions so completeness counts include source nulls.
  let missing = ${planar ? 'bitcast<f32>(0x7f800000u | (index & 0u))' : '181.0'};
  positions[${getViewElementOffset(positions)}u + index * 2u] = select(missing, firstCoordinateValue, included);
  positions[${getViewElementOffset(positions)}u + index * 2u + 1u] = select(missing, secondCoordinateValue, included);
  let finiteFirstCoordinate = (bitcast<u32>(firstCoordinateValue) & 0x7f800000u) != 0x7f800000u;
  let finiteSecondCoordinate = (bitcast<u32>(secondCoordinateValue) & 0x7f800000u) != 0x7f800000u;
  let valid = finiteFirstCoordinate && finiteSecondCoordinate && firstCoordinateValue >= ${bounds[0].toPrecision(17)} && firstCoordinateValue <= ${bounds[2].toPrecision(17)}
    && secondCoordinateValue >= ${bounds[1].toPrecision(17)} && secondCoordinateValue <= ${bounds[3].toPrecision(17)}
    && included;
  coordinateMask[${getViewElementOffset(coordinateMask)}u + index] = select(0u, 1u, valid);
}`,
        shaderLayout: {
          bindings: [
            {name: 'firstCoordinate', type: 'storage', group: 0, location: 0},
            {name: 'secondCoordinate', type: 'storage', group: 0, location: 1},
            {name: 'positions', type: 'storage', group: 0, location: 2},
            {name: 'coordinateMask', type: 'storage', group: 0, location: 3},
            ...(coordinateValidityMask
              ? [{name: 'validity', type: 'storage' as const, group: 0, location: 4}]
              : [])
          ]
        }
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: {
              firstCoordinate: getViewBinding(firstCoordinate, getBuffer),
              secondCoordinate: getViewBinding(secondCoordinate, getBuffer),
              positions: getViewBinding(positions, getBuffer),
              coordinateMask: getViewBinding(coordinateMask, getBuffer),
              ...(coordinateValidityMask
                ? {validity: getViewBinding(coordinateValidityMask, getBuffer)}
                : {})
            },
            ...dispatch
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
}
