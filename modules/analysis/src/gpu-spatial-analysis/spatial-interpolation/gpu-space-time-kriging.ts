// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotInterpolationConstructionProps} from './interpolation-construction-snapshot';

import {
  createTransientView,
  getViewBindingRange,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  assertCompatibleGPUTemporalContexts,
  resolveGPUTemporalContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/temporal-context';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  getGraphViewChunks,
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {
  createGraphChunkGroupKernelNode,
  validateGraphChunkGroupKernelSupport
} from '../../analysis-core/kernels/graph-chunk-groups';
import {validateSpaceTimeKrigingOutputs} from './space-time-kriging-outputs';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH} from './prediction-covariance';
import {createSpaceTimeKrigingPredictionNode} from './space-time-kriging-kernel';
import {
  createSpaceTimeKrigingIndexNodes,
  SPACE_TIME_KRIGING_INDEX_PARAMETER_LENGTH
} from './space-time-kriging-index';

/** Dense local solve uses at most 32 retained samples per invocation. */
const MAXIMUM_NEIGHBOR_COUNT = 32;
/** Reference route scans every sample for every query; retain its historical ceiling. */
export const GPU_SPACE_TIME_KRIGING_MAXIMUM_SAMPLE_COUNT = 1048576;
/** Above this cut-over the default route builds a metric grid instead of scanning samples. */
export const GPU_SPACE_TIME_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT = 4096;
/** Packed float32x4 indices plus a 63-word binding prefix must fit uint32; count scans cannot wrap. */
export const GPU_SPACE_TIME_KRIGING_INDEX_MAXIMUM_SAMPLE_COUNT = 0x3ffffff0;
/** Per-invocation candidate arrays are bounded to 64 slots. */
const MAXIMUM_STAGED_CANDIDATE_COUNT = 64;
/** Eight octants times eight candidates fits the same 64-slot scratch bound. */
const MAXIMUM_NEIGHBORS_PER_OCTANT = 8;
/** Cell coordinates divide by gridSide - 1, so each axis needs at least two cells. */
const MINIMUM_GRID_SIDE = 2;
/** Limits grid metadata to 128^3 cells, or approximately 8 MiB of exclusive offsets. */
const MAXIMUM_GRID_SIDE = 128;
/** About 16 samples per cell balances grid metadata against bounded neighbor walks. */
const INDEX_SAMPLES_PER_CELL = 16;
/** A three-axis extent build plus at most 22 radix key bits and four hierarchical levels fits 512 nodes. */
const INDEX_COMMAND_NODE_ALLOWANCE = 512;
/** Each runtime chunk slot starts at the 256-byte storage binding alignment. */
const CHUNK_PARAMETER_STRIDE = 64;
/** Runtime count plus component offsets of positions, times, values and status. */
const CHUNK_PARAMETER_LENGTH = 5;
/** Query output component indices plus a 63-word aligned-binding prefix must fit uint32. */
const MAXIMUM_QUERY_CHUNK_LENGTH = 0x3ffffff0;

/** Neighbor selection of {@link GPUSpaceTimeKriging}; see the class documentation. */
export type GPUSpaceTimeKrigingNeighborSearch = 'metric' | 'octant' | 'staged';

/** Query and output view kinds, inferred independently from the supplied ports. @internal */
export type SpaceTimeKrigingRowViews = {
  /** Packed or chunked query positions; queryTimes and outputs match its chunk boundaries. */
  queries: GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;
  /** Relative float32 time per query, same epoch and unit as `times`. */
  queryTimes: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  output: {
    /** `[prediction, latent variance, pivot ratio, retained neighbor count]`. */
    values: GraphDataView<'float32x4'> | GraphVectorView<'float32x4'>;
    /** 0 success; 1 invalid or insufficient data; 2 singular or unreliable. */
    status: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  };
};

/** Preserve the original packed public types when no view-kind argument is supplied. */
type SpaceTimeKrigingPackedRowViews = {
  queries: GraphDataView<'float32x2'>;
  queryTimes: GraphDataView<'float32'>;
  output: {
    values: GraphDataView<'float32x4'>;
    status: GraphDataView<'uint32'>;
  };
};

/** Props of {@link GPUSpaceTimeKriging}. */
export type GPUSpaceTimeKrigingProps<
  Rows extends SpaceTimeKrigingRowViews = SpaceTimeKrigingPackedRowViews
> = {
  id?: string;
  /** Planar coordinates only; project longitude and latitude first. */
  spatialContext?: GPUSpatialContext;
  /** Context of `times`, encoding `float32-relative` (`GPUTimeWordsRelative` provides one). */
  temporalContext: GPUTemporalContext;
  /** Context of `queryTimes`, default `temporalContext`; must have the same unit and epoch. */
  queryTemporalContext?: GPUTemporalContext;
  positions: GraphDataView<'float32x2'>;
  /** Relative float32 time per sample. */
  times: GraphDataView<'float32'>;
  /** Rows `[value, observationErrorVariance, reserved, validity (> 0 includes; zero, negative and NaN skip)]`. */
  observations: GraphDataView<'float32x4'>;
  /** Known mean or unknown constant mean. */
  mode: 'simple' | 'ordinary';
  /** Written by `getGPUSpaceTimeKrigingParameterValues`; every value changes without a recompile. */
  parameters: GraphDataView<'float32'>;
  /** Retained neighbors, 1..32 (compile time). Defaults to 16. */
  neighborCount?: number;
  /** Neighbor selection (compile time). Defaults to `'metric'`. */
  neighborSearch?: GPUSpaceTimeKrigingNeighborSearch;
  /** Candidates per octant for `'octant'`, 1..8. Defaults to `ceil(neighborCount / 8)`. */
  neighborsPerOctant?: number;
  /** Metric candidates for `'staged'`, `neighborCount..64`. Defaults to `2 * neighborCount`. */
  stagedCandidateCount?: number;
  /**
   * Compile-time, default 'auto': scans up to 4096 samples, then uses the exact metric grid.
   * Explicit 'brute-force' retains the 1,048,576-sample reference ceiling; 'grid' forces indexing.
   */
  sampleIndex?: 'auto' | 'brute-force' | 'grid';
} & Rows;

/**
 * Local space-time kriging with one neighbor search per query, over a per-frame covariance from
 * `SpaceTimeVariogramModel` (the gstat definitions, plus Gneiting). Uses the same equilibrated
 * partial-pivot solve and the same pivot-ratio limit as `GPULocalKriging`, and keeps that
 * contributor's solver. Auto scans up to the named cut-over, then builds a per-frame 3D metric
 * grid with published reduction, aggregation, scan and sort. Explicit brute-force keeps the
 * reference sample limit. Both routes keep identical neighbors, including distance ties.
 *
 * Neighbor search, compile-time:
 * - `'metric'`: the `neighborCount` samples nearest in `sqrt(h^2 + (searchAnisotropy u)^2)`.
 * - `'octant'`: per quadrant of the sample offset and per side in time (`time < query` is the past,
 *   otherwise the future), at most `neighborsPerOctant` nearest in the metric; then the
 *   `neighborCount` nearest of those. A sample exactly at the query location goes to the quadrant
 *   with positive offsets.
 * - `'staged'`: the `stagedCandidateCount` samples nearest in the metric, then the `neighborCount`
 *   of those with the largest model covariance to the query, ties going to the nearer candidate.
 *
 * `gstat` documents staged and octant search as not yet available, so these are this library's
 * definitions. `searchRadius` and `maximumTimeLag` exclude samples before the search. Equal metric
 * values resolve to the lower row.
 *
 * Query positions, times and outputs may be chunked with matching row boundaries. Only the
 * final output chunk may have spare capacity. Each chunk dispatches its own workgroup count. Samples
 * and parameters remain packed. On the ordinary indexed route with one query and one neighbor,
 * the 16-byte observation plane fits 8,388,608 samples in a 128 MiB storage binding at zero
 * offset, or 8,388,607 after a 16-byte prefix; other bindings and scratch must also fit.
 * Query chunking does not split samples. Auto scans at most 4,096 samples, forced brute-force
 * scans at most 1,048,576 samples, and indexing requires at least eight storage bindings per stage.
 * The grid walks expanding shells in anisotropy-scaled coordinates,
 * pruning only with conservative distance bounds. Coincident samples, very wide distance ties,
 * restrictive filters or unfilled octant quotas can require visiting the entire index.
 *
 * Counts, chunk boundaries, sampleIndex, the neighbor search and the mode specialize the graph.
 * Every input and covariance parameter changes per frame without a recompile. Prediction and
 * variance agree with the reference route to the existing float32 tolerance (5e-4 on ordinary
 * inputs); both routes run the same solve in the same neighbor order.
 */
export class GPUSpaceTimeKriging<
  Rows extends SpaceTimeKrigingRowViews = SpaceTimeKrigingPackedRowViews
> implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: GPUSpaceTimeKrigingProps<Rows>;
  readonly neighborCount: number;
  readonly neighborSearch: GPUSpaceTimeKrigingNeighborSearch;
  readonly neighborsPerOctant: number;
  readonly stagedCandidateCount: number;
  readonly queryCount: number;
  readonly temporalContext: GPUTemporalContext;
  readonly sampleIndex: 'brute-force' | 'grid';
  readonly gridSide: number;
  readonly estimatedCommandNodeCount: number;
  readonly queryChunks: readonly {
    queries: GraphDataView<'float32x2'>;
    queryTimes: GraphDataView<'float32'>;
    values: GraphDataView<'float32x4'>;
    status: GraphDataView<'uint32'>;
  }[];

  constructor(props: GPUSpaceTimeKrigingProps<Rows>) {
    this.props = props = snapshotInterpolationConstructionProps(
      props,
      [
        'id',
        'mode',
        'neighborCount',
        'neighborSearch',
        'neighborsPerOctant',
        'observations',
        'output',
        'parameters',
        'positions',
        'queries',
        'queryTemporalContext',
        'queryTimes',
        'sampleIndex',
        'spatialContext',
        'stagedCandidateCount',
        'temporalContext',
        'times'
      ],
      ['values', 'status']
    );
    this.id = props.id ?? 'space-time-kriging';
    const {id} = this;
    resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    const support = {encodings: ['float32-relative'] as const};
    this.temporalContext = resolveGPUTemporalContext(id, props.temporalContext, support);
    if (props.queryTemporalContext) {
      resolveGPUTemporalContext(id, props.queryTemporalContext, support);
      assertCompatibleGPUTemporalContexts(id, props.temporalContext, props.queryTemporalContext);
    }
    this.neighborCount = props.neighborCount ?? 16;
    this.neighborSearch = props.neighborSearch ?? 'metric';
    this.neighborsPerOctant = props.neighborsPerOctant ?? Math.ceil(this.neighborCount / 8);
    this.stagedCandidateCount = props.stagedCandidateCount ?? 2 * this.neighborCount;
    this.queryCount = props.queries.length;
    validateRequirement(
      id,
      ['auto', 'brute-force', 'grid'].includes(props.sampleIndex ?? 'auto'),
      "sampleIndex must be 'auto', 'brute-force' or 'grid'"
    );
    this.sampleIndex =
      props.sampleIndex === 'grid' ||
      ((props.sampleIndex ?? 'auto') === 'auto' &&
        props.positions.length > GPU_SPACE_TIME_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT)
        ? 'grid'
        : 'brute-force';
    validateCapacity(
      id,
      'positions.length',
      props.positions.length,
      1,
      this.sampleIndex === 'grid'
        ? GPU_SPACE_TIME_KRIGING_INDEX_MAXIMUM_SAMPLE_COUNT
        : GPU_SPACE_TIME_KRIGING_MAXIMUM_SAMPLE_COUNT
    );
    validateCapacity(id, 'neighborCount', this.neighborCount, 1, MAXIMUM_NEIGHBOR_COUNT);
    validateCapacity(id, 'queryCount', this.queryCount, 0);
    validateCapacity(
      id,
      'neighborsPerOctant',
      this.neighborsPerOctant,
      1,
      MAXIMUM_NEIGHBORS_PER_OCTANT
    );
    validateCapacity(
      id,
      'stagedCandidateCount',
      this.stagedCandidateCount,
      this.neighborCount,
      MAXIMUM_STAGED_CANDIDATE_COUNT
    );
    validateRequirement(
      id,
      ['metric', 'octant', 'staged'].includes(this.neighborSearch),
      "neighborSearch must be 'metric', 'octant' or 'staged'"
    );
    validateRequirement(
      id,
      ['simple', 'ordinary'].includes(props.mode),
      "mode must be 'simple' or 'ordinary'"
    );
    validateRequirement(
      id,
      props.times.length === props.positions.length,
      'times.length must equal positions.length'
    );
    validateRequirement(
      id,
      props.observations.length === props.positions.length,
      'observations.length must equal positions.length'
    );
    validateCapacity(
      id,
      'parameters.length',
      props.parameters.length,
      GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH
    );
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    validateSinglePackedView(props.times, ['float32'], `${id} times`);
    validateSinglePackedView(props.observations, ['float32x4'], `${id} observations`);
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    const queryTimes = getGraphViewChunks(props.queryTimes);
    const values = getGraphViewChunks(props.output.values);
    const status = getGraphViewChunks(props.output.status);
    const queries = getGraphViewChunks(props.queries);
    validateRequirement(
      id,
      queryTimes.length === queries.length &&
        values.length === queries.length &&
        status.length === queries.length,
      'queryTimes and output chunk counts must match queries'
    );
    this.queryChunks = queries.map((queryChunk, chunkIndex) => {
      const chunk = {
        queries: queryChunk,
        queryTimes: queryTimes[chunkIndex],
        values: values[chunkIndex],
        status: status[chunkIndex]
      };
      validateCapacity(
        id,
        `queries chunk ${chunkIndex} length`,
        queryChunk.length,
        0,
        MAXIMUM_QUERY_CHUNK_LENGTH
      );
      validateRequirement(
        id,
        chunk.queryTimes.length === queryChunk.length,
        'queryTimes chunk lengths must match queries'
      );
      validateRequirement(
        id,
        chunk.values.length >= queryChunk.length && chunk.status.length >= queryChunk.length,
        'output chunk capacities must cover queries'
      );
      validateRequirement(
        id,
        chunkIndex === queries.length - 1 ||
          (chunk.values.length === queryChunk.length && chunk.status.length === queryChunk.length),
        'non-final output chunk lengths must match queries; spare capacity is allowed only in the final chunk'
      );
      validateSinglePackedView(chunk.queries, ['float32x2'], `${id} queries`);
      validateSinglePackedView(chunk.queryTimes, ['float32'], `${id} queryTimes`);
      validateSinglePackedView(chunk.values, ['float32x4'], `${id} output.values`);
      validateSinglePackedView(chunk.status, ['uint32'], `${id} output.status`);
      return chunk;
    });
    validateRequirement(
      id,
      queries.reduce((total, chunk) => total + chunk.length, 0) === this.queryCount,
      'queries.length must equal its chunk total'
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.values, props.output.status],
      [
        props.positions,
        props.times,
        props.observations,
        props.queries,
        props.queryTimes,
        props.parameters
      ]
    );
    validateSpaceTimeKrigingOutputs(id, props.output);
    this.gridSide = Math.max(
      MINIMUM_GRID_SIDE,
      Math.min(
        MAXIMUM_GRID_SIDE,
        Math.ceil(Math.cbrt(props.positions.length / INDEX_SAMPLES_PER_CELL))
      )
    );
    const nonemptyChunks = this.queryChunks.filter(chunk => chunk.queries.length > 0).length;
    this.estimatedCommandNodeCount =
      nonemptyChunks === 0
        ? 0
        : 4 * nonemptyChunks + 3 + (this.sampleIndex === 'grid' ? INDEX_COMMAND_NODE_ALLOWANCE : 0);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.times,
      props.observations,
      props.parameters,
      props.queries,
      props.queryTimes,
      props.output.values,
      props.output.status
    ]);
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      times: props.times,
      observations: props.observations,
      parameters: props.parameters
    });
    const nonemptyChunks = this.queryChunks.filter(chunk => chunk.queries.length > 0);
    // Empty chunks are never bound. In particular a trailing empty view needs no backing row.
    for (const chunk of nonemptyChunks) validateGraphChunkBindingRanges(graph, id, chunk);
    const maximumChunkQueryCount = Math.max(
      0,
      ...nonemptyChunks.map(chunk => chunk.queries.length)
    );
    const sampleCount = props.positions.length;
    const cellCount = this.gridSide ** 3;
    const scratch = [
      ['sample geometry', sampleCount * 12],
      ['query geometry', maximumChunkQueryCount * 12],
      ['query values', maximumChunkQueryCount * 16],
      ['query status', maximumChunkQueryCount * 4],
      ['private parameters', SPACE_TIME_KRIGING_INDEX_PARAMETER_LENGTH * 4],
      ['chunk parameter slot', CHUNK_PARAMETER_LENGTH * 4],
      ...(this.sampleIndex === 'grid'
        ? [
            ['axis coordinates, keys, sorted keys and row IDs', sampleCount * 4],
            [
              'small bitonic sort scratch',
              sampleCount <= 256 ? 2 ** Math.ceil(Math.log2(sampleCount)) * 4 : 0
            ],
            ['extent reduction partials', Math.ceil(sampleCount / 256) * 8],
            ['cell counts and offsets', (cellCount + 1) * 4],
            ['radix sort histograms', Math.ceil(sampleCount / 256) * 16 * 4]
          ]
        : [])
    ] as const;
    for (const [name, byteLength] of scratch) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        String(name),
        Math.max(Number(byteLength), 4),
        'reduce sample count or query chunk length'
      );
    }
    validateCapacity(
      id,
      'chunk parameter buffer byteLength',
      nonemptyChunks.length * CHUNK_PARAMETER_STRIDE * 4,
      0,
      graph.device.limits.maxBufferSize
    );
    validateRequirement(
      id,
      graph.device.limits.maxStorageBuffersPerShaderStage >= (this.sampleIndex === 'grid' ? 8 : 6),
      `sampleIndex ${this.sampleIndex} requires at least ${this.sampleIndex === 'grid' ? 8 : 6} storage bindings`
    );
    validateGraphChunkGroupKernelSupport(graph, id, this.sampleIndex === 'grid' ? 8 : 6, 32);
    if (nonemptyChunks.length === 0) return [];
    const geometry = createTransientView(graph, `${id}-geometry`, 'float32x3', sampleCount);
    const parameters = createTransientView(
      graph,
      `${id}-parameters`,
      'float32',
      SPACE_TIME_KRIGING_INDEX_PARAMETER_LENGTH
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-pack-samples`,
        operation: 'GPUSpaceTimeKriging',
        invocationCount: sampleCount,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'times', view: props.times, type: 'f32', access: 'read'},
          {name: 'geometry', view: geometry, type: 'f32', access: 'read_write'}
        ],
        body: 'geometry[index * 3u] = positions[positionsOffset + index * 2u]; geometry[index * 3u + 1u] = positions[positionsOffset + index * 2u + 1u]; geometry[index * 3u + 2u] = times[timesOffset + index];'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-pack-parameters`,
        operation: 'GPUSpaceTimeKriging',
        invocationCount: GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH,
        bindings: [
          {name: 'sourceParameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'params', view: parameters, type: 'f32', access: 'read_write'}
        ],
        body: 'params[index] = sourceParameters[sourceParametersOffset + index];'
      })
    ];
    const grid =
      this.sampleIndex === 'grid'
        ? createSpaceTimeKrigingIndexNodes(
            graph,
            id,
            geometry,
            props.observations,
            parameters,
            this.gridSide
          )
        : undefined;
    if (grid) nodes.push(...grid.nodes);
    const chunkValues = new Uint32Array(nonemptyChunks.length * CHUNK_PARAMETER_STRIDE);
    for (const [chunkIndex, chunk] of nonemptyChunks.entries()) {
      chunkValues.set(
        [
          chunk.queries.length,
          getViewElementOffset(chunk.queries),
          getViewElementOffset(chunk.queryTimes),
          getViewElementOffset(chunk.values),
          getViewElementOffset(chunk.status)
        ],
        chunkIndex * CHUNK_PARAMETER_STRIDE
      );
    }
    const chunkParameters = createSpatialStatisticsParameterNodes(
      graph,
      `${id}-chunks`,
      chunkValues
    );
    nodes.push(...chunkParameters.nodes);
    for (const [chunkIndex, chunk] of nonemptyChunks.entries()) {
      const chunkId = `${id}-chunk-${chunkIndex}`;
      const slot = graph.createDataView(chunkParameters.parameters.buffer, {
        format: 'uint32',
        length: CHUNK_PARAMETER_LENGTH,
        byteOffset: chunkIndex * CHUNK_PARAMETER_STRIDE * 4
      });
      const queryGeometry = createTransientView(
        graph,
        `${chunkId}-geometry`,
        'float32x3',
        chunk.queries.length
      );
      const values = createTransientView(
        graph,
        `${chunkId}-values`,
        'float32x4',
        chunk.queries.length
      );
      const status = createTransientView(
        graph,
        `${chunkId}-status`,
        'uint32',
        chunk.queries.length
      );
      // Bind an aligned scalar range; per-chunk component offsets live in the runtime slot.
      const alignView = (view: GraphDataView, format: 'float32' | 'uint32') => {
        const range = getViewBindingRange(view);
        return graph.createDataView(view.buffer, {
          format,
          byteOffset: range.offset,
          length: range.size / 4
        });
      };
      nodes.push(
        createGraphChunkGroupKernelNode(graph, {
          id: `${chunkId}-pack`,
          operation: 'GPUSpaceTimeKriging',
          invocationCount: chunk.queries.length,
          workgroupSize: 32,
          bindings: [
            {
              name: 'queries',
              view: alignView(chunk.queries, 'float32'),
              type: 'f32',
              access: 'read'
            },
            {
              name: 'queryTimes',
              view: alignView(chunk.queryTimes, 'float32'),
              type: 'f32',
              access: 'read'
            },
            {name: 'chunkParameters', view: slot, type: 'u32', access: 'read'},
            {name: 'queryGeometry', view: queryGeometry, type: 'f32', access: 'read_write'}
          ],
          body: `if (index >= chunkParameters[0u]) { return; }
queryGeometry[index * 3u] = queries[chunkParameters[1u] + index * 2u];
queryGeometry[index * 3u + 1u] = queries[chunkParameters[1u] + index * 2u + 1u];
queryGeometry[index * 3u + 2u] = queryTimes[chunkParameters[2u] + index];`
        })
      );
      nodes.push(
        createSpaceTimeKrigingPredictionNode(
          graph,
          this,
          `${chunkId}-predict`,
          chunk.queries.length,
          geometry,
          queryGeometry,
          parameters,
          values,
          status,
          grid
        )
      );
      // Publish fields separately: disjoint fields can share a buffer even when their aligned
      // binding prefixes overlap. Each dispatch has just one writable caller-owned range.
      for (const [field, source, destination, components, offsetSlot] of [
        ['values', values, chunk.values, 4, 3],
        ['status', status, chunk.status, 1, 4]
      ] as const) {
        const type = field === 'values' ? 'f32' : 'u32';
        nodes.push(
          createGraphChunkGroupKernelNode(graph, {
            id: `${chunkId}-publish-${field}`,
            operation: 'GPUSpaceTimeKriging',
            invocationCount: chunk.queries.length,
            workgroupSize: 32,
            bindings: [
              {name: 'sourceValues', view: source, type, access: 'read'},
              {name: 'chunkParameters', view: slot, type: 'u32', access: 'read'},
              {
                name: 'outputValues',
                view: alignView(destination, field === 'values' ? 'float32' : 'uint32'),
                type,
                access: 'read_write'
              }
            ],
            body: `if (index >= chunkParameters[0u]) { return; }
for (var component = 0u; component < ${components}u; component++) { outputValues[chunkParameters[${offsetSlot}u] + index * ${components}u + component] = sourceValues[index * ${components}u + component]; }`
          })
        );
      }
    }
    return nodes;
  }
}
