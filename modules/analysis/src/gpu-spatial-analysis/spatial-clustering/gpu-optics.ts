// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {
  getGPUSphereRadius,
  resolveGPUSpatialContext,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPUTemporalContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/temporal-context';
import {createOPTICSXiNode} from './gpu-optics-xi';
import {GPUNeighborSearch} from '../neighbor-search/gpu-neighbor-search';
import {createNeighborSearchDerivedParameterNodes} from '../neighbor-search/neighbor-search-derived-parameters';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validatePointProcessPositive,
  validatePointProcessView
} from '../point-process/point-process-utils';

/** Time buckets of the core-distance search; only its speed depends on this. */
const OPTICS_TIME_BUCKET_COUNT = 8;

/** Caller-owned exact OPTICS ordering and DBSCAN/xi extraction. */
export type GPUOPTICSOutput = {
  /** Input row IDs in visit order; unused slots are 0xffffffff. */
  ordering: GraphDataView<'uint32'>;
  /** Self-inclusive minimumSamples-th distance; infinity for non-core rows. */
  coreDistances: GraphDataView<'float32'>;
  /** Reachability indexed by input row; infinity for roots and unvisited rows. */
  reachability: GraphDataView<'float32'>;
  /** Predecessor input row, or 0xffffffff. */
  predecessors: GraphDataView<'uint32'>;
  /** Selected extraction indexed by input row; noise is 0xffffffff. */
  labels: GraphDataView<'uint32'>;
  /** [visitedCount, complete, invalidRowCount]. Invalid positions are excluded. */
  status: GraphDataView<'uint32'>;
  /** Optional xi hierarchy, inclusive positions in ordering; capacity must equal input rows. */
  clusterIntervals?: GraphDataView<'uint32x2'>;
  /** Optional [number of xi intervals, overflow]. */
  clusterSummary?: GraphDataView<'uint32'>;
};

/** Bounded exact planar or spherical OPTICS; all data-dependent work executes on the GPU. */
export type GPUOPTICSProps = {
  id?: string;
  /** Defaults to planar native distances. Geographic positions are lon/lat degrees with
   * great-circle distances in meters (or declared kilometers); antimeridian pairs are supported.
   * Absolute Float32 degrees retain their original quantization. */
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  /**
   * Optional relative times, one `float32` per row, which turn the ordering into ST-OPTICS: a pair
   * of rows is related only when `|t_a - t_b| <= maximumTimeDistance`, besides the spatial test
   * `distance <= maximumDistance`. Core distances, reachability and extraction use only related
   * pairs, so a pair outside the time window behaves as if it were farther than
   * `maximumDistance`. This is a hard filter on the neighbor graph, which equals the precomputed
   * matrix of `st-clustering` (distance `2 * eps1` outside the window) whenever
   * `maximumDistance < 2 * eps1`. A row with a non-finite time is invalid like a non-finite
   * position. Typically the output of `GPUTimeWordsRelative`. Requires `temporalContext` and
   * `maximumTimeDistance`. When absent, nothing changes. Compile-time presence; per-frame contents.
   */
  times?: GraphDataView<'float32'>;
  /**
   * Describes `times`: required with `times`, rejected without it. Only `'float32-relative'` times
   * are read; any unit, epoch (a bigint origin), time zone and axis. `unit` is the unit of
   * `maximumTimeDistance`. Nothing is converted.
   */
  temporalContext?: GPUTemporalContext;
  /** Largest time difference of related rows, in the `temporalContext` unit. Positive, finite. Compile-time. */
  maximumTimeDistance?: number;
  /** Includes the focus point, as in sklearn OPTICS. */
  minimumSamples: number;
  maximumDistance: number;
  /** DBSCAN extraction radius, positive and <= maximumDistance. Required only for DBSCAN extraction. */
  extractionDistance?: number;
  /** Default dbscan. Xi retains nested hierarchy and labels only disjoint innermost clusters. */
  extractionMethod?: 'dbscan' | 'xi';
  xi?: number;
  minimumClusterSize?: number;
  predecessorCorrection?: boolean;
  /** Maximum sequential visits in one dispatch, default 64. Every remaining batch is declared. */
  frontierBatchSize?: number;
  /** Maximum visited rows, default input length. Incomplete extraction labels stay noise. */
  maximumIterations?: number;
  output: GPUOPTICSOutput;
};

/**
 * Exact bounded OPTICS with deterministic (reachability, input-row) frontier ties.
 * Core distances come from a `GPUNeighborSearch` k-nearest self join capped at `maximumDistance`
 * (and the time window, with `times`); the resident frontier follows OPTICS' sequential dependency
 * and still performs O(n²) work, so this stays a small-frame reference path (at most 4096 rows), not
 * a claimed speedup over indexed CPU implementations. Graph scratch is O(n * minimumSamples).
 */
export class GPUOPTICS implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUOPTICSProps;
  readonly spatialContext: GPUSpatialContext;
  /** Resolved context of `times` (ST-OPTICS), else undefined. */
  readonly temporalContext?: GPUTemporalContext;

  constructor(props: GPUOPTICSProps) {
    this.props = props;
    this.id = props.id ?? 'optics';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext, {
      ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    const rows = props.positions.length;
    if (props.times) {
      validateRequirement(
        this.id,
        props.temporalContext !== undefined,
        'times need a temporalContext'
      );
      this.temporalContext = resolveGPUTemporalContext(this.id, props.temporalContext, {
        encodings: ['float32-relative']
      });
      validatePointProcessView(props.times, 'float32', rows);
      validatePointProcessPositive(props.maximumTimeDistance ?? NaN);
    } else {
      validateRequirement(
        this.id,
        props.temporalContext === undefined && props.maximumTimeDistance === undefined,
        'temporalContext and maximumTimeDistance need times'
      );
    }
    // Bounded frontier and rank searches must fit the supported small-frame envelope.
    validateCapacity(this.id, 'positions rows', rows, 1, 4096);
    validateCapacity(this.id, 'minimumSamples', props.minimumSamples, 1, 256);
    const iterations = props.maximumIterations ?? rows;
    validateCapacity(this.id, 'maximumIterations', iterations, 1, rows);
    validatePointProcessPositive(props.maximumDistance);
    validateRequirement(
      this.id,
      props.extractionMethod === undefined || ['dbscan', 'xi'].includes(props.extractionMethod),
      "extractionMethod must be 'dbscan' or 'xi'"
    );
    if ((props.extractionMethod ?? 'dbscan') === 'dbscan') {
      validatePointProcessPositive(props.extractionDistance ?? NaN);
      validateRequirement(
        this.id,
        props.extractionDistance! <= props.maximumDistance,
        'extractionDistance must be at most maximumDistance'
      );
    }
    const xi = props.xi ?? 0.05;
    validateRequirement(this.id, Number.isFinite(xi) && xi > 0 && xi < 1, 'xi must be in (0, 1)');
    const clusterSize = props.minimumClusterSize ?? props.minimumSamples;
    if (props.extractionMethod === 'xi') {
      validateCapacity(this.id, 'minimumClusterSize', clusterSize, 2, rows);
    }
    const batchSize = props.frontierBatchSize ?? 64;
    validateCapacity(this.id, 'frontierBatchSize', batchSize, 1, 256);
    if (props.output.clusterIntervals || props.output.clusterSummary) {
      validateRequirement(
        this.id,
        props.extractionMethod === 'xi',
        "clusterIntervals and clusterSummary need extractionMethod 'xi'"
      );
    }
    if (props.output.clusterIntervals)
      validatePointProcessView(props.output.clusterIntervals, 'uint32x2', rows);
    if (props.output.clusterSummary)
      validatePointProcessView(props.output.clusterSummary, 'uint32', 2);
    validatePointProcessView(props.positions, 'float32x2', rows);
    for (const field of [props.output.ordering, props.output.predecessors, props.output.labels]) {
      validatePointProcessView(field, 'uint32', rows);
    }
    for (const field of [props.output.coreDistances, props.output.reachability]) {
      validatePointProcessView(field, 'float32', rows);
    }
    validatePointProcessView(props.output.status, 'uint32', 3);
    validateGraphOutputsDisjointFromInputs(
      this.id,
      Object.values(props.output),
      [props.positions, props.times].filter(Boolean) as GraphDataView[]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, output, minimumSamples} = props;
    const rows = positions.length;
    const {times} = props;
    validateGraphViewsBelongToGraph(id, graph, [positions, times, ...Object.values(output)]);
    // Rows are related only inside the time window; without times every pair is.
    const timeBindings = times
      ? [{name: 'times', view: times, type: 'f32' as const, access: 'read' as const}]
      : [];
    const processed = createTransientView(graph, `${id}-processed`, 'uint32', rows);
    const geographic = this.spatialContext.coordinateSpace === 'longitude-latitude';
    const distanceSource = geographic
      ? `${GEODESIC_WGSL}
fn getPointDistance(first: vec2<f32>, second: vec2<f32>) -> f32 { return geodesicHaversineDistance(first, second, ${getWGSLFloatLiteral(getGPUSphereRadius(this.spatialContext))}); }`
      : 'fn getPointDistance(first: vec2<f32>, second: vec2<f32>) -> f32 { return distance(first, second); }';
    const declarations = `${IS_FINITE_FLOAT_WGSL}
${distanceSource}
const ROWS = ${rows}u;
const MAXIMUM_DISTANCE = ${Math.fround(props.maximumDistance)}f;
fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn validPoint(row: u32) -> bool {
  let value = point(row);
  return isFiniteFloat(value.x) && isFiniteFloat(value.y)${geographic ? ' && abs(value.x) <= 180.0 && abs(value.y) <= 90.0' : ''}${times ? ' && isFiniteFloat(times[timesOffset + row])' : ''};
}
${
  times
    ? `const MAXIMUM_TIME_DISTANCE = ${Math.fround(props.maximumTimeDistance!)}f;
fn isRelated(first: u32, second: u32) -> bool {
  return abs(times[timesOffset + first] - times[timesOffset + second]) <= MAXIMUM_TIME_DISTANCE;
}`
    : ''
}`;
    const otherSamples = minimumSamples - 1;
    const coreSearchNodes: GPUCommandNode<Parameters>[] = [];
    let neighborOffsets: GraphDataView<'uint32'> | undefined;
    let neighborDistances: GraphDataView<'float32'> | undefined;
    if (otherSamples > 0) {
      const search = createNeighborSearchDerivedParameterNodes<Parameters>(graph, {
        id: `${id}-core-neighbors-setup`,
        operation: 'GPUOPTICS',
        positions,
        radius: props.maximumDistance,
        geographic,
        times,
        timeRadius: props.maximumTimeDistance
      });
      const slotCount = rows * otherSamples;
      neighborOffsets = createTransientView(
        graph,
        `${id}-core-neighbor-offsets`,
        'uint32',
        rows + 1
      );
      neighborDistances = createTransientView(
        graph,
        `${id}-core-neighbor-distances`,
        'float32',
        slotCount
      );
      coreSearchNodes.push(
        ...search.nodes,
        ...new GPUNeighborSearch({
          id: `${id}-core-neighbors`,
          mode: 'knn',
          k: otherSamples,
          spatialContext: this.spatialContext,
          positions,
          times,
          temporalContext: this.temporalContext,
          timeBucketCount: times ? OPTICS_TIME_BUCKET_COUNT : undefined,
          parameters: search.parameters,
          gridSize: search.gridSize,
          output: {
            offsets: neighborOffsets,
            neighbors: createTransientView(graph, `${id}-core-neighbor-ids`, 'uint32', slotCount),
            weights: createTransientView(
              graph,
              `${id}-core-neighbor-weights`,
              'float32',
              slotCount
            ),
            distances: neighborDistances,
            status: {
              overflow: createTransientView(graph, `${id}-core-neighbor-overflow`, 'uint32', 1)
            }
          }
        }).getCommandNodes(graph)
      );
    }
    const nodes: GPUCommandNode<Parameters>[] = [
      ...coreSearchNodes,
      createWGSLKernelNode(graph, {
        id: `${id}-core`,
        operation: 'GPUOPTICS',
        variant: 'core',
        invocationCount: rows,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...timeBindings,
          ...(neighborOffsets && neighborDistances
            ? [
                {
                  name: 'neighborOffsets',
                  view: neighborOffsets,
                  type: 'u32' as const,
                  access: 'read' as const
                },
                {
                  name: 'neighborDistances',
                  view: neighborDistances,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'core', view: output.coreDistances, type: 'f32', access: 'read_write'}
        ],
        declarations,
        // The search lists the other related rows within maximumDistance; the row itself is the
        // first sample, so minimumSamples - 1 of them reach the self-inclusive core distance.
        body: `var coreDistance = bitcast<f32>(0x7f800000u | (index & 0u));
  if (validPoint(index)) {
    ${
      neighborOffsets
        ? `let begin = neighborOffsets[neighborOffsetsOffset + index];
    let end = neighborOffsets[neighborOffsetsOffset + index + 1u];
    if (end - begin + 1u >= ${minimumSamples}u) {
      coreDistance = 0.0;
      for (var slot = begin; slot < end; slot++) {
        coreDistance = max(coreDistance, neighborDistances[neighborDistancesOffset + slot]);
      }
    }`
        : 'coreDistance = 0.0;'
    }
  }
  core[coreOffset + index] = coreDistance;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: 'GPUOPTICS',
        variant: 'frontier',
        invocationCount: 256,
        workgroupSize: 256,
        guardIndex: false,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...timeBindings,
          {name: 'ordering', view: output.ordering, type: 'u32', access: 'read_write'},
          {name: 'reachability', view: output.reachability, type: 'f32', access: 'read_write'},
          {name: 'predecessors', view: output.predecessors, type: 'u32', access: 'read_write'},
          {name: 'processed', view: processed, type: 'u32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'}
        ],
        declarations: `${declarations}
var<workgroup> invalidCounts: array<u32, 256>;`,
        body: `let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  var invalidCount = 0u;
  for (var row = index; row < ROWS; row += 256u) {
    let invalid = !validPoint(row);
    invalidCount += u32(invalid);
    processed[processedOffset + row] = u32(invalid);
    ordering[orderingOffset + row] = 0xffffffffu;
    predecessors[predecessorsOffset + row] = 0xffffffffu;
    reachability[reachabilityOffset + row] = infinity;
    labels[labelsOffset + row] = 0xffffffffu;
  }
  invalidCounts[index] = invalidCount;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (index < stride) { invalidCounts[index] += invalidCounts[index + stride]; }
    workgroupBarrier();
  }
  if (index == 0u) {
    status[statusOffset] = 0u;
    status[statusOffset + 1u] = 0u;
    status[statusOffset + 2u] = invalidCounts[0];
  }`
      })
    ];
    for (
      let batch = 0;
      batch < Math.ceil((props.maximumIterations ?? rows) / (props.frontierBatchSize ?? 64));
      batch++
    ) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-frontier-${batch}`,
          operation: 'GPUOPTICS',
          variant: 'frontier-batch',
          invocationCount: 256,
          workgroupSize: 256,
          guardIndex: false,
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            ...timeBindings,
            {name: 'core', view: output.coreDistances, type: 'f32', access: 'read'},
            {name: 'ordering', view: output.ordering, type: 'u32', access: 'read_write'},
            {name: 'reachability', view: output.reachability, type: 'f32', access: 'read_write'},
            {name: 'predecessors', view: output.predecessors, type: 'u32', access: 'read_write'},
            {name: 'processed', view: processed, type: 'u32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
          ],
          declarations: `${declarations}
var<workgroup> candidateRows: array<u32, 256>;
var<workgroup> candidateDistances: array<f32, 256>;
var<workgroup> visited: u32;`,
          body: `let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  if (index == 0u) { visited = status[statusOffset]; }
  workgroupBarrier();
  for (var iteration = 0u; iteration < ${Math.min(props.frontierBatchSize ?? 64, (props.maximumIterations ?? rows) - batch * (props.frontierBatchSize ?? 64))}u; iteration++) {
    var selected = 0xffffffffu; var best = infinity;
    for (var row = index; row < ROWS; row += 256u) {
      let candidate = reachability[reachabilityOffset + row];
      if (processed[processedOffset + row] == 0u && (candidate < best || (candidate == best && row < selected))) {
        selected = row; best = candidate;
      }
    }
    candidateRows[index] = selected; candidateDistances[index] = best;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
      if (index < stride) {
        let otherDistance = candidateDistances[index + stride]; let otherRow = candidateRows[index + stride];
        if (otherDistance < candidateDistances[index] || (otherDistance == candidateDistances[index] && otherRow < candidateRows[index])) {
          candidateDistances[index] = otherDistance; candidateRows[index] = otherRow;
        }
      }
      workgroupBarrier();
    }
    let focus = workgroupUniformLoad(&candidateRows[0]);
    if (focus == 0xffffffffu) { break; }
    if (index == 0u) { ordering[orderingOffset + visited] = focus; visited++; processed[processedOffset + focus] = 1u; }
    storageBarrier(); workgroupBarrier();
    let coreDistance = core[coreOffset + focus];
    if (isFiniteFloat(coreDistance)) {
      for (var row = index; row < ROWS; row += 256u) {
        if (processed[processedOffset + row] != 0u) { continue; }
        ${times ? 'if (!isRelated(focus, row)) { continue; }\n        ' : ''}let pairDistance = getPointDistance(point(focus), point(row));
        if (pairDistance > MAXIMUM_DISTANCE) { continue; }
        let candidate = max(coreDistance, pairDistance);
        if (candidate < reachability[reachabilityOffset + row]) {
          reachability[reachabilityOffset + row] = candidate; predecessors[predecessorsOffset + row] = focus;
        }
      }
    }
    storageBarrier(); workgroupBarrier();
  }
  if (index == 0u) {
    status[statusOffset] = visited;
    status[statusOffset + 1u] = select(0u, 1u, visited + status[statusOffset + 2u] == ROWS);
  }`
        })
      );
    }
    if (props.extractionMethod === 'xi') {
      nodes.push(createOPTICSXiNode(graph, props, id));
      return nodes;
    }
    const clusterStarts = createTransientView(graph, `${id}-cluster-starts`, 'uint32', rows);
    const clusterCounts = createTransientView(graph, `${id}-cluster-counts`, 'uint32', rows);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-cluster-starts`,
        operation: 'GPUOPTICS',
        variant: 'dbscan-cluster-starts',
        invocationCount: rows,
        bindings: [
          {name: 'core', view: output.coreDistances, type: 'f32', access: 'read'},
          {name: 'ordering', view: output.ordering, type: 'u32', access: 'read'},
          {name: 'reachability', view: output.reachability, type: 'f32', access: 'read'},
          {name: 'status', view: output.status, type: 'u32', access: 'read'},
          {name: 'starts', view: clusterStarts, type: 'u32', access: 'read_write'}
        ],
        body: `var startsCluster = false;
  if (status[statusOffset + 1u] != 0u && index < status[statusOffset]) {
    let row = ordering[orderingOffset + index];
    startsCluster = reachability[reachabilityOffset + row] > ${Math.fround(props.extractionDistance!)}f && core[coreOffset + row] <= ${Math.fround(props.extractionDistance!)}f;
  }
  starts[startsOffset + index] = u32(startsCluster);`
      }),
      ...new GPUScan({
        id: `${id}-cluster-scan`,
        input: clusterStarts,
        output: clusterCounts,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-extract`,
        operation: 'GPUOPTICS',
        variant: 'dbscan-extraction',
        invocationCount: rows,
        bindings: [
          {name: 'starts', view: clusterStarts, type: 'u32', access: 'read'},
          {name: 'counts', view: clusterCounts, type: 'u32', access: 'read'},
          {name: 'ordering', view: output.ordering, type: 'u32', access: 'read'},
          {name: 'reachability', view: output.reachability, type: 'f32', access: 'read'},
          {name: 'status', view: output.status, type: 'u32', access: 'read'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'}
        ],
        body: `if (status[statusOffset + 1u] == 0u || index >= status[statusOffset]) { return; }
  let row = ordering[orderingOffset + index];
  if (starts[startsOffset + index] != 0u || reachability[reachabilityOffset + row] <= ${Math.fround(props.extractionDistance!)}f) {
    labels[labelsOffset + row] = counts[countsOffset + index] - 1u;
  }`
      })
    );
    return nodes;
  }
}
