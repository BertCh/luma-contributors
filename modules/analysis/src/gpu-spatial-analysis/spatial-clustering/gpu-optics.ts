// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
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
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createOPTICSXiNode} from './gpu-optics-xi';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validatePointProcessPositive,
  validatePointProcessView
} from '../point-process/point-process-utils';

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
 * Core distances are computed in parallel; the resident frontier follows OPTICS' sequential
 * dependency. Core selection evaluates each candidate distance once, maintaining a private
 * maximum heap: O(n² log(minimumSamples + 1)) work and O(minimumSamples) private storage per
 * invocation. The frontier still performs O(n²) work; graph scratch is O(n). Explicitly a small-frame
 * reference path (at most 4096 rows), not a claimed speedup over indexed CPU implementations.
 */
export class GPUOPTICS implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUOPTICSProps;
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUOPTICSProps) {
    this.props = props;
    this.id = props.id ?? 'optics';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext, {
      ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    const rows = props.positions.length;
    // Bounded frontier and rank searches must fit the supported small-frame envelope.
    assert(Number.isInteger(rows) && rows > 0 && rows <= 4096);
    assert(
      Number.isInteger(props.minimumSamples) &&
        props.minimumSamples > 0 &&
        props.minimumSamples <= 256
    );
    const iterations = props.maximumIterations ?? rows;
    assert(Number.isInteger(iterations) && iterations > 0 && iterations <= rows);
    validatePointProcessPositive(props.maximumDistance);
    assert(
      props.extractionMethod === undefined || ['dbscan', 'xi'].includes(props.extractionMethod)
    );
    if ((props.extractionMethod ?? 'dbscan') === 'dbscan') {
      validatePointProcessPositive(props.extractionDistance ?? NaN);
      assert(props.extractionDistance! <= props.maximumDistance);
    }
    assert(Number.isFinite(props.xi ?? 0.05) && (props.xi ?? 0.05) > 0 && (props.xi ?? 0.05) < 1);
    const clusterSize = props.minimumClusterSize ?? props.minimumSamples;
    if (props.extractionMethod === 'xi')
      assert(Number.isInteger(clusterSize) && clusterSize >= 2 && clusterSize <= rows);
    const batchSize = props.frontierBatchSize ?? 64;
    assert(Number.isInteger(batchSize) && batchSize >= 1 && batchSize <= 256);
    if (props.output.clusterIntervals || props.output.clusterSummary)
      assert(props.extractionMethod === 'xi');
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
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [props.positions]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, output, minimumSamples} = props;
    const rows = positions.length;
    validateGraphViewsBelongToGraph(id, graph, [positions, ...Object.values(output)]);
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
  return isFiniteFloat(value.x) && isFiniteFloat(value.y)${geographic ? ' && abs(value.x) <= 180.0 && abs(value.y) <= 90.0' : ''};
}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-core`,
        operation: 'GPUOPTICS',
        variant: 'core',
        invocationCount: rows,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'core', view: output.coreDistances, type: 'f32', access: 'read_write'}
        ],
        declarations: `${declarations}
var<private> nearestDistances: array<f32, ${minimumSamples}>;`,
        body: `let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  var selectedCount = 0u;
  if (validPoint(index)) {
    for (var row = 0u; row < ROWS; row++) {
      if (!validPoint(row)) { continue; }
      let candidateDistance = getPointDistance(point(index), point(row));
      if (!(candidateDistance <= MAXIMUM_DISTANCE)) { continue; }
      if (selectedCount < ${minimumSamples}u) {
        var slot = selectedCount;
        selectedCount++;
        loop {
          if (slot == 0u) { break; }
          let parent = (slot - 1u) / 2u;
          if (nearestDistances[parent] >= candidateDistance) { break; }
          nearestDistances[slot] = nearestDistances[parent];
          slot = parent;
        }
        nearestDistances[slot] = candidateDistance;
      } else if (candidateDistance < nearestDistances[0]) {
        var slot = 0u;
        loop {
          var child = slot * 2u + 1u;
          if (child >= selectedCount) { break; }
          if (child + 1u < selectedCount && nearestDistances[child + 1u] > nearestDistances[child]) { child++; }
          if (nearestDistances[child] <= candidateDistance) { break; }
          nearestDistances[slot] = nearestDistances[child];
          slot = child;
        }
        nearestDistances[slot] = candidateDistance;
      }
    }
  }
  // Equal distances count separately, as do their distinct source rows in the rank-order path.
  core[coreOffset + index] = select(infinity, nearestDistances[0], selectedCount == ${minimumSamples}u);`
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
        let pairDistance = getPointDistance(point(focus), point(row));
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
