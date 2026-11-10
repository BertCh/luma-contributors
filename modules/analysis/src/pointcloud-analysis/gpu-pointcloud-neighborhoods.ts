// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  validateViewLength
} from '../analysis-core/kernels/gpu-contributor-utils';
import {createPointCloudGridIndex, validatePointCloudSearchGrid} from './pointcloud-grid-index';
import {getPointCloudKernelNodes} from './pointcloud-kernel-nodes';
import {
  GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY,
  GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK
} from './pointcloud-limits';

export type GPUPointCloudNeighborhoodsProps = {
  id?: string;
  positions: GraphDataView<'float32x3'>;
  /** Optional owned query rows into positions; positions may include halo rows. Returned IDs remain local source rows. */
  queryRows?: GraphDataView<'uint32'>;
  /** Certified complete spatial extent of supplied points; status bit4 marks queries whose radius exceeds this extent. */
  supportBounds?: {
    minimum: readonly [number, number, number];
    maximum: readonly [number, number, number];
  };
  /** Inclusive Euclidean 3D search radius. */
  radius: number;
  /** Retain the nearest 1–64 other points; ties prefer smaller original IDs. */
  capacity: number;
  /** Cooperative workgroups target dense candidate sets; serial remains the default. */
  strategy?: 'serial' | 'cooperative';
  /** Row-major fixed-stride original IDs; missing slots contain 0xffffffff. */
  indices: GraphDataView<'uint32'>;
  squaredDistances: GraphDataView<'float32'>;
  /** Actual radius support, including candidates omitted by capacity. */
  counts: GraphDataView<'uint32'>;
  /**
   * Exact 3D grid acceleration, rebuilt each encoding. Points outside the half-open grid do not
   * participate. Required above `GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK` query-candidate pairs: without a grid
   * every query scans every point and the constructor refuses larger inputs.
   */
  grid?: {
    origin: readonly [number, number, number];
    cellSize: number;
    dimensions: readonly [number, number, number];
  };
  /**
   * Optional per-query bit set, zero when clean. Bit 1 (`GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_INVALID_QUERY`):
   * the query row is out of range, not finite, or outside `grid`; its list is empty. Bit 2
   * (`..._TRUNCATED`): more than `capacity` points lie within `radius`; the nearest `capacity` are
   * kept and `counts` holds the true support. Bit 4 (`..._UNSUPPORTED`): only with `supportBounds`,
   * the query's search sphere reaches outside the certified bounds, so the support may be
   * incomplete. Bits above 4 are zero.
   */
  status?: GraphDataView<'uint32'>;
};

/** `GPUPointCloudNeighborhoods` status bit 1: invalid, out-of-range or outside-grid query. */
export const GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_INVALID_QUERY = 1;
/** `GPUPointCloudNeighborhoods` status bit 2: more points within radius than `capacity`. */
export const GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_TRUNCATED = 2;
/** `GPUPointCloudNeighborhoods` status bit 4: the search radius exceeds `supportBounds`. */
export const GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_UNSUPPORTED = 4;

/**
 * Exact 3D neighborhoods with observable truncation. With a `grid` the search visits only the
 * cells the radius can reach; without one every query scans every point, which the constructor
 * allows only up to `GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK` queries times points.
 */
export class GPUPointCloudNeighborhoods implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUPointCloudNeighborhoodsProps) {
    this.id = props.id ?? 'gpu-pointcloud-neighborhoods';
    validateRequirement(
      this.id,
      props.strategy === undefined ||
        props.strategy === 'serial' ||
        props.strategy === 'cooperative',
      "strategy must be 'serial' or 'cooperative'"
    );
    validateCapacity(
      this.id,
      'capacity',
      props.capacity,
      1,
      GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY
    );
    validateFiniteNumber(this.id, 'radius', props.radius);
    validateRequirement(
      this.id,
      props.radius > 0 &&
        Math.fround(props.radius * props.radius) > 0 &&
        Number.isFinite(Math.fround(props.radius * props.radius)),
      'radius must be positive with a finite nonzero float32 square'
    );
    const queryCount = props.queryRows?.length ?? props.positions.length;
    validateViewLength(this.id, 'indices', props.indices, queryCount * props.capacity);
    validateViewLength(this.id, 'squaredDistances', props.squaredDistances, props.indices.length);
    validateViewLength(this.id, 'counts', props.counts, queryCount);
    if (props.status) {
      validateViewLength(this.id, 'status', props.status, queryCount);
    }
    if (props.supportBounds) {
      validateRequirement(this.id, Boolean(props.status), 'supportBounds requires a status view');
      validateRequirement(
        this.id,
        props.supportBounds.minimum.every(
          (value, axis) =>
            Number.isFinite(Math.fround(value)) && value < props.supportBounds!.maximum[axis]
        ) && props.supportBounds.maximum.every(value => Number.isFinite(Math.fround(value))),
        'supportBounds must hold finite minimum values below the maximum values'
      );
    }
    if (props.grid) {
      validatePointCloudSearchGrid(this.id, 'grid', props.grid);
      validateRequirement(
        this.id,
        Number.isFinite(Math.fround(props.radius / props.grid.cellSize)),
        'radius divided by grid.cellSize must be finite in float32'
      );
    } else {
      // Brute force is never the silent default at scale (contributor contract rule 3).
      validateRequirement(
        this.id,
        queryCount * props.positions.length <= GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK,
        `queries x positions is ${queryCount * props.positions.length}, above the ${GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK} limit of a search without an index; pass grid`
      );
    }
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {positions, indices, squaredDistances, counts, radius, capacity, grid, status} =
      this.props;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const cooperative = this.props.strategy === 'cooperative';
    // At most 8 KiB for sorted pairs plus one count per lane, independent of input length.
    const workgroupSize = Math.min(128, 2 ** Math.floor(Math.log2(1024 / capacity)));
    let gridInputs: Record<string, GraphDataView<'uint32'>> = {};
    let querySetup = '';
    let validQuery = 'all(abs(position) <= vec3f(3.402823e38))';
    let startCandidates = cooperative
      ? `for (var candidateRound = 0u; candidateRound < ${Math.ceil(positions.length / workgroupSize)}u; candidateRound++) {
    let candidate = candidateRound * ${workgroupSize}u + localInvocationIndex;
    if (candidate >= ${positions.length}u) { continue; }`
      : `for (var candidate = 0u; candidate < ${positions.length}u; candidate++) {`;
    let endCandidates = '}';
    if (grid) {
      const built = createPointCloudGridIndex(graph, `${this.id}-grid`, positions, grid);
      nodes.push(...built.nodes);
      gridInputs = {sortedIndices: built.sortedIndices, cellOffsets: built.offsets};
      querySetup = `let coordinate=${built.coordinate};`;
      validQuery += ` && (${built.inside})`;
      // An extra cell protects inclusive radius boundaries from float32 bin division rounding.
      const reach = Math.min(Math.ceil(radius / grid.cellSize) + 1, Math.max(...grid.dimensions));
      startCandidates = `let center=vec3i(floor(coordinate));
let lower=max(center-vec3i(${reach}),vec3i(0));
let upper=min(center+vec3i(${reach}),vec3i(${grid.dimensions.map(value => value - 1).join(',')}));
for (var depth=lower.z; depth<=upper.z; depth++) {
 for (var row=lower.y; row<=upper.y; row++) {
  for (var column=lower.x; column<=upper.x; column++) {
   let cell=(u32(depth)*${grid.dimensions[1]}u+u32(row))*${grid.dimensions[0]}u+u32(column);
   ${
     cooperative
       ? `let cellBegin = readCellOffsets(cell);
   let cellLength = readCellOffsets(cell + 1u) - cellBegin;
   let candidateRounds = cellLength / ${workgroupSize}u + u32(cellLength % ${workgroupSize}u != 0u);
   for (var candidateRound = 0u; candidateRound < candidateRounds; candidateRound++) {
    let relativeSlot = candidateRound * ${workgroupSize}u + localInvocationIndex;
    if (relativeSlot >= cellLength) { continue; }
    let slotIndex = cellBegin + relativeSlot;`
       : 'for (var slotIndex=readCellOffsets(cell); slotIndex<readCellOffsets(cell+1u); slotIndex++) {'
   }
    let candidate=readSortedIndices(slotIndex);`;
      endCandidates = '} } } }';
    }
    nodes.push(
      ...getPointCloudKernelNodes(graph, {
        id: this.id,
        length: counts.length,
        ...(cooperative
          ? {
              cooperative: {
                workgroupSize,
                declarations: `var<workgroup> retainedIds: array<u32, ${workgroupSize * capacity}>;
var<workgroup> retainedDistances: array<f32, ${workgroupSize * capacity}>;
var<workgroup> supportCounts: array<u32, ${workgroupSize}>;`
              }
            }
          : {}),
        inputs: {
          positions,
          ...gridInputs,
          ...(this.props.queryRows ? {queryRows: this.props.queryRows} : {})
        },
        outputs: {indices, squaredDistances, counts, ...(status ? {status} : {})},
        body: `
var nearestIds: array<u32, ${capacity}>;
var nearestDistances: array<f32, ${capacity}>;
for (var slot = 0u; slot < ${capacity}u; slot++) {
  nearestIds[slot] = 0xffffffffu; nearestDistances[slot] = 3.402823e38;
}
let queryRow = ${this.props.queryRows ? 'readQueryRows(index)' : 'index'};
let position = readPositions(min(queryRow,${Math.max(positions.length - 1, 0)}u));
var count = 0u;
${querySetup}
let validQuery = queryRow < ${positions.length}u && (${validQuery});
if (validQuery) {
  ${startCandidates}
    if (candidate == queryRow) { continue; }
    let difference = readPositions(candidate) - position;
    let distanceSquared = dot(difference, difference);
    if (!(distanceSquared <= ${Math.fround(radius * radius)})) { continue; }
    count++;
    ${
      cooperative
        ? `// Count every radius match before rejecting a noncompetitive candidate.
    if (distanceSquared > nearestDistances[${capacity - 1}u] ||
        (distanceSquared == nearestDistances[${capacity - 1}u] && candidate >= nearestIds[${capacity - 1}u])) { continue; }`
        : ''
    }
    var insertion = ${capacity}u;
    for (var slot = 0u; slot < ${capacity}u; slot++) {
      if (distanceSquared < nearestDistances[slot] || (distanceSquared == nearestDistances[slot] && candidate < nearestIds[slot])) {
        insertion = slot; break;
      }
    }
    if (insertion < ${capacity}u) {
      for (var slot = ${capacity - 1}u; slot > insertion; slot--) {
        nearestIds[slot] = nearestIds[slot - 1u]; nearestDistances[slot] = nearestDistances[slot - 1u];
      }
      nearestIds[insertion] = candidate; nearestDistances[insertion] = distanceSquared;
    }
  ${endCandidates}
}
${
  cooperative
    ? `
for (var slot = 0u; slot < ${capacity}u; slot++) {
  retainedIds[localInvocationIndex * ${capacity}u + slot] = nearestIds[slot];
  retainedDistances[localInvocationIndex * ${capacity}u + slot] = nearestDistances[slot];
}
supportCounts[localInvocationIndex] = count;
workgroupBarrier();
for (var mergeStride = 1u; mergeStride < ${workgroupSize}u; mergeStride *= 2u) {
  if (localInvocationIndex % (mergeStride * 2u) == 0u) {
    let firstStart = localInvocationIndex * ${capacity}u;
    let secondStart = (localInvocationIndex + mergeStride) * ${capacity}u;
    var firstSlot = 0u;
    var secondSlot = 0u;
    // Read complete inputs into lane-private output before writing shared storage in place.
    for (var slot = 0u; slot < ${capacity}u; slot++) {
      let firstDistance = retainedDistances[firstStart + min(firstSlot, ${capacity - 1}u)];
      let firstId = retainedIds[firstStart + min(firstSlot, ${capacity - 1}u)];
      let secondDistance = retainedDistances[secondStart + min(secondSlot, ${capacity - 1}u)];
      let secondId = retainedIds[secondStart + min(secondSlot, ${capacity - 1}u)];
      let takeFirst = secondSlot == ${capacity}u ||
        (firstSlot < ${capacity}u && (firstDistance < secondDistance ||
          (firstDistance == secondDistance && firstId <= secondId)));
      nearestIds[slot] = select(secondId, firstId, takeFirst);
      nearestDistances[slot] = select(secondDistance, firstDistance, takeFirst);
      firstSlot += u32(takeFirst);
      secondSlot += u32(!takeFirst);
    }
    for (var slot = 0u; slot < ${capacity}u; slot++) {
      retainedIds[firstStart + slot] = nearestIds[slot];
      retainedDistances[firstStart + slot] = nearestDistances[slot];
    }
    supportCounts[localInvocationIndex] += supportCounts[localInvocationIndex + mergeStride];
  }
  workgroupBarrier();
}
if (localInvocationIndex != 0u) { return; }
count = supportCounts[0];
`
    : ''
}
writeCounts(index, count);
${status ? `writeStatus(index,select(${GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_INVALID_QUERY}u,0u,validQuery)|select(0u,${GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_TRUNCATED}u,count>${capacity}u)${this.props.supportBounds ? `|select(${GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_UNSUPPORTED}u,0u,all(position-vec3f(${Math.fround(radius)})>=vec3f(${this.props.supportBounds.minimum.map(Math.fround).join(',')}))&&all(position+vec3f(${Math.fround(radius)})<=vec3f(${this.props.supportBounds.maximum.map(Math.fround).join(',')})))` : ''});` : ''}
for (var slot = 0u; slot < ${capacity}u; slot++) {
  writeIndices(index * ${capacity}u + slot, nearestIds[slot]);
  writeSquaredDistances(index * ${capacity}u + slot, nearestDistances[slot]);
}`
      })
    );
    return nodes;
  }
}
