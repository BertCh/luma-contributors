// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getPointCloudReadbackLaneReader} from './pointcloud-readback';
import {POINT_CLOUD_NORM_COMPARISON_WGSL} from './pointcloud-norm-comparison-wgsl';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import {createTransientView} from '@luma.gl/graph/gpu-core';
import {fp64arithmetic} from '@luma.gl/shaders';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateGraphPorts,
  validateSinglePackedView,
  validateFiniteNumber,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createPointCloudGridIndex,
  validatePointCloudSearchGrid,
  validatePointCloudGridIndexBindings
} from './pointcloud-grid-index';
import {
  getPointCloudKernelNodes,
  validatePointCloudPackedViews,
  validatePointCloudScratchBindings
} from './pointcloud-kernel-nodes';
import {
  GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY,
  GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK
} from './pointcloud-limits';

import {
  GRAPH_CHUNK_MAXIMUM_ROW_COUNT,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {validatePointCloudGridParameters} from './pointcloud-grid-parameters';
import {inferPointCloudNeighborCapacity} from './pointcloud-neighbor-capacity';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';

// Radius and its float32 square, six support bounds and a support-bounds-enabled flag.
export const GPU_POINT_CLOUD_NEIGHBORHOODS_PARAMETER_LENGTH = 9;

export type GPUPointCloudNeighborhoodsSettings = {
  radius: number;
  supportBounds?: GPUPointCloudNeighborhoodsProps['supportBounds'];
};

/** Pack runtime radius and certified support bounds; grid/capacity/strategy stay compile-time. */
export function getGPUPointCloudNeighborhoodsParameterValues(
  settings: GPUPointCloudNeighborhoodsSettings
): Float32Array {
  validatePointCloudNeighborhoodsSettings('gpu-pointcloud-neighborhoods', settings);
  return new Float32Array([
    settings.radius,
    settings.radius ** 2,
    ...(settings.supportBounds?.minimum ?? [0, 0, 0]),
    ...(settings.supportBounds?.maximum ?? [0, 0, 0]),
    settings.supportBounds ? 1 : 0
  ]);
}

function validatePointCloudNeighborhoodsSettings(
  identifier: string,
  settings: GPUPointCloudNeighborhoodsSettings
): void {
  validateFiniteNumber(identifier, 'radius', settings.radius);
  validateRequirement(
    identifier,
    settings.radius > 0 &&
      Math.fround(settings.radius ** 2) > 0 &&
      Number.isFinite(Math.fround(settings.radius ** 2)),
    'radius must be positive with a finite nonzero float32 square'
  );
  if (settings.supportBounds)
    validateRequirement(
      identifier,
      settings.supportBounds.minimum.every(
        (value, axis) =>
          Number.isFinite(Math.fround(value)) &&
          Math.fround(value) < Math.fround(settings.supportBounds!.maximum[axis])
      ) && settings.supportBounds.maximum.every(value => Number.isFinite(Math.fround(value))),
      'supportBounds must hold finite float32 minimum values below the maximum values'
    );
}

export type GPUPointCloudNeighborhoodsOutput = {
  /** Query-major fixed-stride source-row indices into positions; missing slots contain 0xffffffff. */
  indices: GraphDataView<'uint32'>;
  /** Query-major squared Euclidean distances in squared coordinate units. Read min(counts[query], capacity) slots; missing slots hold float32 bits 0x7f7fffff. */
  squaredDistances: GraphDataView<'float32'>;
  /** Actual radius support, including candidates omitted by capacity. */
  counts: GraphDataView<'uint32'>;
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

export type GPUPointCloudNeighborhoodsProps = {
  /** Runtime grid origin/cellSize; grid dimensions stay fixed. Requires grid. */
  gridParameters?: GraphDataView<'float32'>;
  /** Runtime radius/supportBounds, packed by getGPUPointCloudNeighborhoodsParameterValues. Requires `output.status`. */
  parameters?: GraphDataView<'float32'>;
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
  /** Retain the nearest 1–64 other points; exact norm ties prefer smaller original IDs. Omitted infers exact output slots per query; zero queries require an explicit capacity. */
  capacity?: number;
  /** Cooperative workgroups target dense candidate sets; serial remains the default. */
  strategy?: 'serial' | 'cooperative';
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
  /** Caller-owned result views; their contents are written by the graph. */
  output: GPUPointCloudNeighborhoodsOutput;
};

/** `GPUPointCloudNeighborhoods` status bit 1: invalid, out-of-range or outside-grid query. */
export const GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_INVALID_QUERY = 1;
/** `GPUPointCloudNeighborhoods` status bit 2: more points within radius than `capacity`. */
export const GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_TRUNCATED = 2;
/** `GPUPointCloudNeighborhoods` status bit 4: the search radius exceeds `supportBounds`. */
export const GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_UNSUPPORTED = 4;

/** Decode one query's neighborhood status; pass a one-word subview for a later query. */
export function decodeGPUPointCloudNeighborhoodsStatus(
  values: Uint32Array | Uint8Array | ArrayBuffer
): {
  invalidInput: boolean;
  truncated: boolean;
  incompleteSupport: boolean;
  complete: boolean;
} {
  const flags = getPointCloudReadbackLaneReader(
    'decodeGPUPointCloudNeighborhoodsStatus',
    'status',
    values,
    'uint32',
    1
  )(0);
  return {
    invalidInput: (flags & GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_INVALID_QUERY) !== 0,
    truncated: (flags & GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_TRUNCATED) !== 0,
    incompleteSupport: (flags & GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_UNSUPPORTED) !== 0,
    complete: (flags & 7) === 0
  };
}

/**
 * Exact 3D neighborhoods with observable truncation. Per-frame: parameters (radius/supportBounds)
 * and input/gridParameters contents. Compile-time: grid dimensions, capacity, strategy and view shapes.
 * Capacity is 1–64 retained slots per query (GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY), across serial and cooperative routes.
 * With a `grid` the search visits only the
 * cells the radius can reach; without one every query scans every point, which the constructor
 * allows only up to `GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK` queries times points.
 */
export class GPUPointCloudNeighborhoods implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUPointCloudNeighborhoodsProps & {capacity: number};
  constructor(props: GPUPointCloudNeighborhoodsProps) {
    props = snapshotConstructionFields(props, [
      'gridParameters',
      'parameters',
      'id',
      'positions',
      'queryRows',
      'supportBounds',
      'radius',
      'capacity',
      'strategy',
      'grid',
      'output'
    ]);
    this.id = props.id ?? 'gpu-pointcloud-neighborhoods';
    validateGraphPorts(this.id, props, {
      positions: {required: true, viewType: 'data', formats: ['float32x3']},
      queryRows: {required: false, viewType: 'data', formats: ['uint32']},
      'output.indices': {required: true, viewType: 'data', formats: ['uint32']},
      'output.squaredDistances': {required: true, viewType: 'data', formats: ['float32']},
      'output.counts': {required: true, viewType: 'data', formats: ['uint32']},
      'output.status': {required: false, viewType: 'data', formats: ['uint32']}
    });
    validateRequirement(
      this.id,
      props.output !== undefined && props.output !== null,
      'output must be given'
    );
    const queryCount = props.queryRows?.length ?? props.positions.length;
    const capacity =
      props.capacity ??
      inferPointCloudNeighborCapacity(this.id, queryCount, props.output.indices.length);
    this.props = Object.freeze({
      ...props,
      capacity,
      output: snapshotConstructionFields(props.output, [
        'indices',
        'squaredDistances',
        'counts',
        'status'
      ])
    });
    props = this.props;
    validatePointCloudGridParameters(this.id, props.gridParameters);
    validateRequirement(
      this.id,
      !props.gridParameters || Boolean(props.grid),
      'gridParameters requires grid'
    );
    validateRequirement(
      this.id,
      props.strategy === undefined ||
        props.strategy === 'serial' ||
        props.strategy === 'cooperative',
      "strategy must be 'serial' or 'cooperative'"
    );
    validateCapacity(this.id, 'capacity', capacity, 1, GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY);
    validatePointCloudNeighborhoodsSettings(this.id, props);
    for (const [name, view, format] of [
      ['positions', props.positions, 'float32x3'],
      ['queryRows', props.queryRows, 'uint32'],
      ['indices', props.output.indices, 'uint32'],
      ['squaredDistances', props.output.squaredDistances, 'float32'],
      ['counts', props.output.counts, 'uint32'],
      ['status', props.output.status, 'uint32'],
      ['parameters', props.parameters, 'float32']
    ] as const)
      if (view) validateSinglePackedView(view, [format], `${this.id} ${name}`);
    if (props.parameters)
      validateViewLength(
        this.id,
        'parameters',
        props.parameters,
        GPU_POINT_CLOUD_NEIGHBORHOODS_PARAMETER_LENGTH
      );
    const inputs = {
      positions: props.positions,
      queryRows: props.queryRows,
      parameters: props.parameters,
      gridParameters: props.gridParameters
    };
    const outputs = {
      indices: props.output.indices,
      squaredDistances: props.output.squaredDistances,
      counts: props.output.counts,
      status: props.output.status
    };
    validatePointCloudPackedViews(
      this.id,
      inputs,
      outputs,
      props.parameters || props.gridParameters
        ? [
            {
              inputs: props.gridParameters ? {} : inputs,
              outputs: {
                indices: props.output.indices,
                squaredDistances: props.output.squaredDistances,
                counts: props.output.counts
              }
            },
            ...(props.output.status
              ? [
                  {
                    inputs: {...inputs, counts: props.output.counts},
                    outputs: {status: props.output.status}
                  }
                ]
              : [])
          ]
        : [{inputs, outputs}]
    );
    if (props.gridParameters)
      validateCapacity(
        this.id,
        'runtime-grid packed neighbor words',
        queryCount * (2 * capacity + 1),
        0,
        GRAPH_CHUNK_MAXIMUM_ROW_COUNT
      );
    validateViewLength(this.id, 'indices', props.output.indices, queryCount * capacity);
    validateViewLength(
      this.id,
      'squaredDistances',
      props.output.squaredDistances,
      props.output.indices.length
    );
    validateViewLength(this.id, 'counts', props.output.counts, queryCount);
    if (props.output.status) {
      validateViewLength(this.id, 'status', props.output.status, queryCount);
    }
    if (props.supportBounds) {
      validateRequirement(
        this.id,
        Boolean(props.output.status),
        'supportBounds requires a status view'
      );
    }
    // Runtime parameters can enable supportBounds on any frame, so they need the same destination
    // for status bit 4 as static supportBounds.
    validateRequirement(
      this.id,
      !props.parameters || Boolean(props.output.status),
      'parameters requires a status view (runtime supportBounds report status bit 4 there)'
    );
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
    const {
      positions,
      radius,
      capacity,
      grid,
      output: {indices, squaredDistances, counts, status}
    } = this.props;
    const parameters = this.props.parameters;
    const gridParameters = this.props.gridParameters;
    const deferredStatus = Boolean(parameters || gridParameters);
    validateGraphChunkBindingRanges(graph, this.id, {
      positions,
      indices,
      squaredDistances,
      counts,
      status,
      parameters,
      gridParameters,
      queryRows: this.props.queryRows
    });
    if (grid)
      validatePointCloudGridIndexBindings(
        graph,
        this.id,
        positions,
        grid.dimensions,
        gridParameters
      );
    if (gridParameters)
      validatePointCloudScratchBindings(graph, this.id, {
        'runtime-grid neighborhood scratch': counts.length * (2 * capacity + 1) * 4
      });
    const radiusValue = parameters ? 'readParameters(0u)' : getWGSLFloatLiteral(radius);
    const radiusSquared = parameters ? 'readParameters(1u)' : getWGSLFloatLiteral(radius * radius);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const packedNeighbors = gridParameters
      ? createTransientView(
          graph,
          `${this.id}-packed-neighbors`,
          'uint32',
          counts.length * (2 * capacity + 1)
        )
      : undefined;
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
      const built = createPointCloudGridIndex(
        graph,
        `${this.id}-grid`,
        positions,
        grid,
        false,
        'north',
        gridParameters
      );
      nodes.push(...built.nodes);
      gridInputs = {sortedIndices: built.sortedIndices, cellOffsets: built.offsets};
      querySetup = `let coordinate=${built.coordinate};`;
      validQuery += ` && (${built.inside})`;
      // An extra cell protects inclusive radius boundaries from float32 bin division rounding.
      const reach =
        parameters || gridParameters
          ? `i32(min(ceil(getScaledQuotient(${radiusValue},${gridParameters ? 'readGridParameters(3u)' : Math.fround(grid.cellSize)},1.0,0))+1.0,${Math.max(...grid.dimensions)}.0))`
          : Math.min(Math.ceil(radius / grid.cellSize) + 1, Math.max(...grid.dimensions));
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
          ...(parameters ? {parameters} : {}),
          ...(gridParameters ? {gridParameters} : {}),
          ...gridInputs,
          ...(this.props.queryRows ? {queryRows: this.props.queryRows} : {})
        },
        outputs: packedNeighbors
          ? {packedNeighbors}
          : {indices, squaredDistances, counts, ...(!deferredStatus && status ? {status} : {})},
        modules: [fp64arithmetic],
        declarations: `${getPointCloudNeighborComparisonWGSL()}
${!deferredStatus && this.props.supportBounds ? getPointCloudSupportWGSL() : ''}
${
  packedNeighbors
    ? `
fn writeIndices(row: u32, value: u32) { writePackedNeighbors((row / ${capacity}u) * ${2 * capacity + 1}u + row % ${capacity}u, value); }
fn writeSquaredDistances(row: u32, value: f32) { writePackedNeighbors((row / ${capacity}u) * ${2 * capacity + 1}u + ${capacity}u + row % ${capacity}u, bitcast<u32>(value)); }
fn writeCounts(row: u32, value: u32) { writePackedNeighbors(row * ${2 * capacity + 1}u + ${2 * capacity}u, value); }`
    : ''
}`,
        body: `
var nearestIds: array<u32, ${capacity}>;
var nearestDistances: array<f32, ${capacity}>;
for (var slot = 0u; slot < ${capacity}u; slot++) {
  nearestIds[slot] = 0xffffffffu; nearestDistances[slot] = bitcast<f32>(0x7f7fffffu);
}
let queryRow = ${this.props.queryRows ? 'readQueryRows(index)' : 'index'};
let position = readPositions(min(queryRow,${Math.max(positions.length - 1, 0)}u));
var count = 0u;
${querySetup}
let validQuery = queryRow < ${positions.length}u && (${validQuery});
if (validQuery) {
  ${startCandidates}
    if (candidate == queryRow) { continue; }
    let candidatePosition = readPositions(candidate);
    if (!all((bitcast<vec3u>(candidatePosition) & vec3u(0x7fffffffu)) < vec3u(0x7f800000u))) { continue; }
    let difference = candidatePosition - position;
    // An overflowing coordinate difference is larger than every legal finite-radius search.
    if (!all((bitcast<vec3u>(difference) & vec3u(0x7fffffffu)) < vec3u(0x7f800000u))) { continue; }
    let distanceSquared = dot(difference, difference);
    if (pointCloudNormKeysSeparated(distanceSquared, ${radiusSquared})) {
      if (distanceSquared > ${radiusSquared}) { continue; }
    } else if (comparePointCloudNorms(candidatePosition, position, vec3f(${radiusValue}, 0.0, 0.0), vec3f(0.0)) > 0) { continue; }
    count++;
    ${
      cooperative
        ? `// Count every radius match before rejecting a noncompetitive candidate.
    if (!pointCloudNeighborBefore(candidate, distanceSquared, nearestIds[${capacity - 1}u], nearestDistances[${capacity - 1}u], position)) { continue; }`
        : ''
    }
    var insertion = ${capacity}u;
    for (var slot = 0u; slot < ${capacity}u; slot++) {
      if (pointCloudNeighborBefore(candidate, distanceSquared, nearestIds[slot], nearestDistances[slot], position)) {
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
        (firstSlot < ${capacity}u && (firstId == secondId || pointCloudNeighborBefore(firstId, firstDistance, secondId, secondDistance, position)));
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
${
  !deferredStatus && status
    ? `if (queryRow >= ${positions.length}u) { writeStatus(index, ${GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_INVALID_QUERY}u); }
else { writeStatus(index,select(${GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_INVALID_QUERY}u,0u,validQuery)|select(0u,${GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_TRUNCATED}u,count>${capacity}u)${this.props.supportBounds ? `|select(${GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_UNSUPPORTED}u,0u,pointCloudSupportContainsSphere(position, vec3f(${this.props.supportBounds.minimum.map(getWGSLFloatLiteral).join(',')}), vec3f(${this.props.supportBounds.maximum.map(getWGSLFloatLiteral).join(',')}), ${radiusValue}))` : ''}); }`
    : ''
}
for (var slot = 0u; slot < ${capacity}u; slot++) {
  writeIndices(index * ${capacity}u + slot, nearestIds[slot]);
  writeSquaredDistances(index * ${capacity}u + slot, nearestDistances[slot]);
}`
      })
    );
    if (packedNeighbors)
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-unpack`,
          length: counts.length,
          inputs: {packedNeighbors},
          outputs: {indices, squaredDistances, counts},
          body: `let base = index * ${2 * capacity + 1}u;
writeCounts(index, readPackedNeighbors(base + ${2 * capacity}u));
for (var slot = 0u; slot < ${capacity}u; slot++) {
  writeIndices(index * ${capacity}u + slot, readPackedNeighbors(base + slot));
  writeSquaredDistances(index * ${capacity}u + slot, bitcast<f32>(readPackedNeighbors(base + ${capacity}u + slot)));
}`
        })
      );
    if (deferredStatus && status) {
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-status`,
          length: counts.length,
          inputs: {
            positions,
            counts,
            ...(parameters ? {parameters} : {}),
            ...(gridParameters ? {gridParameters} : {}),
            ...(this.props.queryRows ? {queryRows: this.props.queryRows} : {})
          },
          outputs: {status},
          modules: [fp64arithmetic],
          declarations: getPointCloudSupportWGSL(),
          body: `let queryRow = ${this.props.queryRows ? 'readQueryRows(index)' : 'index'};
let position = readPositions(min(queryRow,${Math.max(positions.length - 1, 0)}u));
${querySetup}
let validQuery = queryRow < ${positions.length}u && (${validQuery});
if (queryRow >= ${positions.length}u) { writeStatus(index, ${GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_INVALID_QUERY}u); return; }
${
  parameters
    ? `let lower = vec3f(readParameters(2u), readParameters(3u), readParameters(4u));
let upper = vec3f(readParameters(5u), readParameters(6u), readParameters(7u));
let supported = readParameters(8u) == 0.0 || pointCloudSupportContainsSphere(position, lower, upper, ${radiusValue});`
    : `let supported = ${this.props.supportBounds ? `pointCloudSupportContainsSphere(position, vec3f(${this.props.supportBounds.minimum.map(getWGSLFloatLiteral).join(',')}), vec3f(${this.props.supportBounds.maximum.map(getWGSLFloatLiteral).join(',')}), ${radiusValue})` : 'true'};`
}
writeStatus(index, select(${GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_INVALID_QUERY}u, 0u, validQuery) | select(0u, ${GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_TRUNCATED}u, readCounts(index) > ${capacity}u) | select(${GPU_POINT_CLOUD_NEIGHBORHOODS_STATUS_UNSUPPORTED}u, 0u, supported));`
        })
      );
    }
    return nodes;
  }
}

/** Higher precision resolves overlapping float32 rounding-error intervals. Scaling by an exact power of two keeps
 * squared norms finite, including the maximum legal radius. Published integer fp64 arithmetic
 * prevents Metal reassociation from erasing the low words that decide membership and order.
 */
function getPointCloudNeighborComparisonWGSL(): string {
  return `
${POINT_CLOUD_NORM_COMPARISON_WGSL}
fn pointCloudNeighborBefore(firstId: u32, firstDistance: f32, secondId: u32, secondDistance: f32, query: vec3f) -> bool {
  if (firstId == 0xffffffffu) { return false; }
  if (secondId == 0xffffffffu) { return true; }
  if (pointCloudNormKeysSeparated(firstDistance, secondDistance)) { return firstDistance < secondDistance; }
  let comparison = comparePointCloudNorms(readPositions(firstId), query, readPositions(secondId), query);
  return comparison < 0 || (comparison == 0 && firstId < secondId);
}`;
}

/** Compare radius to exact coordinate margins; rounded position +/- radius can hide halo loss. */
function getPointCloudSupportWGSL(): string {
  return `
fn pointCloudMarginContainsRadius(maximum: f32, minimum: f32, radius: f32) -> bool {
  let margin = maximum - minimum;
  // This separated case also handles a positive overflowing margin without fp64 overflow.
  if (margin >= radius * 2.0) { return true; }
  if (margin < 0.0) { return false; }
  let exactMargin = sub_fp64(vec2f(maximum, 0.0), vec2f(minimum, 0.0));
  return exactMargin.x > radius || (exactMargin.x == radius && exactMargin.y >= 0.0);
}
fn pointCloudSupportContainsSphere(position: vec3f, lower: vec3f, upper: vec3f, radius: f32) -> bool {
  for (var axis = 0u; axis < 3u; axis++) {
    if (!pointCloudMarginContainsRadius(position[axis], lower[axis], radius) ||
        !pointCloudMarginContainsRadius(upper[axis], position[axis], radius)) { return false; }
  }
  return true;
}`;
}
