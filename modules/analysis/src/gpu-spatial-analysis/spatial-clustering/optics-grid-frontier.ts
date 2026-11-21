// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPUNeighborSearch} from '../neighbor-search/gpu-neighbor-search';
import {createNeighborSearchDerivedParameterNodes} from '../neighbor-search/neighbor-search-derived-parameters';
import {createOPTICSParallelFrontierNodes} from './gpu-optics-parallel-frontier';
import {GPU_OPTICS_DEFAULT_FRONTIER_BATCH_SIZE, type GPUOPTICS} from './gpu-optics';
const OPTICS_TIME_BUCKET_COUNT = 8;
const OPTICS_CANDIDATE_RADIUS_MARGIN = 1e-4;
/** Largest finite binary32; widening a legal radius must stay representable by WGSL. */
const OPTICS_MAXIMUM_SEARCH_RADIUS = 3.4028234663852886e38;

/**
 * Grid reachability: a widened radius self join lists candidate rows once; every visit relaxes
 * its candidates after the exhaustive step's exact distance and time tests, and selects the next
 * row by (reachability, row) from the seed list (rows with finite reachability that are not yet
 * visited) or, when that list is empty, as the lowest unvisited row past a cursor. Every
 * unvisited row outside the seed list has infinite reachability, so the selection equals the
 * exhaustive argmin.
 *
 * Seven storage bindings per frontier batch: a packed `[x, y, time, coreDistance]` row copy, the
 * CSR offsets and IDs, the three outputs, and one scratch view holding the seed list, the row
 * states (unvisited, visited, or seed slot) and the frontier words.
 */
export function createOPTICSGridFrontierNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  context: Pick<
    GPUOPTICS,
    'id' | 'props' | 'neighborCapacity' | 'spatialContext' | 'temporalContext'
  >,
  searchFrame: {positions: GraphDataView<'float32x2'>; radius: number},
  distanceSource: string,
  timeBindings: {name: string; view: GraphDataView; type: 'f32'; access: 'read'}[],
  processed: GraphDataView<'uint32'>,
  overflow: GraphDataView<'uint32'>,
  frontierStatus: GraphDataView<'uint32'>
): GPUCommandNode<Parameters>[] {
  const {id, props, neighborCapacity} = context;
  const {positions, output, times} = props;
  const rows = positions.length;
  const geographic = context.spatialContext.coordinateSpace === 'longitude-latitude';
  const search = createNeighborSearchDerivedParameterNodes<Parameters>(graph, {
    id: `${id}-reach-neighbors-setup`,
    operation: 'GPUOPTICS',
    positions: searchFrame.positions,
    radius: Math.min(
      OPTICS_MAXIMUM_SEARCH_RADIUS,
      searchFrame.radius * (1 + OPTICS_CANDIDATE_RADIUS_MARGIN)
    ),
    geographic,
    times,
    timeRadius:
      props.maximumTimeDistance === undefined
        ? undefined
        : Math.min(
            OPTICS_MAXIMUM_SEARCH_RADIUS,
            props.maximumTimeDistance * (1 + OPTICS_CANDIDATE_RADIUS_MARGIN)
          )
  });
  const offsets = createTransientView(graph, `${id}-reach-neighbor-offsets`, 'uint32', rows + 1);
  const neighbors = createTransientView(
    graph,
    `${id}-reach-neighbor-ids`,
    'uint32',
    neighborCapacity
  );
  const points = createTransientView(graph, `${id}-frontier-points`, 'float32', rows * 4);
  // [seed rows (rows) | row states (rows) | seedCount, cursor, visited, overflow].
  const scratch = createTransientView(graph, `${id}-frontier-scratch`, 'uint32', rows * 2 + 4);
  const gate =
    props.orderingMethod === 'parallel'
      ? undefined
      : createWGSLDispatchGate(graph, {
          id: `${id}-frontier-dispatch`,
          operation: 'GPUOPTICS',
          shapes: [[256, 256]]
        });
  const declarations = `${IS_FINITE_FLOAT_WGSL}
${distanceSource}
const ROWS = ${rows}u;
const MAXIMUM_DISTANCE = ${Math.fround(props.maximumDistance)}f;
const ROW_STATES = ${rows}u;
const SEED_COUNT = ${rows * 2}u;
const CURSOR = ${rows * 2 + 1}u;
const VISITED = ${rows * 2 + 2}u;
const OVERFLOW = ${rows * 2 + 3}u;
// A row state is UNVISITED, VISITED, or the row's slot in the seed list.
const UNVISITED = 0xffffffffu;
const VISITED_ROW = 0xfffffffeu;
fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + 4u * row], points[pointsOffset + 4u * row + 1u]);
}
${
  times
    ? `const MAXIMUM_TIME_DISTANCE = ${Math.fround(props.maximumTimeDistance!)}f;
fn isRelated(first: u32, second: u32) -> bool {
  return abs(points[pointsOffset + 4u * first + 2u] - points[pointsOffset + 4u * second + 2u]) <= MAXIMUM_TIME_DISTANCE;
}`
    : ''
}`;
  const nodes: GPUCommandNode<Parameters>[] = [
    ...search.nodes,
    ...new GPUNeighborSearch({
      id: `${id}-reach-neighbors`,
      mode: 'radius',
      spatialContext: context.spatialContext,
      positions: searchFrame.positions,
      times,
      temporalContext: context.temporalContext,
      timeBucketCount: times ? OPTICS_TIME_BUCKET_COUNT : undefined,
      parameters: search.parameters,
      gridSize: search.gridSize,
      output: {
        offsets,
        neighbors,
        weights: createTransientView(
          graph,
          `${id}-reach-neighbor-weights`,
          'float32',
          neighborCapacity
        ),
        status: {overflow}
      }
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-frontier-prepare`,
      operation: 'GPUOPTICS',
      variant: 'grid-frontier-prepare',
      invocationCount: rows,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        ...timeBindings,
        {name: 'core', view: output.coreDistances, type: 'f32', access: 'read'},
        {name: 'processed', view: processed, type: 'u32', access: 'read'},
        {name: 'overflow', view: overflow, type: 'u32', access: 'read'},
        {name: 'points', view: points, type: 'f32', access: 'read_write'},
        {name: 'scratch', view: scratch, type: 'u32', access: 'read_write'},
        ...(gate ? [gate.binding] : [])
      ],
      body: `points[pointsOffset + 4u * index] = positions[positionsOffset + 2u * index];
  points[pointsOffset + 4u * index + 1u] = positions[positionsOffset + 2u * index + 1u];
  points[pointsOffset + 4u * index + 2u] = ${times ? 'times[timesOffset + index]' : '0.0'};
  points[pointsOffset + 4u * index + 3u] = core[coreOffset + index];
  // Invalid rows were marked processed by the initialize pass.
  scratch[scratchOffset + ${rows}u + index] = select(0xffffffffu, 0xfffffffeu, processed[processedOffset + index] != 0u);
  if (index == 0u) {
    scratch[scratchOffset + ${rows * 2}u] = 0u;
    scratch[scratchOffset + ${rows * 2 + 1}u] = 0u;
    scratch[scratchOffset + ${rows * 2 + 2}u] = 0u;
    scratch[scratchOffset + ${rows * 2 + 3}u] = overflow[overflowOffset];
    ${gate ? `${gate.armWGSL}\n    if (overflow[overflowOffset] != 0u) { ${gate.disarmWGSL} }` : ''}
  }`
    })
  ];
  if (props.orderingMethod === 'parallel') {
    nodes.push(
      ...createOPTICSParallelFrontierNodes(graph, props, {
        distanceSource,
        points,
        offsets,
        neighbors,
        overflow,
        processed,
        frontierStatus
      })
    );
    return nodes;
  }
  const batchSize = props.frontierBatchSize ?? GPU_OPTICS_DEFAULT_FRONTIER_BATCH_SIZE;
  const iterations = props.maximumIterations ?? rows;
  const bindings = [
    {name: 'points', view: points, type: 'f32' as const, access: 'read' as const},
    {name: 'offsets', view: offsets, type: 'u32' as const, access: 'read' as const},
    {name: 'neighbors', view: neighbors, type: 'u32' as const, access: 'read' as const},
    {
      name: 'ordering',
      view: output.ordering,
      type: 'u32' as const,
      access: 'read_write' as const
    },
    {
      name: 'reachability',
      view: output.reachability,
      type: 'f32' as const,
      access: 'read_write' as const
    },
    {
      name: 'predecessors',
      view: output.predecessors,
      type: 'u32' as const,
      access: 'read_write' as const
    },
    {name: 'scratch', view: scratch, type: 'u32' as const, access: 'read_write' as const}
  ];
  for (let batch = 0; batch < Math.ceil(iterations / batchSize); batch++) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-frontier-${batch}`,
        operation: 'GPUOPTICS',
        variant: 'grid-frontier-batch',
        invocationCount: 256,
        workgroupSize: 256,
        guardIndex: false,
        ...gate!.getGate(0),
        bindings,
        declarations: `${declarations}
var<workgroup> candidateRows: array<u32, 256>;
var<workgroup> candidateDistances: array<f32, 256>;
var<workgroup> visited: u32;
var<workgroup> seedCount: u32;
var<workgroup> cursor: u32;
var<workgroup> focusRow: u32;
var<workgroup> appendCount: atomic<u32>;
var<workgroup> overflowed: u32;`,
        body: `let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  if (index == 0u) {
    overflowed = scratch[scratchOffset + OVERFLOW];
    visited = scratch[scratchOffset + VISITED];
    seedCount = scratch[scratchOffset + SEED_COUNT];
    cursor = scratch[scratchOffset + CURSOR];
    atomicStore(&appendCount, 0u);
  }
  // A truncated candidate list would drop relaxations: leave the ordering incomplete.
  if (workgroupUniformLoad(&overflowed) != 0u) { return; }
  for (var iteration = 0u; iteration < ${Math.min(batchSize, iterations - batch * batchSize)}u; iteration++) {
    let count = workgroupUniformLoad(&seedCount);
    if (count > 0u) {
      var selected = 0xffffffffu; var best = infinity;
      for (var slot = index; slot < count; slot += 256u) {
        let row = scratch[scratchOffset + slot];
        let candidate = reachability[reachabilityOffset + row];
        if (candidate < best || (candidate == best && row < selected)) { selected = row; best = candidate; }
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
      if (index == 0u) { focusRow = candidateRows[0]; }
    } else {
      // Every unvisited row has infinite reachability: the next root is the lowest unvisited row.
      if (index == 0u) { focusRow = 0xffffffffu; }
      loop {
        let base = workgroupUniformLoad(&cursor);
        if (base >= ROWS) { break; }
        let row = base + index;
        candidateRows[index] = select(0xffffffffu, row, row < ROWS && scratch[scratchOffset + ROW_STATES + min(row, ROWS - 1u)] == UNVISITED);
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride /= 2u) {
          if (index < stride) { candidateRows[index] = min(candidateRows[index], candidateRows[index + stride]); }
          workgroupBarrier();
        }
        if (index == 0u) {
          if (candidateRows[0] == 0xffffffffu) { cursor = base + 256u; } else { cursor = candidateRows[0]; focusRow = candidateRows[0]; }
        }
        if (workgroupUniformLoad(&focusRow) != 0xffffffffu) { break; }
      }
    }
    let focus = workgroupUniformLoad(&focusRow);
    if (focus == 0xffffffffu) { break; }
    if (index == 0u) {
      ordering[orderingOffset + visited] = focus; visited++;
      let slot = scratch[scratchOffset + ROW_STATES + focus];
      if (slot != UNVISITED) {
        // The focus is a seed: move the last seed into its slot.
        let last = scratch[scratchOffset + seedCount - 1u];
        scratch[scratchOffset + slot] = last; scratch[scratchOffset + ROW_STATES + last] = slot;
        seedCount--;
      }
      scratch[scratchOffset + ROW_STATES + focus] = VISITED_ROW;
    }
    storageBarrier(); workgroupBarrier();
    let coreDistance = points[pointsOffset + 4u * focus + 3u];
    let seedBase = workgroupUniformLoad(&seedCount);
    if (isFiniteFloat(coreDistance)) {
      let end = offsets[offsetsOffset + focus + 1u];
      for (var slot = offsets[offsetsOffset + focus] + index; slot < end; slot += 256u) {
        let row = neighbors[neighborsOffset + slot];
        let state = scratch[scratchOffset + ROW_STATES + row];
        if (state == VISITED_ROW) { continue; }
        ${times ? 'if (!isRelated(focus, row)) { continue; }\n        ' : ''}let pairDistance = getPointDistance(point(focus), point(row));
        if (pairDistance > MAXIMUM_DISTANCE) { continue; }
        let candidate = max(coreDistance, pairDistance);
        if (candidate < reachability[reachabilityOffset + row]) {
          reachability[reachabilityOffset + row] = candidate; predecessors[predecessorsOffset + row] = focus;
          if (state == UNVISITED) {
            let seedSlot = seedBase + atomicAdd(&appendCount, 1u);
            scratch[scratchOffset + seedSlot] = row; scratch[scratchOffset + ROW_STATES + row] = seedSlot;
          }
        }
      }
    }
    storageBarrier(); workgroupBarrier();
    if (index == 0u) { seedCount += atomicExchange(&appendCount, 0u); }
    workgroupBarrier();
  }
  if (index == 0u) {
    scratch[scratchOffset + SEED_COUNT] = seedCount;
    scratch[scratchOffset + CURSOR] = cursor;
    scratch[scratchOffset + VISITED] = visited;
  }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-frontier-gate-${batch}`,
        operation: 'GPUOPTICS',
        variant: 'grid-frontier-gate',
        invocationCount: 1,
        bindings: [
          {name: 'scratch', view: scratch, type: 'u32', access: 'read'},
          {name: 'status', view: frontierStatus, type: 'u32', access: 'read'},
          gate!.binding
        ],
        body: `if (scratch[scratchOffset + ${rows * 2}u] == 0u &&
      (scratch[scratchOffset + ${rows * 2 + 1}u] >= ${rows}u ||
       scratch[scratchOffset + ${rows * 2 + 2}u] + status[statusOffset + 2u] == ${rows}u)) {
    ${gate!.disarmWGSL}
  }`
      })
    );
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-frontier-status`,
      operation: 'GPUOPTICS',
      variant: 'grid-frontier-status',
      invocationCount: 1,
      bindings: [
        {name: 'scratch', view: scratch, type: 'u32', access: 'read'},
        {name: 'status', view: frontierStatus, type: 'u32', access: 'read_write'}
      ],
      body: `let visited = scratch[scratchOffset + ${rows * 2 + 2}u];
  status[statusOffset] = visited;
  status[statusOffset + 1u] = select(0u, 1u, visited + status[statusOffset + 2u] == ${rows}u);`
    })
  );
  return nodes;
}
