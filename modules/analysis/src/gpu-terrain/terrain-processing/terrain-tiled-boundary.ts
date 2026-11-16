// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createGraphChunkGroupKernelNode,
  type GraphChunkGroupKernelProps
} from '../../analysis-core/kernels/graph-chunk-groups';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';
import {getTerrainFlowTracingWGSL} from '../hydrology/terrain-flow-tracing';
import type {GPUTerrainTiledFlowProps} from './gpu-terrain-tiled-flow';
import {
  appendTerrainForestSchedule,
  appendTerrainCycleCutSchedule,
  createTerrainForestStorage,
  sliceTerrainForestStorage
} from './terrain-tiled-forest';
import {
  appendTerrainAlternatingSchedule,
  createTerrainTileSequence,
  type TerrainKernelSchedule
} from './terrain-tiled-sequence';
import {
  TERRAIN_TILE_GEOMETRY_WGSL,
  TERRAIN_TILE_LOOKUP_WGSL,
  type TerrainD8TilePlan
} from './terrain-tiled-plan';
import {terrainBinding, TERRAIN_FINITE_WGSL} from './terrain-processing-utils';

// A tally invocation handles at most one fixed 256-cell block, then adds two integer counts.
const TERRAIN_TILED_TALLY_BLOCK_SIZE = 256;

/** Barnes-style D8 boundary forest: summarize tiles, solve the coarse graph, then replay tiles. @internal */
export function createTerrainBoundaryTileNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUTerrainTiledFlowProps,
  id: string,
  plan: TerrainD8TilePlan
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const parameter = (name: string, values: Uint32Array) => {
    const result = createSpatialStatisticsParameterNodes(graph, `${id}-${name}`, values);
    nodes.push(...result.nodes);
    return result.parameters;
  };
  const rectangles = parameter('tile-rectangles', plan.rectangles);
  const rowBands = parameter('tile-row-bands', plan.rowBands);
  const columnBands = parameter('tile-column-bands', plan.columnBands);
  const lookup = [
    terrainBinding('rectangles', rectangles),
    terrainBinding('rowBands', rowBands),
    terrainBinding('columnBands', columnBands)
  ];
  const boundaryWords = createTransientView(
    graph,
    `${id}-boundary-words`,
    'uint32',
    plan.boundaryCount * 2
  );
  const boundaryReach = [0, 1].map(side =>
    createTransientView(graph, `${id}-boundary-reach-${side}`, 'uint32', plan.boundaryCount * 2)
  );
  const coarse = createTerrainForestStorage(graph, `${id}-boundary-forest`, plan.boundaryCount);
  // One scratch set is reused by every tile in both passes; its size depends on the largest tile.
  const scratch = createTerrainForestStorage(graph, `${id}-tile-forest`, plan.maximumCellCount);
  const pointerStorage = [0, 1].map(side =>
    createTransientView(graph, `${id}-tile-pointers-${side}`, 'uint32', plan.maximumCellCount)
  );
  const reachStorage = [0, 1].map(side =>
    createTransientView(graph, `${id}-tile-reach-${side}`, 'uint32', plan.maximumCellCount)
  );
  const totals = createTransientView(graph, `${id}-totals`, 'uint32', 2);
  const common = `${getTerrainFlowTracingWGSL(props.grid.width, props.grid.height, props.directionEncoding)}\n${TERRAIN_FINITE_WGSL}\n${TERRAIN_TILE_GEOMETRY_WGSL}`;
  const node = (
    suffix: string,
    invocationCount: number,
    bindings: GraphChunkGroupKernelProps<Parameters>['bindings'],
    body: string,
    parameterValues: readonly number[] = [],
    declarations = common
  ) =>
    createGraphChunkGroupKernelNode(graph, {
      id: `${id}-${suffix}`,
      pipelineId: `terrain-tiled-${suffix.replace(/-\d+/g, '')}`,
      operation: 'GPUTerrainTiledFlow',
      invocationCount,
      bindings,
      body,
      declarations,
      parameterValues
    });
  nodes.push(
    node(
      'initialize',
      1,
      [terrainBinding('totals', totals, 'read_write')],
      'totals[totalsOffset] = 0u; totals[totalsOffset + 1u] = 0u;',
      [],
      ''
    )
  );
  const settings = (tileIndex: number, phase = 0) => {
    const tile = props.tiles[tileIndex];
    const shape = plan.tiles[tileIndex];
    return [
      tile.width,
      tile.height,
      ...tile.owned,
      shape.column,
      shape.row,
      shape.boundaryBase,
      shape.boundaryCount,
      phase
    ];
  };
  for (const [tileIndex, tile] of props.tiles.entries()) {
    const shape = plan.tiles[tileIndex];
    const parameters = settings(tileIndex);
    nodes.push(
      node(
        `boundary-words-${tileIndex}`,
        shape.boundaryCount,
        [
          terrainBinding('directions', tile.flowDirections),
          terrainBinding('streams', tile.streams),
          terrainBinding('words', boundaryWords, 'read_write')
        ],
        `
let local = boundaryLocal(index, tileWidth(), tileHeight()); let source = localInput(local);
let destination = (tileParameter(8u) + index) * 2u + wordsOffset;
words[destination] = getFlowDirectionCode(directions[directionsOffset + source]); words[destination + 1u] = streams[streamsOffset + source];`,
        parameters
      )
    );
  }
  const predecessor = `${common}\n${TERRAIN_TILE_LOOKUP_WGSL}
fn streamPredecessor(cell: u32) -> vec2<u32> {
 var donor = TILE_NONE; var count = 0u; let global = globalCell(cell);
 for (var direction = 0u; direction < 8u; direction++) {
  let neighbor = getFlowReceiver(global, 1u << direction); if (neighbor == NO_CELL) { continue; }
  let local = localCell(neighbor); var code = 0u; var stream = 0u;
  if (local != TILE_NONE) { let source = localInput(local); code = getFlowDirectionCode(directions[directionsOffset + source]); stream = streams[streamsOffset + source]; }
  else { let source = wordsOffset + globalBoundary(neighbor) * 2u; code = words[source]; stream = words[source + 1u]; }
  if (isStream(stream) && getFlowReceiver(neighbor, code) == global) { donor = neighbor; count++; }
 }
 return vec2<u32>(donor, count);
}`;
  const replay = (tileIndex: number, phase: number) => {
    const tile = props.tiles[tileIndex];
    const shape = plan.tiles[tileIndex];
    const count = shape.width * shape.height;
    const parameters = settings(tileIndex, phase);
    const local = sliceTerrainForestStorage(graph, scratch, count);
    const pointers = pointerStorage.map(view =>
      graph.createDataView(view.buffer, {format: 'uint32', length: count})
    );
    const reach = reachStorage.map(view =>
      graph.createDataView(view.buffer, {format: 'uint32', length: count})
    );
    const schedule: TerrainKernelSchedule<Parameters> = [];
    schedule.push({
      repetitions: 1,
      nodes: [
        node(
          `prepare-${phase}-${tileIndex}`,
          count,
          [
            terrainBinding('directions', tile.flowDirections),
            terrainBinding('runoff', tile.runoff),
            terrainBinding('streams', tile.streams),
            terrainBinding('boundaryValues', coarse.values),
            terrainBinding('boundaryStates', coarse.resolved),
            terrainBinding('parents', local.parents, 'read_write'),
            terrainBinding('weights', local.weights, 'read_write'),
            terrainBinding('states', local.states, 'read_write')
          ],
          `
let source = localInput(index); let global = globalCell(index);
let code = getFlowDirectionCode(directions[directionsOffset + source]); let receiver = getFlowReceiver(global, code);
let validCode = code == 0u || (code <= 128u && (code & (code - 1u)) == 0u && receiver != NO_CELL);
var weight = runoff[runoffOffset + source]; var state = u32(validCode && streams[streamsOffset + source] < UNRESOLVED_DIRECTION && isFiniteValue(weight) && weight >= 0.0);
var parent = TILE_NONE; if (receiver != NO_CELL) { parent = localCell(receiver); }
if (tileParameter(10u) == 0u) { if (isBoundary(index)) { parent = TILE_NONE; } }
else {
 if (parent != TILE_NONE && isBoundary(parent)) { parent = TILE_NONE; }
 if (isBoundary(index)) { let slot = tileParameter(8u) + boundarySlot(index, tileWidth(), tileHeight()); weight = boundaryValues[boundaryValuesOffset + slot]; state = boundaryStates[boundaryStatesOffset + slot]; }
}
parents[parentsOffset + index] = parent; weights[weightsOffset + index] = weight; states[statesOffset + index] = state;`,
          parameters
        )
      ]
    });
    if (phase === 0) {
      schedule.push({
        repetitions: 1,
        nodes: [
          node(
            `downstream-seeds-${tileIndex}`,
            count,
            [
              terrainBinding('directions', tile.flowDirections),
              terrainBinding('pointers', pointers[0], 'read_write')
            ],
            `
let code = getFlowDirectionCode(directions[directionsOffset + localInput(index)]); let receiver = getFlowReceiver(globalCell(index), code);
var parent = index; if (!isBoundary(index) && receiver != NO_CELL) { let local = localCell(receiver); if (local != TILE_NONE) { parent = local; } }
pointers[pointersOffset + index] = parent;`,
            parameters
          )
        ]
      });
    }
    schedule.push({
      repetitions: 1,
      nodes: [
        node(
          `reach-seeds-${tileIndex}`,
          count,
          [
            terrainBinding('directions', tile.flowDirections),
            terrainBinding('streams', tile.streams),
            terrainBinding('words', boundaryWords),
            ...lookup,
            terrainBinding('reach', reach[0], 'read_write')
          ],
          `
var parent = index;
if (!isBoundary(index) && isStream(streams[streamsOffset + localInput(index)])) {
 let donor = streamPredecessor(index); if (donor.y == 1u) { let local = localCell(donor.x); if (local != TILE_NONE) { parent = local; } }
}
reach[reachOffset + index] = parent;`,
          parameters,
          predecessor
        )
      ]
    });
    const rounds = Math.ceil(Math.log2(Math.max(count, 1)));
    const appendPointers = (views: GraphDataView<'uint32'>[], name: string) => {
      const passes = [0, 1].map(side =>
        node(
          `${name}-${tileIndex}-${side}`,
          count,
          [
            terrainBinding('previous', views[side]),
            terrainBinding('next', views[1 - side], 'read_write')
          ],
          'next[nextOffset + index] = previous[previousOffset + previous[previousOffset + index]];',
          [],
          ''
        )
      ) as [GPUCommandNode<Parameters>, GPUCommandNode<Parameters>];
      appendTerrainAlternatingSchedule(schedule, passes, rounds);
      return views[rounds % 2];
    };
    const downstream = phase === 0 ? appendPointers(pointers, 'pointer-jump') : pointers[0];
    const reachRoots = appendPointers(reach, 'pointer-jump');
    appendTerrainCycleCutSchedule(
      graph,
      schedule,
      `${id}-tile-${tileIndex}-phase-${phase}-cycles`,
      local
    );
    schedule.push({
      repetitions: 1,
      nodes: [
        node(
          `topology-${tileIndex}`,
          count,
          [
            terrainBinding('parents', local.parents),
            terrainBinding('children', local.firstChildren, 'read_write'),
            terrainBinding('siblings', local.nextSiblings, 'read_write')
          ],
          `
var child = TILE_NONE;
for (var direction = 0u; direction < 8u; direction++) { let neighbor = localNeighbor(index, direction); if (neighbor != TILE_NONE && parents[parentsOffset + neighbor] == index && child == TILE_NONE) { child = neighbor; } }
children[childrenOffset + index] = child;
let parent = parents[parentsOffset + index]; var sibling = TILE_NONE; var passed = false;
if (parent != TILE_NONE) { for (var direction = 0u; direction < 8u; direction++) {
 let neighbor = localNeighbor(parent, direction); if (neighbor == index) { passed = true; continue; }
 if (neighbor != TILE_NONE && passed && sibling == TILE_NONE && parents[parentsOffset + neighbor] == parent) { sibling = neighbor; }
} }
siblings[siblingsOffset + index] = sibling;`,
          parameters
        )
      ]
    });
    appendTerrainForestSchedule(graph, schedule, `${id}-tile-${tileIndex}-phase-${phase}`, local);
    if (phase === 0) {
      schedule.push({
        repetitions: 1,
        nodes: [
          node(
            `boundary-seeds-${tileIndex}`,
            shape.boundaryCount,
            [
              terrainBinding('values', local.values),
              terrainBinding('resolved', local.resolved),
              terrainBinding('weights', coarse.weights, 'read_write'),
              terrainBinding('states', coarse.states, 'read_write')
            ],
            `
let local = boundaryLocal(index, tileWidth(), tileHeight()); let destination = tileParameter(8u) + index;
weights[weightsOffset + destination] = values[valuesOffset + local]; states[statesOffset + destination] = resolved[resolvedOffset + local];`,
            parameters
          ),
          node(
            `boundary-parents-${tileIndex}`,
            shape.boundaryCount,
            [
              terrainBinding('directions', tile.flowDirections),
              terrainBinding('pointers', downstream),
              ...lookup,
              terrainBinding('parents', coarse.parents, 'read_write')
            ],
            `
let local = boundaryLocal(index, tileWidth(), tileHeight()); let receiver = getFlowReceiver(globalCell(local), getFlowDirectionCode(directions[directionsOffset + localInput(local)]));
var parent = TILE_NONE;
if (receiver != NO_CELL) {
 let receiverLocal = localCell(receiver);
 if (receiverLocal == TILE_NONE) { parent = globalBoundary(receiver); }
 else { let terminal = pointers[pointersOffset + receiverLocal]; if (isBoundary(terminal)) { parent = tileParameter(8u) + boundarySlot(terminal, tileWidth(), tileHeight()); } }
}
parents[parentsOffset + tileParameter(8u) + index] = parent;`,
            parameters,
            `${common}\n${TERRAIN_TILE_LOOKUP_WGSL}`
          ),
          node(
            `boundary-reach-seeds-${tileIndex}`,
            shape.boundaryCount,
            [
              terrainBinding('directions', tile.flowDirections),
              terrainBinding('streams', tile.streams),
              terrainBinding('words', boundaryWords),
              ...lookup,
              terrainBinding('pointers', reachRoots),
              terrainBinding('reach', boundaryReach[0], 'read_write')
            ],
            `
let local = boundaryLocal(index, tileWidth(), tileHeight()); var parent = TILE_NONE; var source = TILE_NONE;
if (isStream(streams[streamsOffset + localInput(local)])) {
 let donor = streamPredecessor(local);
 if (donor.y != 1u) { source = globalCell(local); }
 else {
  let donorLocal = localCell(donor.x);
  if (donorLocal == TILE_NONE) { parent = globalBoundary(donor.x); }
  else { let terminal = pointers[pointersOffset + donorLocal]; if (isBoundary(terminal)) { parent = tileParameter(8u) + boundarySlot(terminal, tileWidth(), tileHeight()); } else { source = globalCell(terminal); } }
 }
}
let destination = reachOffset + (tileParameter(8u) + index) * 2u; reach[destination] = parent; reach[destination + 1u] = source;`,
            parameters,
            predecessor
          )
        ]
      });
    }
    nodes.push(createTerrainTileSequence(`${id}-tile-${tileIndex}-phase-${phase}`, schedule));
    return {local, reachRoots, parameters};
  };
  for (let tileIndex = 0; tileIndex < props.tiles.length; tileIndex++) replay(tileIndex, 0);
  const cycleSchedule: TerrainKernelSchedule<Parameters> = [];
  appendTerrainCycleCutSchedule(graph, cycleSchedule, `${id}-boundary-cycles`, coarse);
  nodes.push(createTerrainTileSequence(`${id}-boundary-cycles`, cycleSchedule));
  const identifiers = createTransientView(
    graph,
    `${id}-boundary-identifiers`,
    'uint32',
    plan.boundaryCount
  );
  const sortedParents = createTransientView(
    graph,
    `${id}-boundary-sorted-parents`,
    'uint32',
    plan.boundaryCount
  );
  const sortedChildren = createTransientView(
    graph,
    `${id}-boundary-sorted-children`,
    'uint32',
    plan.boundaryCount
  );
  nodes.push(
    node(
      'boundary-identifiers',
      plan.boundaryCount,
      [terrainBinding('identifiers', identifiers, 'read_write')],
      'identifiers[identifiersOffset + index] = index;',
      [],
      ''
    ),
    ...new GPUSort({
      id: `${id}-boundary-order`,
      keys: coarse.parents,
      values: identifiers,
      outputKeys: sortedParents,
      outputValues: sortedChildren
    }).getCommandNodes(graph)
  );
  nodes.push(
    node(
      'boundary-topology',
      plan.boundaryCount,
      [
        terrainBinding('parents', coarse.parents),
        terrainBinding('sortedParents', sortedParents),
        terrainBinding('sortedChildren', sortedChildren),
        terrainBinding('children', coarse.firstChildren, 'read_write'),
        terrainBinding('siblings', coarse.nextSiblings, 'read_write')
      ],
      `
let count = getChunkParameter(1u); let first = lowerBoundSortedParents(0u, count, index);
var child = 0xffffffffu; if (first < count && sortedParents[sortedParentsOffset + first] == index) { child = sortedChildren[sortedChildrenOffset + first]; }
children[childrenOffset + index] = child;
let parent = parents[parentsOffset + index]; var sibling = 0xffffffffu;
if (parent != 0xffffffffu) {
 let firstSibling = lowerBoundSortedParents(0u, count, parent); let end = upperBoundSortedParents(firstSibling, count, parent);
 // Stable radix sorting preserves ascending child IDs within one parent segment.
 let position = lowerBoundSortedChildren(firstSibling, end, index);
 if (position + 1u < end) { sibling = sortedChildren[sortedChildrenOffset + position + 1u]; }
}
siblings[siblingsOffset + index] = sibling;`,
      [],
      `${getBinarySearchWGSL('sortedParents')}\n${getBinarySearchWGSL('sortedChildren')}`
    )
  );
  const coarseSchedule: TerrainKernelSchedule<Parameters> = [];
  appendTerrainForestSchedule(graph, coarseSchedule, `${id}-boundary`, coarse);
  nodes.push(createTerrainTileSequence(`${id}-boundary-solve`, coarseSchedule));
  const reachRounds = Math.ceil(Math.log2(Math.max(plan.boundaryCount, 1)));
  const reachSchedule: TerrainKernelSchedule<Parameters> = [];
  const reachPasses = [0, 1].map(side =>
    node(
      `boundary-reach-jump-${side}`,
      plan.boundaryCount,
      [
        terrainBinding('previous', boundaryReach[side]),
        terrainBinding('next', boundaryReach[1 - side], 'read_write')
      ],
      `
let base = previousOffset + index * 2u; let parent = previous[base];
var following = parent; var source = previous[base + 1u];
if (parent != 0xffffffffu) { following = previous[previousOffset + parent * 2u]; source = previous[previousOffset + parent * 2u + 1u]; }
next[nextOffset + index * 2u] = following; next[nextOffset + index * 2u + 1u] = source;`,
      [],
      ''
    )
  ) as [GPUCommandNode<Parameters>, GPUCommandNode<Parameters>];
  appendTerrainAlternatingSchedule(reachSchedule, reachPasses, reachRounds);
  if (reachSchedule.length)
    nodes.push(createTerrainTileSequence(`${id}-boundary-reaches`, reachSchedule));
  for (const [tileIndex, tile] of props.tiles.entries()) {
    const {local, reachRoots, parameters} = replay(tileIndex, 1);
    nodes.push(
      node(
        `publish-bounded-${tileIndex}`,
        tile.width * tile.height,
        [
          terrainBinding('values', local.values),
          terrainBinding('resolved', local.resolved),
          terrainBinding('pointers', reachRoots),
          terrainBinding('reach', boundaryReach[reachRounds % 2]),
          terrainBinding('streams', tile.streams),
          terrainBinding('accumulation', tile.output.accumulation, 'read_write'),
          terrainBinding('status', tile.output.status, 'read_write'),
          terrainBinding('reachIds', tile.output.reachIds, 'read_write')
        ],
        `
let column = index % tileParameter(0u); let row = index / tileParameter(0u);
var value = getNaN(); var state = 0u; var source = TILE_NONE;
if (column >= tileParameter(2u) && row >= tileParameter(3u) && column - tileParameter(2u) < tileWidth() && row - tileParameter(3u) < tileHeight()) {
 let local = (row - tileParameter(3u)) * tileWidth() + column - tileParameter(2u);
 state = resolved[resolvedOffset + local]; value = values[valuesOffset + local];
 if (state == 1u && isStream(streams[streamsOffset + index])) {
  let terminal = pointers[pointersOffset + local];
  if (isBoundary(terminal)) { source = reach[reachOffset + (tileParameter(8u) + boundarySlot(terminal, tileWidth(), tileHeight())) * 2u + 1u]; }
  else { source = globalCell(terminal); }
 }
}
accumulation[accumulationOffset + index] = value; status[statusOffset + index] = state; reachIds[reachIdsOffset + index] = source;`,
        parameters
      )
    );
    nodes.push(
      node(
        `tally-${tileIndex}`,
        Math.ceil(local.parents.length / TERRAIN_TILED_TALLY_BLOCK_SIZE),
        [
          terrainBinding('states', tile.output.status),
          {name: 'totals', view: totals, type: 'atomic<u32>', access: 'read_write'}
        ],
        `
var invalid = 0u; var unresolved = 0u;
for (var position = index * ${TERRAIN_TILED_TALLY_BLOCK_SIZE}u; position < min((index + 1u) * ${TERRAIN_TILED_TALLY_BLOCK_SIZE}u, tileWidth() * tileHeight()); position++) {
 let state = states[statesOffset + localInput(position)]; invalid += u32(state == 0u); unresolved += u32(state == 2u);
}
atomicAdd(&totals[totalsOffset], invalid); atomicAdd(&totals[totalsOffset + 1u], unresolved);`,
        parameters
      )
    );
  }
  for (const [kind, view] of [
    props.output?.status.invalidCount,
    props.output?.status.converged,
    props.output?.status.iterationLimitReached
  ].entries()) {
    if (view)
      nodes.push(
        node(
          `aggregate-${kind}`,
          1,
          [terrainBinding('totals', totals), terrainBinding('port', view, 'read_write')],
          `
var value = 0u; let kind = getChunkParameter(36u);
if (kind == 0u) { value = totals[totalsOffset]; } else if (kind == 1u) { value = u32(totals[totalsOffset + 1u] == 0u); }
port[portOffset] = value;`,
          [kind],
          ''
        )
      );
  }
  return nodes;
}
