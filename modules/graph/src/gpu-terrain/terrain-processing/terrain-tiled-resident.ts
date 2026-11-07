// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {getTerrainFlowTracingWGSL} from '../hydrology/terrain-flow-tracing';
import {GPUTerrainBranchingFlow} from './gpu-terrain-branching-flow';
import type {GPUTerrainTiledFlowProps} from './gpu-terrain-tiled-flow';
import {terrainBinding, TERRAIN_FINITE_WGSL} from './terrain-processing-utils';

/** Owned-cell exchange into a resident global DAG; no halo or routing recomputation. @internal */
export function createTerrainResidentTileNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUTerrainTiledFlowProps,
  id: string
): GPUCommandNode<Parameters>[] {
  const count = props.grid.width * props.grid.height;
  const fractions = createTransientView(graph, `${id}-fractions`, 'float32', count * 8);
  const runoff = createTransientView(graph, `${id}-runoff`, 'float32', count);
  const streams = createTransientView(graph, `${id}-streams`, 'uint32', count);
  const accumulation = createTransientView(graph, `${id}-accumulation`, 'float32', count);
  const status = createTransientView(graph, `${id}-status`, 'uint32', count);
  const unresolved = createTransientView(graph, `${id}-unresolved`, 'uint32', 1);
  const reach = [0, 1].map(side =>
    createTransientView(graph, `${id}-reach-${side}`, 'uint32', count)
  );
  const declarations = `${getTerrainFlowTracingWGSL(props.grid.width, props.grid.height)}\n${TERRAIN_FINITE_WGSL}`;
  const nodes: GPUCommandNode<Parameters>[] = [];
  props.tiles.forEach((tile, tileIndex) => {
    const [column, row, width, height] = tile.owned;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-exchange-owned-${tileIndex}`,
        operation: 'GPUTerrainTiledFlow',
        invocationCount: width * height,
        declarations,
        bindings: [
          terrainBinding('directions', tile.flowDirections),
          ...(tile.fractions ? [terrainBinding('inputFractions', tile.fractions)] : []),
          terrainBinding('inputRunoff', tile.runoff),
          terrainBinding('inputStreams', tile.streams),
          terrainBinding('fractions', fractions, 'read_write'),
          terrainBinding('runoff', runoff, 'read_write'),
          terrainBinding('streams', streams, 'read_write')
        ],
        body: `let local = (index / ${width}u + ${row}u) * ${tile.width}u + index % ${width}u + ${column}u;
let cell = (index / ${width}u + ${tile.row + row}u) * GRID_WIDTH + index % ${width}u + ${tile.column + column}u;
let stream = inputStreams[inputStreamsOffset + local];
let code = directions[directionsOffset + local];
var valid = stream <= 1u;
${tile.fractions ? '' : 'valid = valid && (code == 0u || (code <= 128u && (code & (code - 1u)) == 0u && getFlowReceiver(cell, code) != NO_CELL));'}
runoff[runoffOffset + cell] = select(missing(), inputRunoff[inputRunoffOffset + local], valid);
streams[streamsOffset + cell] = stream;
for (var direction = 0u; direction < 8u; direction++) { fractions[fractionsOffset + cell * 8u + direction] = ${tile.fractions ? 'inputFractions[inputFractionsOffset + local * 8u + direction]' : 'select(0.0, 1.0, code == (1u << direction))'}; }`
      })
    );
  });
  nodes.push(
    ...new GPUTerrainBranchingFlow({
      id: `${id}-dag`,
      grid: props.grid,
      fractions,
      values: runoff,
      mode: 'transport',
      maximumIterations: props.maximumIterations,
      output: {values: accumulation, status, unresolvedCount: unresolved}
    }).getCommandNodes(graph)
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-reach-seeds`,
      operation: 'GPUTerrainTiledFlow',
      invocationCount: count,
      declarations,
      bindings: [
        terrainBinding('fractions', fractions),
        terrainBinding('streams', streams),
        terrainBinding('status', status),
        terrainBinding('reach', reach[0], 'read_write')
      ],
      body: `var parent = NO_CELL;
if (status[statusOffset + index] == 1u && streams[streamsOffset + index] == 1u) {
 var donors = 0u; var inherited = index;
 for (var direction = 0u; direction < 8u; direction++) {
  let neighbor = getFlowReceiver(index, 1u << direction);
  if (neighbor == NO_CELL || streams[streamsOffset + neighbor] != 1u || !(fractions[fractionsOffset + neighbor * 8u + (direction + 4u) % 8u] > 0.0)) { continue; }
  donors++;
  var receiverCount = 0u;
  for (var receiverDirection = 0u; receiverDirection < 8u; receiverDirection++) { receiverCount += u32(fractions[fractionsOffset + neighbor * 8u + receiverDirection] > 0.0); }
  inherited = select(neighbor, index, receiverCount > 1u);
 }
 parent = select(index, inherited, donors == 1u);
}
reach[reachOffset + index] = parent;`
    })
  );
  const rounds = Math.ceil(Math.log2(Math.max(count, 1)));
  for (let iteration = 0; iteration < rounds; iteration++)
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-reach-jump-${iteration}`,
        operation: 'GPUTerrainTiledFlow',
        invocationCount: count,
        bindings: [
          terrainBinding('previous', reach[iteration % 2]),
          terrainBinding('next', reach[(iteration + 1) % 2], 'read_write')
        ],
        body: 'let parent = previous[previousOffset + index]; var ancestor = parent; if (parent != 0xffffffffu) { ancestor = previous[previousOffset + parent]; } next[nextOffset + index] = ancestor;'
      })
    );
  props.tiles.forEach((tile, tileIndex) => {
    const [column, row, width, height] = tile.owned;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-publish-${tileIndex}`,
        operation: 'GPUTerrainTiledFlow',
        invocationCount: tile.width * tile.height,
        declarations,
        bindings: [
          terrainBinding('values', accumulation),
          terrainBinding('globalStatus', status),
          terrainBinding('globalReach', reach[rounds % 2]),
          terrainBinding('accumulation', tile.output.accumulation, 'read_write'),
          terrainBinding('status', tile.output.status, 'read_write'),
          terrainBinding('reachIds', tile.output.reachIds, 'read_write')
        ],
        body: `let column = index % ${tile.width}u; let row = index / ${tile.width}u;
let owned = column >= ${column}u && column < ${column + width}u && row >= ${row}u && row < ${row + height}u;
var value = missing(); var state = 0u; var reach = NO_CELL;
if (owned) {
 let cell = u32(i32(row) + ${tile.row}) * GRID_WIDTH + u32(i32(column) + ${tile.column});
 value = values[valuesOffset + cell]; state = globalStatus[globalStatusOffset + cell]; reach = globalReach[globalReachOffset + cell];
}
accumulation[accumulationOffset + index] = value; status[statusOffset + index] = state; reachIds[reachIdsOffset + index] = reach;`
      })
    );
  });
  return nodes;
}
