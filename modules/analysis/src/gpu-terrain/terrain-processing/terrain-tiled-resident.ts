// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createGraphChunkGroupKernelNode,
  GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET
} from '../../analysis-core/kernels/graph-chunk-groups';
import {getTerrainFlowTracingWGSL} from '../hydrology/terrain-flow-tracing';
import {createTerrainBranchingFlowNodes} from './gpu-terrain-branching-flow';
import type {GPUTerrainTiledFlowProps} from './gpu-terrain-tiled-flow';
import {createTerrainResolutionStatusNodes} from './terrain-resolution-status';
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
  // A stream cell is any nonzero word except the two unresolved sentinels (0xffffffff, 0xfffffffe),
  // the same test as stream order, stream network, flow distance and height above drainage.
  const declarations = `${getTerrainFlowTracingWGSL(props.grid.width, props.grid.height, props.directionEncoding)}\n${TERRAIN_FINITE_WGSL}
fn isStreamWord(word: u32) -> bool { return word != 0u && word < UNRESOLVED_DIRECTION; }`;
  const tileDeclarations = `${declarations}
fn getTileParameter(word: u32) -> u32 { return getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u + word); }`;
  const nodes: GPUCommandNode<Parameters>[] = [];
  props.tiles.forEach((tile, tileIndex) => {
    const [column, row, width, height] = tile.owned;
    nodes.push(
      createGraphChunkGroupKernelNode(graph, {
        id: `${id}-exchange-owned-${tileIndex}`,
        pipelineId: `${id}-exchange-owned`,
        operation: 'GPUTerrainTiledFlow',
        invocationCount: width * height,
        declarations: tileDeclarations,
        parameterValues: [
          tile.width,
          column,
          row,
          width,
          height,
          tile.column + column,
          tile.row + row
        ],
        bindings: [
          terrainBinding('directions', tile.flowDirections),
          ...(tile.fractions ? [terrainBinding('inputFractions', tile.fractions)] : []),
          terrainBinding('inputRunoff', tile.runoff),
          terrainBinding('inputStreams', tile.streams),
          terrainBinding('fractions', fractions, 'read_write'),
          terrainBinding('runoff', runoff, 'read_write'),
          terrainBinding('streams', streams, 'read_write')
        ],
        body: `let ownedWidth = getTileParameter(3u);
let local = (index / ownedWidth + getTileParameter(2u)) * getTileParameter(0u) + index % ownedWidth + getTileParameter(1u);
let cell = (index / ownedWidth + getTileParameter(6u)) * GRID_WIDTH + index % ownedWidth + getTileParameter(5u);
let stream = inputStreams[inputStreamsOffset + local];
let code = getFlowDirectionCode(directions[directionsOffset + local]);
var valid = stream < UNRESOLVED_DIRECTION;
${tile.fractions ? 'valid = valid && code != NO_CELL;' : 'valid = valid && (code == 0u || (code <= 128u && (code & (code - 1u)) == 0u && getFlowReceiver(cell, code) != NO_CELL));'}
runoff[runoffOffset + cell] = select(getNaN(), inputRunoff[inputRunoffOffset + local], valid);
streams[streamsOffset + cell] = stream;
for (var direction = 0u; direction < 8u; direction++) { fractions[fractionsOffset + cell * 8u + direction] = ${tile.fractions ? 'inputFractions[inputFractionsOffset + local * 8u + direction]' : 'select(0.0, 1.0, code == (1u << direction))'}; }`
      })
    );
  });
  const maximumIterations =
    props.maximumIterations ?? clamp(props.grid.width + props.grid.height, 128, 1024);
  const solver = createTerrainBranchingFlowNodes(graph, {
    id: `${id}-dag`,
    grid: props.grid,
    fractions,
    values: runoff,
    mode: 'transport',
    maximumIterations,
    output: {values: accumulation, status, unresolvedCount: unresolved}
  });
  nodes.push(...solver.nodes);
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
if (status[statusOffset + index] == 1u && isStreamWord(streams[streamsOffset + index])) {
 var donors = 0u; var inherited = index;
 for (var direction = 0u; direction < 8u; direction++) {
  let neighbor = getFlowReceiver(index, 1u << direction);
  if (neighbor == NO_CELL || !isStreamWord(streams[streamsOffset + neighbor]) || !(fractions[fractionsOffset + neighbor * 8u + (direction + 4u) % 8u] > 0.0)) { continue; }
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
      createGraphChunkGroupKernelNode(graph, {
        id: `${id}-publish-${tileIndex}`,
        pipelineId: `${id}-publish`,
        operation: 'GPUTerrainTiledFlow',
        invocationCount: tile.width * tile.height,
        declarations: tileDeclarations,
        parameterValues: [
          tile.width,
          column,
          row,
          width,
          height,
          tile.column + column,
          tile.row + row
        ],
        bindings: [
          terrainBinding('values', accumulation),
          terrainBinding('globalStatus', status),
          terrainBinding('globalReach', reach[rounds % 2]),
          terrainBinding('accumulation', tile.output.accumulation, 'read_write'),
          terrainBinding('status', tile.output.status, 'read_write'),
          terrainBinding('reachIds', tile.output.reachIds, 'read_write')
        ],
        body: `let column = index % getTileParameter(0u); let row = index / getTileParameter(0u);
let ownedColumn = getTileParameter(1u); let ownedRow = getTileParameter(2u);
let owned = column >= ownedColumn && column - ownedColumn < getTileParameter(3u) && row >= ownedRow && row - ownedRow < getTileParameter(4u);
var value = getNaN(); var state = 0u; var reach = NO_CELL;
if (owned) {
 let cell = (row - ownedRow + getTileParameter(6u)) * GRID_WIDTH + column - ownedColumn + getTileParameter(5u);
 value = values[valuesOffset + cell]; state = globalStatus[globalStatusOffset + cell]; reach = globalReach[globalReachOffset + cell];
}
accumulation[accumulationOffset + index] = value; status[statusOffset + index] = state; reachIds[reachIdsOffset + index] = reach;`
      })
    );
  });
  nodes.push(
    ...createTerrainResolutionStatusNodes(graph, {
      id,
      operation: 'GPUTerrainTiledFlow',
      states: status,
      status: props.output?.status,
      iterationLimit: {
        source: solver.iterationStatus,
        expression: `u32(source[sourceOffset + 2u] >= ${maximumIterations}u && source[sourceOffset + 3u] == 0u)`
      }
    })
  );
  return nodes;
}
