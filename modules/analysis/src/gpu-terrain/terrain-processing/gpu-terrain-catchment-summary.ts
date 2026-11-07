// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateRasterGrid} from '../../raster-analysis/raster-band-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getTerrainFlowTracingWGSL} from '../hydrology/terrain-flow-tracing';
import {
  checkTerrainViews,
  checkTerrainGraph,
  terrainBinding,
  TERRAIN_FINITE_WGSL
} from './terrain-processing-utils';

export type GPUTerrainCatchmentSummaryProps = {
  id?: string;
  /** Planar grid. Area and length use the affine coordinate units; geographic degrees are not ground meters. */
  grid: GPURasterMetadata;
  elevation: GraphDataView<'float32'>;
  /** Dense zero-based basin IDs; 0xffffffff excludes a cell. */
  basinIds: GraphDataView<'uint32'>;
  basinCount: number;
  /** Resolved boolean stream mask (0/1); existing flow unresolved sentinels are excluded. */
  streams: GraphDataView<'uint32'>;
  flowDirections: GraphDataView<'uint32'>;
  histogramBins: number;
  maximumSteps?: number;
  output: {
    /** Six values per basin: valid area, mean, minimum, maximum, stream length, drainage density. */
    statistics: GraphDataView<'float32'>;
    /** Cumulative area by normalized elevation bin (upper edges). Last bin is total valid area. */
    hypsometry: GraphDataView<'float32'>;
    /** Three values per cell: downstream stream distance, elevation, outgoing elevation gradient. */
    profiles: GraphDataView<'float32'>;
    /** Per cell: 0 excluded/invalid, 1 complete profile, 2 cycle or exhausted walk. */
    profileStatus: GraphDataView<'uint32'>;
  };
};

/** Composes sorted basin reductions and pointer-jumped stream profiles. Each edge belongs to its source cell's basin. */
export class GPUTerrainCatchmentSummary implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUTerrainCatchmentSummaryProps) {
    this.id = props.id ?? 'terrain-catchment-summary';
    const count = validateRasterGrid(this.id, props.grid);
    assert(Number.isInteger(props.basinCount) && props.basinCount > 0);
    assert(Number.isInteger(props.histogramBins) && props.histogramBins > 0);
    assert(Number.isInteger(props.maximumSteps ?? count) && (props.maximumSteps ?? count) > 0);
    assert(
      Math.abs(
        props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3]
      ) > 0
    );
    for (const view of [props.elevation, props.basinIds, props.streams, props.flowDirections])
      assert(view.length === count);
    assert(
      props.output.statistics.length === props.basinCount * 6 &&
        props.output.hypsometry.length === props.basinCount * props.histogramBins
    );
    assert(
      props.output.profiles.length === count * 3 && props.output.profileStatus.length === count
    );
    checkTerrainViews(
      this.id,
      [props.elevation, props.basinIds, props.streams, props.flowDirections],
      Object.values(props.output)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const count = props.elevation.length;
    checkTerrainGraph(id, graph, [
      props.elevation,
      props.basinIds,
      props.streams,
      props.flowDirections,
      ...Object.values(props.output)
    ]);
    const affine = props.grid.affine.map(getWGSLFloatLiteral);
    const area = getWGSLFloatLiteral(
      Math.abs(
        props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3]
      )
    );
    const declarations = `${getTerrainFlowTracingWGSL(props.grid.width, props.grid.height)}\n${TERRAIN_FINITE_WGSL}
fn edgeLength(cell: u32, receiver: u32) -> f32 {
 let delta = vec2<f32>(f32(receiver % GRID_WIDTH) - f32(cell % GRID_WIDTH), f32(receiver / GRID_WIDTH) - f32(cell / GRID_WIDTH));
 return length(vec2<f32>(${affine[0]} * delta.x + ${affine[1]} * delta.y, ${affine[3]} * delta.x + ${affine[4]} * delta.y));
}`;
    const inputs = [
      terrainBinding('elevation', props.elevation),
      terrainBinding('basins', props.basinIds),
      terrainBinding('streams', props.streams),
      terrainBinding('directions', props.flowDirections)
    ];
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', count);
    const cells = createTransientView(graph, `${id}-cells`, 'uint32', count);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', count);
    const order = createTransientView(graph, `${id}-order`, 'uint32', count);
    const ranges = createTransientView(graph, `${id}-ranges`, 'uint32', props.basinCount * 2);
    const states = [0, 1].map(side =>
      createTransientView(graph, `${id}-profiles-${side}`, 'uint32', count * 4)
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-basin-keys`,
        operation: 'GPUTerrainCatchmentSummary',
        invocationCount: count,
        declarations: TERRAIN_FINITE_WGSL,
        bindings: [
          terrainBinding('elevation', props.elevation),
          terrainBinding('basins', props.basinIds),
          terrainBinding('keys', keys, 'read_write'),
          terrainBinding('cells', cells, 'read_write')
        ],
        body: `let basin = basins[basinsOffset + index]; keys[keysOffset + index] = select(0xffffffffu, basin, basin < ${props.basinCount}u && finite(elevation[elevationOffset + index])); cells[cellsOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-basin-sort`,
        keys,
        values: cells,
        outputKeys: sortedKeys,
        outputValues: order
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-ranges-initialize`,
        operation: 'GPUTerrainCatchmentSummary',
        invocationCount: props.basinCount,
        bindings: [terrainBinding('ranges', ranges, 'read_write')],
        body: 'ranges[rangesOffset + index * 2u] = 0u; ranges[rangesOffset + index * 2u + 1u] = 0u;'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-ranges`,
        operation: 'GPUTerrainCatchmentSummary',
        invocationCount: count,
        bindings: [
          terrainBinding('keys', sortedKeys),
          terrainBinding('ranges', ranges, 'read_write')
        ],
        body: `let basin = keys[keysOffset + index]; if (basin >= ${props.basinCount}u) { return; }
if (index == 0u || keys[keysOffset + index - 1u] != basin) { ranges[rangesOffset + basin * 2u] = index; }
if (index + 1u == ${count}u || keys[keysOffset + index + 1u] != basin) { ranges[rangesOffset + basin * 2u + 1u] = index + 1u; }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-statistics`,
        operation: 'GPUTerrainCatchmentSummary',
        invocationCount: props.basinCount,
        declarations,
        bindings: [
          ...inputs,
          terrainBinding('order', order),
          terrainBinding('ranges', ranges),
          terrainBinding('statistics', props.output.statistics, 'read_write'),
          terrainBinding('hypsometry', props.output.hypsometry, 'read_write')
        ],
        body: `var sampleCount = 0u; var sum = 0.0; var lower = infinity(); var upper = -lower; var streamLength = 0.0;
for (var entry = ranges[rangesOffset + index * 2u]; entry < ranges[rangesOffset + index * 2u + 1u]; entry++) {
 let cell = order[orderOffset + entry];
 let value = elevation[elevationOffset + cell];
 if (basins[basinsOffset + cell] != index || !finite(value)) { continue; }
 sampleCount++; sum += value; lower = min(lower, value); upper = max(upper, value);
 let receiver = getFlowReceiver(cell, directions[directionsOffset + cell]);
 if (streams[streamsOffset + cell] == 1u && receiver != NO_CELL && streams[streamsOffset + receiver] == 1u && finite(elevation[elevationOffset + receiver])) { streamLength += edgeLength(cell, receiver); }
}
let area = f32(sampleCount) * ${area};
let offset = statisticsOffset + index * 6u;
statistics[offset] = area; statistics[offset + 1u] = select(missing(), sum / f32(max(sampleCount, 1u)), sampleCount > 0u);
statistics[offset + 2u] = select(missing(), lower, sampleCount > 0u); statistics[offset + 3u] = select(missing(), upper, sampleCount > 0u);
statistics[offset + 4u] = streamLength; statistics[offset + 5u] = select(missing(), streamLength / max(area, 1e-30), area > 0.0);
for (var bin = 0u; bin < ${props.histogramBins}u; bin++) {
 let threshold = lower + (upper - lower) * f32(bin + 1u) / ${getWGSLFloatLiteral(props.histogramBins)};
 var belowCount = 0u;
 for (var entry = ranges[rangesOffset + index * 2u]; entry < ranges[rangesOffset + index * 2u + 1u]; entry++) {
  let cell = order[orderOffset + entry];
  let value = elevation[elevationOffset + cell];
  if (basins[basinsOffset + cell] == index && finite(value) && (value <= threshold || bin + 1u == ${props.histogramBins}u)) { belowCount++; }
 }
 hypsometry[hypsometryOffset + index * ${props.histogramBins}u + bin] = f32(belowCount) * ${area};
}`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-profile-seeds`,
        operation: 'GPUTerrainCatchmentSummary',
        invocationCount: count,
        declarations,
        bindings: [...inputs, terrainBinding('state', states[0], 'read_write')],
        body: `var parent = NO_CELL; var distance = 0.0; var steps = 0u; var result = 2u;
let code = directions[directionsOffset + index];
if (finite(elevation[elevationOffset + index]) && streams[streamsOffset + index] == 1u) {
 if (code == 0u) { result = 1u; } else {
  let receiver = getFlowReceiver(index, code);
  if (receiver != NO_CELL && finite(elevation[elevationOffset + receiver]) && streams[streamsOffset + receiver] <= 1u) {
   if (streams[streamsOffset + receiver] == 0u) { result = 1u; }
   else { parent = receiver; distance = edgeLength(index, receiver); steps = 1u; }
  }
 }
}
state[stateOffset + index * 4u] = parent; state[stateOffset + index * 4u + 1u] = bitcast<u32>(distance);
state[stateOffset + index * 4u + 2u] = steps; state[stateOffset + index * 4u + 3u] = result;`
      })
    ];
    const rounds = Math.ceil(Math.log2(Math.max(count, 1)));
    for (let iteration = 0; iteration < rounds; iteration++)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-profile-jump-${iteration}`,
          operation: 'GPUTerrainCatchmentSummary',
          invocationCount: count,
          bindings: [
            terrainBinding('previous', states[iteration % 2]),
            terrainBinding('next', states[(iteration + 1) % 2], 'read_write')
          ],
          body: `let offset = previousOffset + index * 4u; let destination = nextOffset + index * 4u;
for (var component = 0u; component < 4u; component++) { next[destination + component] = previous[offset + component]; }
let parent = previous[offset];
if (parent != 0xffffffffu) {
 let ancestor = previousOffset + parent * 4u;
 next[destination] = previous[ancestor];
 next[destination + 1u] = bitcast<u32>(bitcast<f32>(previous[offset + 1u]) + bitcast<f32>(previous[ancestor + 1u]));
 next[destination + 2u] = min(${count}u, previous[offset + 2u] + previous[ancestor + 2u]);
 next[destination + 3u] = previous[ancestor + 3u];
}`
        })
      );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-profiles`,
        operation: 'GPUTerrainCatchmentSummary',
        invocationCount: count,
        declarations,
        bindings: [
          ...inputs,
          terrainBinding('state', states[rounds % 2]),
          terrainBinding('profiles', props.output.profiles, 'read_write'),
          terrainBinding('status', props.output.profileStatus, 'read_write')
        ],
        body: `let center = elevation[elevationOffset + index]; let offset = stateOffset + index * 4u;
let distance = bitcast<f32>(state[offset + 1u]); var gradient = 0.0;
var result = select(2u, 1u, state[offset] == NO_CELL && state[offset + 3u] == 1u && state[offset + 2u] < ${props.maximumSteps ?? count}u && finite(distance));
if (!finite(center) || basins[basinsOffset + index] >= ${props.basinCount}u || streams[streamsOffset + index] != 1u) { result = select(0u, 2u, streams[streamsOffset + index] > 1u); }
let receiver = getFlowReceiver(index, directions[directionsOffset + index]);
if (receiver != NO_CELL && streams[streamsOffset + receiver] == 1u) { gradient = (center - elevation[elevationOffset + receiver]) / edgeLength(index, receiver); }
status[statusOffset + index] = result;
profiles[profilesOffset + index * 3u] = select(missing(), distance, result == 1u);
profiles[profilesOffset + index * 3u + 1u] = center;
profiles[profilesOffset + index * 3u + 2u] = select(missing(), gradient, result == 1u);`
      })
    );
    return nodes;
  }
}
