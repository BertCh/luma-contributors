// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkStatistics: a packed summary of a network (component counts, degree extremes and degree
// histograms). The bars are the out-degree histogram from the summary the operation wrote;
// closing roads moves nodes to lower degrees.

import {
  GPUNetworkStatistics,
  GPU_NETWORK_STATISTICS_HEADER_LENGTH,
  getGPUNetworkStatisticsLength
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {createPointIndex, getChartPosition, getChartWorldX, type PointIndex} from '../../pick';
import type {RunGlyph} from '../../interaction';
import {addSlice} from './shared/display';
import {createRoadNetwork, createRoads, getRoadContext, uploadRoadTopology} from './shared/roads';
import {parameters} from './network-statistics.parameters';

const BIN_COUNT = 12;
const RECTANGLE = [0.06, 0.05, 0.94, 0.3] as const;
// The network is drawn in the upper part of the picture, the histogram strip below it.
const MAP_SCALE = 0.6;
const MAP_OFFSET_Y = 0.38;

function getBin(degree: number, isLogarithmic: boolean): number {
  if (isLogarithmic)
    return degree === 0 ? 0 : Math.min(1 + Math.floor(Math.log2(degree)), BIN_COUNT - 1);
  return Math.min(degree, BIN_COUNT - 1);
}

type View = {bins: Uint8Array; counts: number[]; index: PointIndex};

export default defineOperationRun({
  parameters,
  caption: 'Histogram of node degree from the packed network summary, linked to the intersections.',
  createInputs: ({random, world}) => ({
    base: createRoads(random, world, 14),
    world,
    view: null as View | null,
    bin: 3
  }),
  build(builder, inputs, values) {
    const {base, world} = inputs;
    const network = createRoadNetwork(base, values.closures);
    const {topology} = uploadRoadTopology(builder, network);
    const summary = builder.allocate('summary', 'uint32', getGPUNetworkStatisticsLength(BIN_COUNT));
    builder.graph.add(
      new GPUNetworkStatistics({
        topology,
        degreeBinCount: BIN_COUNT,
        degreeBinning: values.degreeBinning === 0 ? 'linear' : 'log2',
        output: summary.view
      })
    );
    // CPU copy of the same binning, to link intersections and bars.
    const isLogarithmic = values.degreeBinning === 1;
    const displayNodes = Float32Array.from(network.nodes, (value, index) =>
      index % 2 === 0
        ? (world.width * (1 - MAP_SCALE)) / 2 + value * MAP_SCALE
        : MAP_OFFSET_Y + value * MAP_SCALE
    );
    const bins = Uint8Array.from({length: network.nodeCount}, (_, node) =>
      getBin(network.offsets[node + 1] - network.offsets[node], isLogarithmic)
    );
    const counts = new Array<number>(BIN_COUNT).fill(0);
    for (const bin of bins) counts[bin]++;
    inputs.view = {bins, counts, index: createPointIndex(displayNodes, network.nodeCount)};
    return {
      outputs: {
        nodes: builder.upload('display-nodes', 'float32x2', displayNodes),
        edges: builder.upload('road-edges', 'uint32x2', network.edgePairs),
        selection: builder.allocate('selection', 'uint32', network.nodeCount),
        hoverBars: builder.allocate('hover-bars', 'float32', BIN_COUNT),
        totalDegree: addSlice(builder, {
          id: 'out-degree',
          source: summary,
          offset: GPU_NETWORK_STATISTICS_HEADER_LENGTH,
          length: BIN_COUNT
        })
      }
    };
  },
  rebuildOn: ['degreeBinning', 'closures'],
  // Probe: a bar (or an intersection) picks a degree class; the matching intersections light up.
  interaction: {
    kind: 'probe',
    hint: 'hover a bar',
    rest: ({world, toUnit}) => toUnit([getChartWorldX(RECTANGLE, world, 3.5 / BIN_COUNT), 0.17]),
    resolve(pointer, {inputs, outputs, world}) {
      const view = inputs.view;
      if (!view) return {};
      let bin = inputs.bin;
      const chart = getChartPosition(RECTANGLE, world, pointer.x, pointer.y);
      if (chart.isInside || pointer.y < MAP_OFFSET_Y) {
        bin = Math.min(BIN_COUNT - 1, Math.floor(chart.u * BIN_COUNT));
      } else {
        const node = view.index.nearest(pointer.x, pointer.y);
        if (node >= 0) bin = view.bins[node];
      }
      inputs.bin = bin;
      outputs.hoverBars.write(
        Float32Array.from(view.counts, (count, index) => (index === bin ? count : 0))
      );
      outputs.selection.write(
        Uint32Array.from(view.bins, nodeBin => (nodeBin === bin ? 0 : 0xffffffff))
      );
      const left = getChartWorldX(RECTANGLE, world, bin / BIN_COUNT);
      const right = getChartWorldX(RECTANGLE, world, (bin + 1) / BIN_COUNT);
      const glyphs: RunGlyph[] = [
        {
          shape: 'box',
          minimum: [left, RECTANGLE[1] * world.height],
          maximum: [right, RECTANGLE[3] * world.height]
        }
      ];
      return {
        spotlight: null,
        glyphs,
        readout: `degree class ${bin} · ${view.counts[bin]} intersections`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => {
    const maximum = Math.max(1, ...(inputs.view?.counts ?? [1]));
    return [
      getRoadContext(outputs),
      {
        layer: 'dots',
        positions: outputs.nodes,
        values: outputs.selection,
        ramp: 'categorical',
        categories: [[...theme.categories[1], 1]],
        radius: 4.5
      },
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.totalDegree,
        ramp: 'sequential',
        range: [0, maximum],
        rectangle: RECTANGLE
      },
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.hoverBars,
        color: [...theme.ink, 1],
        range: [0, maximum],
        rectangle: RECTANGLE
      }
    ];
  }
});
