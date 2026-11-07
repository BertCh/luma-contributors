// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkStatistics: a packed summary of a network (component counts, degree extremes and degree
// histograms). The bars are the total-degree histogram from the summary the operation wrote;
// closing roads moves nodes to lower degrees.

import {
  GPUNetworkStatistics,
  GPU_NETWORK_STATISTICS_HEADER_LENGTH,
  getGPUNetworkStatisticsLength
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addSlice} from './shared/display';
import {createRoadNetwork, createRoads, uploadRoadTopology} from './shared/roads';
import {parameters} from './network-statistics.parameters';

const BIN_COUNT = 12;

export default defineOperationRun({
  parameters,
  caption: 'Histogram of node degree from the packed network summary.',
  createInputs: ({random, world}) => ({base: createRoads(random, world, 14)}),
  build(builder, {base}, values) {
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
    return {
      outputs: {
        totalDegree: addSlice(builder, {
          id: 'total-degree',
          source: summary,
          offset: GPU_NETWORK_STATISTICS_HEADER_LENGTH + 2 * BIN_COUNT,
          length: BIN_COUNT
        })
      }
    };
  },
  rebuildOn: ['degreeBinning', 'closures'],
  draw: ({outputs}) => [
    {layer: 'chart', kind: 'bars', values: outputs.totalDegree, ramp: 'sequential'}
  ]
});
