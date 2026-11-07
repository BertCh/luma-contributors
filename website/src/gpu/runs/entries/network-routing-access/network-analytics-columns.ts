// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkAnalyticsColumns: degree, normalized degree and connected components as per-node
// columns. Closing roads splits the network into components.

import {GPUNetworkAnalyticsColumns} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {
  createRoadNetwork,
  createRoads,
  getRoadContext,
  getRoadValues,
  uploadRoadTopology,
  uploadRoads
} from './shared/roads';
import {parameters} from './network-analytics-columns.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Per-node analytics columns: component label, degree, normalized degree.',
  createInputs: ({random, world}) => ({base: createRoads(random, world)}),
  build(builder, {base}, values) {
    const network = createRoadNetwork(base, values.closures);
    const {topology} = uploadRoadTopology(builder, network);
    const degree = builder.allocate('degree', 'uint32', network.nodeCount);
    const normalized = builder.allocate('degree-normalized', 'float32', network.nodeCount);
    const components = builder.allocate('components', 'uint32', network.nodeCount);
    builder.graph.add(
      new GPUNetworkAnalyticsColumns({
        topology,
        reverseTopology: topology,
        degree: {output: degree.view, normalized: normalized.view},
        components: {output: components.view, maximumIterations: 64}
      })
    );
    return {outputs: {...uploadRoads(builder, network), degree, normalized, components}};
  },
  draw: ({outputs, values}) => {
    const columns = [outputs.components, outputs.degree, outputs.normalized];
    return [
      getRoadContext(outputs),
      getRoadValues(outputs, columns[values.column], {
        radius: 2.2,
        ramp: values.column === 0 ? 'categorical' : 'sequential',
        range: values.column === 1 ? [0, 5] : values.column === 2 ? [0, 1] : undefined
      })
    ];
  }
});
