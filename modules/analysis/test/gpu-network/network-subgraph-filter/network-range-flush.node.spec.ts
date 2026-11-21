// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUNetworkSubgraphFilter} from '../../../src/gpu-network/network-subgraph-filter/gpu-network-subgraph-filter';
import {getGPUNetworkSubgraphFilterParameterValues} from '../../../src/gpu-network/network-subgraph-filter/network-subgraph-filter-parameters';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';
import {executeSpatialAuditKernel} from '../../gpu-spatial-analysis/spatial-validation/spatial-design-audit.test-utils';

it.each([
  'vertex-columns-0',
  'edge-times'
] as const)('zero rows obey subnormal bounds when float32 comparisons flush: %s (S3)', target => {
  const device = createNullWebGPUDevice();
  const context = createOracleContext(device);
  try {
    const {graph, input, output} = context;
    recordWGSLKernelNodes(graph);
    const parameters = getGPUNetworkSubgraphFilterParameterValues(
      {nodeColumnCount: 1, edgeColumnCount: 0, hasEdgeTimes: true},
      {nodeAttributeRanges: [[0, 2 ** -151]], closedEdgeTimeWindow: [-1, -(2 ** -151)]}
    );
    const nodes = new GPUNetworkSubgraphFilter({
      id: 'flush',
      directed: true,
      topology: {
        offsets: input('offsets', 'uint32', [0, 1, 1]),
        neighbors: input('neighbors', 'uint32', [1])
      },
      nodeColumns: [input('column', 'float32', [0, -0])],
      edgeTimes: input('time', 'float32', [0]),
      parameters: input('parameters', 'float32', Array.from(parameters)),
      output: {
        nodeMask: output('node-mask', 'uint32', 2),
        edgeMask: output('edge-mask', 'uint32', 1)
      }
    }).getCommandNodes(graph);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    for (const [suffix, expected] of [
      ['vertex-columns-0', [1, 1]],
      ['edge-times', [0]]
    ] as const) {
      if (suffix !== target) continue;
      const kernel = kernels.find(kernel => kernel.id === `flush-${suffix}`)!;
      const record = getWGSLKernelNodeRecord(graph, kernel.id)!;
      const maskOut = Uint32Array.of(...expected.map(() => 1));
      const arrays: Record<string, Float32Array | Uint32Array> = {
        maskOut,
        parameters,
        column0: Float32Array.of(0, -0),
        times: Float32Array.of(0)
      };
      for (const binding of record.bindings) {
        const values = arrays[binding.name];
        if (binding.type === 'u32' && values instanceof Float32Array)
          arrays[binding.name] = new Uint32Array(values.buffer);
      }
      executeSpatialAuditKernel(kernel, arrays, {flushSubnormalComparisons: true});
      // 0 belongs to [0, 2^-151), but 0 exceeds the closed time end -2^-151.
      expect(Array.from(maskOut), suffix).toEqual(expected);
    }
  } finally {
    context.destroy();
    device.destroy();
  }
});
