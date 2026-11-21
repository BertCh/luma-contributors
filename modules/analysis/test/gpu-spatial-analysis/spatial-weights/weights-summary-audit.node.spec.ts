// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsSummary} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-summary';
import {createWeights7TransientChunks} from './weights7-chunk-harness';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('summary gates every output on matrix and chunk loss flags', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const weights = createWeights7TransientChunks(graph, [3, 5]);
    weights.status = {overflow: createTransientView(graph, 'matrix-loss', 'uint32', 1)};
    weights.chunks[1].weights.status = {
      candidateOverflow: createTransientView(graph, 'chunk-loss', 'uint32', 1)
    };
    const nodes = new GPUSpatialWeightsSummary({
      weights,
      output: {
        statistics: createTransientView(graph, 'statistics', 'float32', 5),
        counts: createTransientView(graph, 'counts', 'uint32', 8),
        cardinality: createTransientView(graph, 'cardinality', 'uint32', 8)
      }
    }).getCommandNodes(graph);
    const gates = getGPUCommandNodeKernelSources(nodes).filter(kernel =>
      kernel.id.includes('-incomplete-')
    );
    // Every distinct runtime loss source must gate every dependent public port.
    expect(gates).toHaveLength(6);
    for (const {source} of gates) {
      expect(source).toContain('loss[lossOffset] != 0u');
      // Separate writable bindings preserve disjoint packed output views that
      // share an aligned range but are never co-bound by the ordinary kernels.
      expect(source.match(/var<storage, read_write>/g)).toHaveLength(1);
    }
    for (const [name, assignment, length] of [
      ['statistics', 'getQuietNaN(index)', 3],
      ['counts', '0u', 6],
      ['cardinality', '0u', 8]
    ] as const) {
      const outputGates = gates.filter(kernel => kernel.id.includes(`-${name}-incomplete-`));
      expect(outputGates).toHaveLength(2);
      for (const {source} of outputGates) {
        expect(source).toContain(`${name}[${name}Offset + index] = ${assignment}`);
        expect(source).toContain(`index < ${length}u`);
      }
    }
    expect(nodes.slice(-6).map(node => node.id)).toEqual(gates.map(kernel => kernel.id));
  } finally {
    device.destroy();
  }
});
