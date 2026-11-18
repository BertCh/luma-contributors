// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUHuffTradeAreas} from '../../../../analysis/src/gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas';
import {recordWGSLKernelNodes} from '../../../../analysis/src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createGPUComputationRecord,
  describeGPUCommandGraph
} from '../../../../analysis/src/computation-record/index';
import {createNullWebGPUDevice} from '../../../../analysis/src/testing/gpu-contributor-test-utils';
import {ACCESSIBILITY_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/accessibility';
import {resolveSymbolBinding} from '../../../src/computation-views/views/formula-tex';

it('resolves the curated Huff denominator against a freshly compiled Huff record', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  Object.defineProperty(device, 'createComputePipeline', {
    value: () => ({hash: '', destroy() {}})
  });
  const graph = new GPUCommandGraph(device, {id: 'fresh-huff'});
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    recordWGSLKernelNodes(graph);
    graph.add(
      new GPUHuffTradeAreas({
        attractiveness: createTransientView(graph, 'attractiveness', 'float32', 2),
        demandWeights: {
          offsets: createTransientView(graph, 'offsets', 'uint32', 2),
          neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
          weights: createTransientView(graph, 'weights', 'float32', 2)
        },
        output: {
          tradeArea: createTransientView(graph, 'trade-area', 'uint32', 1),
          probabilities: createTransientView(graph, 'probabilities', 'float32', 2)
        }
      })
    );
    compiled = graph.compile();
    const formula = ACCESSIBILITY_OPERATORS['GPUHuffTradeAreas'].formula;
    if (!('symbols' in formula)) throw new Error('Huff needs a curated formula');
    const symbol = formula.symbols?.find(candidate => candidate.id === 'denominator');
    expect(symbol).toBeDefined();
    const record = createGPUComputationRecord({
      id: 'fresh-huff-record',
      adapter: {backend: 'webgpu'},
      domains: ['spatial'],
      graphs: [describeGPUCommandGraph(compiled, {graph, sourceText: true})],
      formulas: [formula]
    });
    const resolved = resolveSymbolBinding(record, symbol!);
    expect(resolved).toMatchObject({status: 'bound', kind: 'wgsl', target: 'total'});
    if (resolved.status === 'bound') {
      expect(resolved.nodeIds.some(node => node.endsWith('-trade-area'))).toBe(true);
      expect(resolved.nodeIds.some(node => node.endsWith('-probabilities'))).toBe(true);
    }
  } finally {
    compiled?.destroy();
    device.destroy();
  }
});
