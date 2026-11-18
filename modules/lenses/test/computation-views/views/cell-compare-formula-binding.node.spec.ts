// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCellTableCompare} from '../../../../analysis/src/gpu-spatial-analysis/cell-table-compare/index';
import type {GPUCellTable} from '../../../../analysis/src/gpu-spatial-analysis/cell-aggregation/index';
import {recordWGSLKernelNodes} from '../../../../analysis/src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createGPUComputationRecord,
  describeGPUCommandGraph
} from '../../../../analysis/src/computation-record/index';
import {createNullWebGPUDevice} from '../../../../analysis/src/testing/gpu-contributor-test-utils';
import {CELLS_FORMULAS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/cells-formulas';
import {resolveSymbolBinding} from '../../../src/computation-views/views/formula-tex';

it('binds cell comparison decoding to the signed full-width difference in a fresh record', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  Object.defineProperty(device, 'createComputePipeline', {
    value: () => ({hash: '', destroy() {}})
  });
  const graph = new GPUCommandGraph(device, {id: 'fresh-cell-compare'});
  const createTable = (name: string): GPUCellTable => ({
    cells: createTransientView(graph, `${name}-cells`, 'uint32x2', 1),
    counts: createTransientView(graph, `${name}-counts`, 'uint32', 1),
    sums: createTransientView(graph, `${name}-sums`, 'uint32x2', 1),
    status: {
      count: createTransientView(graph, `${name}-count`, 'uint32', 1),
      overflow: createTransientView(graph, `${name}-overflow`, 'uint32', 1),
      candidateOverflow: createTransientView(graph, `${name}-candidate-overflow`, 'uint32', 1)
    }
  });
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    recordWGSLKernelNodes(graph);
    graph.add(
      new GPUCellTableCompare({
        measure: 'sum',
        sumScale: 1,
        before: createTable('before'),
        after: createTable('after'),
        output: {
          cells: createTransientView(graph, 'cells', 'uint32x2', 1),
          delta: createTransientView(graph, 'delta', 'float32', 1),
          status: {
            count: createTransientView(graph, 'count', 'uint32', 1),
            overflow: createTransientView(graph, 'overflow', 'uint32', 1),
            candidateOverflow: createTransientView(graph, 'candidate-overflow', 'uint32', 1)
          }
        }
      })
    );
    compiled = graph.compile();
    const formula = CELLS_FORMULAS.compare;
    const symbol = formula.symbols.find(candidate => candidate.id === 'decode');
    expect(symbol).toBeDefined();
    const record = createGPUComputationRecord({
      id: 'fresh-cell-compare-record',
      adapter: {backend: 'webgpu'},
      domains: ['spatial'],
      graphs: [describeGPUCommandGraph(compiled, {graph, sourceText: true})],
      formulas: [formula]
    });
    // cellI64ToF32 reads bit 63 as a sign, so it cannot decode the mathematical delta +2^63.
    // The formula must highlight the sign-applied float difference instead of that input decoder.
    const resolved = resolveSymbolBinding(record, symbol!);
    expect(resolved).toMatchObject({status: 'bound', kind: 'wgsl', target: 'difference'});
    if (resolved.status === 'bound') {
      expect(resolved.nodeIds.some(node => node.endsWith('-measures'))).toBe(true);
    }
  } finally {
    compiled?.destroy();
    device.destroy();
  }
});
