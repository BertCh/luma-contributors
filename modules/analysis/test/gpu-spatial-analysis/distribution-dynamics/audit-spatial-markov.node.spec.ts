// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-spatial-markov';
import {GPULISAMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-lisa-markov';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getGPUClassBreaksParameterValues} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {getGPUSpatialAutocorrelationParameterValues} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getPaddedPeriodStride} from '../../../src/gpu-spatial-analysis/distribution-dynamics/period-views';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('rounds period strides to the next 64-element boundary', () => {
  // ceil(rows/64)*64, including an exact boundary and the first row past it.
  expect([0, 1, 8, 63, 64, 65, 127, 128].map(getPaddedPeriodStride)).toEqual([
    0, 64, 64, 64, 64, 128, 128, 128
  ]);
});

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `audit-${serial++}`, format, length);
  return {
    device,
    graph,
    spatial: {
      values: view('float32', 24),
      classes: view('uint32', 24),
      classCount: 2,
      weights: {
        offsets: view('uint32', 9),
        neighbors: view('uint32', 6),
        weights: view('float32', 6),
        status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}
      },
      periodCount: 3,
      maximumLagClassCount: 2,
      lagParameters: view(
        'float32',
        getGPUClassBreaksParameterValues({method: 'quantile', classCount: 2}, 2).length
      ),
      output: {
        counts: view('uint32', 8),
        probabilities: view('float32', 8),
        rowTotals: view('uint32', 4),
        lagBreaks: view('float32', 3),
        lagClassCount: view('uint32', 1)
      }
    },
    lisa: {
      values: view('float32', 24),
      periodCount: 3,
      weights: {
        offsets: view('uint32', 9),
        neighbors: view('uint32', 6),
        weights: view('float32', 6)
      },
      parameters: view('float32', getGPUSpatialAutocorrelationParameterValues({}).length),
      output: {counts: view('uint32', 25)}
    }
  };
}

it('preserves both neighborhood loss flags through every period of spatial Markov', () => {
  const {device, graph, spatial} = createFixture();
  try {
    const nodes = new GPUSpatialMarkov(spatial).getCommandNodes(graph);
    for (let period = 0; period < spatial.periodCount; period++) {
      const readers = nodes.filter(node => node.id.startsWith(`spatial-markov-lag-${period}-`));
      for (const flag of Object.values(spatial.weights.status)) {
        expect(
          readers.some(node =>
            getWGSLKernelNodeRecord(graph, node.id)?.bindings.some(binding => binding.view === flag)
          ),
          `period ${period} reads both loss flags`
        ).toBe(true);
      }
    }
    const guards = nodes.filter(node => node.id.includes('-incomplete-'));
    const output = spatial.output;
    const writeStaleResults = (memory: ReturnType<typeof createCPUGraphMemory>) => {
      memory.write(output.counts, new Uint32Array(8).fill(42));
      memory.write(output.probabilities, new Float32Array(8).fill(0.5));
      memory.write(output.rowTotals, new Uint32Array(4).fill(42));
      memory.write(output.lagBreaks, new Float32Array(3).fill(42));
      memory.write(output.lagClassCount, Uint32Array.of(42));
    };
    // Independent final-guard check (policy, not the producer): either flag alone must replace
    // stale results with zero counts, row totals and class count and NaN probabilities and edges.
    for (const failedFlag of Object.values(spatial.weights.status)) {
      const memory = createCPUGraphMemory();
      for (const flag of Object.values(spatial.weights.status))
        memory.write(flag, Uint32Array.of(Number(flag === failedFlag)));
      writeStaleResults(memory);
      executeCommandNodesOnCPU(guards, memory);
      expect(Array.from(memory.readUint32(output.counts))).toEqual(new Array(8).fill(0));
      expect(Array.from(memory.readUint32(output.rowTotals))).toEqual([0, 0, 0, 0]);
      expect(Array.from(memory.readUint32(output.lagClassCount))).toEqual([0]);
      const probabilities = Array.from(memory.readFloat32(output.probabilities));
      expect(probabilities).toHaveLength(8);
      expect(probabilities.every(Number.isNaN), 'eight NaN probabilities').toBe(true);
      const breaks = Array.from(memory.readFloat32(output.lagBreaks));
      expect(breaks).toHaveLength(3);
      expect(breaks.every(Number.isNaN)).toBe(true);
    }
    // Control: complete weights leave the ordinary writers' results untouched.
    const memory = createCPUGraphMemory();
    for (const flag of Object.values(spatial.weights.status)) memory.write(flag, Uint32Array.of(0));
    writeStaleResults(memory);
    executeCommandNodesOnCPU(guards, memory);
    expect(Array.from(memory.readUint32(output.counts))).toEqual(new Array(8).fill(42));
    expect(Array.from(memory.readFloat32(output.probabilities))).toEqual(new Array(8).fill(0.5));
    expect(Array.from(memory.readUint32(output.rowTotals))).toEqual(new Array(4).fill(42));
    expect(Array.from(memory.readUint32(output.lagClassCount))).toEqual([42]);
  } finally {
    device.destroy();
  }
});

it('refuses invalid LISA gating rather than silently selecting significance gating', () => {
  const {device, lisa} = createFixture();
  try {
    expect(() => new GPULISAMarkov({...lisa, gating: 'invalid' as 'none'})).toThrow(
      /lisa-markov gating/
    );
  } finally {
    device.destroy();
  }
});

it('rejects undersized transition outputs in both parent constructors without graph lowering', () => {
  const {device, graph, spatial, lisa} = createFixture();
  try {
    expect(
      () =>
        new GPUSpatialMarkov({
          ...spatial,
          output: {
            ...spatial.output,
            counts: createTransientView(graph, 'short-spatial', 'uint32', 7)
          }
        })
    ).toThrow(/output.counts/);
    expect(
      () =>
        new GPULISAMarkov({
          ...lisa,
          output: {counts: createTransientView(graph, 'short-lisa', 'uint32', 24)}
        })
    ).toThrow(/output.counts/);
    for (const [name, format, spatialLength, lisaLength] of [
      ['probabilities', 'float32', 7, 24],
      ['rowTotals', 'uint32', 3, 4],
      ['ignored', 'uint32', 0, 0]
    ] as const) {
      expect(
        () =>
          new GPUSpatialMarkov({
            ...spatial,
            output: {
              ...spatial.output,
              [name]: createTransientView(graph, `spatial-${name}`, format, spatialLength)
            }
          })
      ).toThrow(new RegExp(`output.${name}`));
      expect(
        () =>
          new GPULISAMarkov({
            ...lisa,
            output: {
              ...lisa.output,
              [name]: createTransientView(graph, `lisa-${name}`, format, lisaLength)
            }
          })
      ).toThrow(new RegExp(`output.${name}`));
    }
  } finally {
    device.destroy();
  }
});

it('rejects disjoint transition output ranges backed by one buffer in both constructors', () => {
  const {device, graph, spatial, lisa} = createFixture();
  try {
    const storage = createTransientView(graph, 'shared-transition-storage', 'uint32', 100);
    const counts = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 25,
      byteOffset: 0
    });
    const rowTotals = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 5,
      byteOffset: 256
    });
    expect(
      () => new GPUSpatialMarkov({...spatial, output: {...spatial.output, counts, rowTotals}})
    ).toThrow(/outputs must use separate buffers/);
    expect(() => new GPULISAMarkov({...lisa, output: {counts, rowTotals}})).toThrow(
      /outputs must use separate buffers/
    );
  } finally {
    device.destroy();
  }
});

for (const name of ['overflow', 'candidateOverflow'] as const) {
  it(`rejects LISA output aliasing its input ${name} flag at construction`, () => {
    const {device, graph, lisa} = createFixture();
    try {
      const flag = createTransientView(graph, `loss-${name}`, 'uint32', 1);
      expect(
        () =>
          new GPULISAMarkov({
            ...lisa,
            weights: {...lisa.weights, status: {[name]: flag}},
            output: {...lisa.output, ignored: flag}
          })
      ).toThrow(/must not share buffers with inputs/);
    } finally {
      device.destroy();
    }
  });
}
