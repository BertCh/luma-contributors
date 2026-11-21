// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {
  GPUGlobalPermutationTest,
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  getGPUPermutationParameterValues
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {PERMUTATION_TOTAL} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-inference-kernels';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  type RecordedGPUKernelSource
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  findStatisticsBarrierRisks,
  findStatisticsNonFiniteConstants
} from '../spatial-weights/statistics-wgsl-audit';

function execute(
  record: RecordedGPUKernelSource,
  buffers: Record<string, Float32Array | Uint32Array>
) {
  // The interpreter does not execute phony pointer expressions or resolve overrides itself.
  const source = record.source
    .replace(/_ = [^;]+;/g, '')
    .replace(
      /override (\w+): u32;/g,
      (_, name: string) => `const ${name}: u32 = ${record.constants[name]}u;`
    );
  const bindings: Record<number, Float32Array | Uint32Array> = {};
  for (const resource of new WgslReflect(source).storage) {
    expect(buffers[resource.name], resource.name).toBeDefined();
    bindings[resource.binding] = buffers[resource.name];
  }
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {0: bindings});
}

for (const permutationBatchSize of [undefined, 2]) {
  it(`global G publishes invalid results with exact-count scaling (${permutationBatchSize ? 'materialized' : 'inline'})`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `negative-publication-${serial++}`, format, length);
    try {
      const records = getGPUCommandNodeKernelSources(
        new GPUGlobalPermutationTest({
          id: 'negative-publication',
          weights: {
            offsets: view('uint32', 5),
            neighbors: view('uint32', 4),
            weights: view('float32', 4)
          },
          values: view('float32', 4),
          parameters: view('uint32', 4),
          statistic: 'getisOrdG',
          maximumPermutations: 3,
          permutationBatchSize,
          output: {
            summary: view('float32', 12),
            referenceDistribution: view('float32', 3),
            histogram: view('uint32', 4),
            status: {invalidCount: view('uint32', 1)}
          }
        }).getCommandNodes(graph)
      );
      const reject = records.find(record => record.id.endsWith('-reject-negative'))!;
      const finalize = records.find(record => record.id === 'negative-publication-finalize')!;
      for (const record of records) {
        expect(findReservedIdentifiers(stripWGSLComments(record.source)), record.id).toEqual([]);
        expect(findStatisticsNonFiniteConstants(record.source), record.id).toEqual([]);
        expect(findStatisticsBarrierRisks(record.source), record.id).toEqual([]);
      }
      const totals = new Float32Array(PERMUTATION_TOTAL.offDiagonalMass + 1);
      const results = new Float32Array(12);
      const referenceDistribution = new Float32Array(3);
      const histogram = new Uint32Array(4);
      const negativeCount = new Uint32Array(1);
      const positiveCount = new Uint32Array(1);
      const invalidCount = new Uint32Array(1);
      const buffers = {
        totals,
        results,
        referenceDistribution,
        histogram,
        negativeCount,
        positiveCount,
        invalidCount,
        pairSums: Float32Array.of(20, 15, 25, 30),
        rowPositions: Uint32Array.of(0, 1, 2, 3, 4),
        islandCount: new Uint32Array(1),
        parameters: getGPUPermutationParameterValues({seed: 7, permutationCount: 3})
      };
      for (const rejectedRows of [0, 3, 0]) {
        // Earlier graph stages overwrite the moments before the rejection/publication passes.
        totals[PERMUTATION_TOTAL.count] = 4;
        totals[PERMUTATION_TOTAL.mean] = 2.5;
        totals[PERMUTATION_TOTAL.sumSquares] = 5;
        // For the valid values [1,2,3,4], sum_i!=j xi*xj = 10^2 - 30 = 70.
        // The current General-G route reads this explicit mass instead of reconstructing it.
        totals[PERMUTATION_TOTAL.offDiagonalMass] = 70;
        negativeCount[0] = rejectedRows;
        positiveCount[0] = 4 - rejectedRows;
        histogram.fill(123);
        execute(reject, buffers);
        execute(finalize, buffers);
        expect(invalidCount[0]).toBe(rejectedRows);
        if (rejectedRows) {
          for (const field of [
            'observed',
            'pseudoPValue',
            'exceedances',
            'simulatedMean',
            'simulatedStandardDeviation',
            'zSimulated',
            'pZSimulated',
            'minimum',
            'maximum'
          ] as const)
            expect(results[GPU_GLOBAL_PERMUTATION_SUMMARY[field]], field).toBeNaN();
          expect(Array.from(referenceDistribution).every(Number.isNaN)).toBe(true);
          expect(Array.from(histogram)).toEqual([0, 0, 0, 0]);
        } else {
          // Four values 1,2,3,4 have sum over ordered distinct pairs 70; observed pair sum is 20.
          expect(results[GPU_GLOBAL_PERMUTATION_SUMMARY.observed]).toBeCloseTo(2 / 7, 6);
        }
        expect(results[GPU_GLOBAL_PERMUTATION_SUMMARY.analysisCount]).toBe(4);
      }
    } finally {
      device.destroy();
    }
  });
}
