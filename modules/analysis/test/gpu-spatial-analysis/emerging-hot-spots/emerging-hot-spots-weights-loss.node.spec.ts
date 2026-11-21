// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {recordWGSLKernelNodes} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {GPUEmergingHotSpots} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const minimumTrendSliceCount of [0, 2])
  it(`AMG-19 aggregates live loss before inference and recovers, minimum trend=${minimumTrendSliceCount}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32' | 'sint32'>(format: Format, length: number) =>
      createTransientView(graph, `loss-${serial++}`, format, length);
    try {
      const overflow = view('uint32', 1);
      const candidateOverflow = view('uint32', 1);
      const parameters = view('float32', 8);
      const output = {
        giZScores: view('float32', 6),
        pValues: view('float32', 6),
        adjustedPValues: view('float32', 6),
        lastValidSlice: view('uint32', 3),
        trendZ: view('float32', 3),
        trendP: view('float32', 3),
        trendS: view('sint32', 3),
        category: view('uint32', 3),
        hotSliceCount: view('uint32', 3),
        coldSliceCount: view('uint32', 3),
        globalStatistics: view('float32', 4),
        includedCount: view('uint32', 1)
      };
      const nodes = new GPUEmergingHotSpots({
        values: view('float32', 6),
        weights: {
          offsets: view('uint32', 4),
          neighbors: view('uint32', 2),
          weights: view('float32', 2),
          status: {overflow, candidateOverflow}
        },
        sliceCount: 2,
        multipleTesting: 'none',
        parameters,
        output
      }).getCommandNodes(graph);
      const tail = nodes.slice(nodes.findIndex(node => node.id.includes('-incomplete-')));
      // Two flag-aggregation nodes and one Gi* suppression; downstream writers publish once.
      expect(tail.filter(node => node.id.includes('-incomplete-'))).toHaveLength(3);
      const memory = createCPUGraphMemory();
      memory.write(
        parameters,
        Float32Array.of(0, 0, 1.96, 0.05, 0.9, minimumTrendSliceCount, 0.05, 0)
      );
      // These independent population outputs are retained from the moment passes.
      memory.write(output.globalStatistics, Float32Array.of(6, 2, 1, 1));
      memory.write(output.includedCount, Uint32Array.of(6));
      for (const [finalLoss, candidateLoss] of [
        [0, 0],
        [0, 1],
        [0, 0],
        [1, 0],
        [0, 0]
      ]) {
        memory.write(overflow, Uint32Array.of(finalLoss));
        memory.write(candidateOverflow, Uint32Array.of(candidateLoss));
        // A monotone two-slice cell, its cold mirror and an ordinary entirely missing cell.
        memory.write(output.giZScores, Float32Array.of(2, 3, -2, -3, NaN, NaN));
        executeCommandNodesOnCPU(tail, memory);
        const lost = Boolean(finalLoss || candidateLoss);
        expect(Array.from(memory.readUint32(output.category))).toEqual(
          lost ? [17, 17, 17] : [4, 12, 0]
        );
        expect(Array.from(memory.readUint32(output.hotSliceCount))).toEqual(
          lost ? [0, 0, 0] : [2, 0, 0]
        );
        expect(Array.from(memory.readUint32(output.coldSliceCount))).toEqual(
          lost ? [0, 0, 0] : [0, 2, 0]
        );
        expect(Array.from(memory.readUint32(output.lastValidSlice))).toEqual(
          lost ? [0xffffffff, 0xffffffff, 0xffffffff] : [1, 1, 0xffffffff]
        );
        // n=2: S=+/-1, continuity correction gives z=0 and p=1. Missing trends obey the minimum.
        expectNumericArrayClose(
          memory.readFloat32(output.trendZ),
          lost ? [NaN, NaN, NaN] : [0, 0, minimumTrendSliceCount === 0 ? 0 : NaN]
        );
        expectNumericArrayClose(
          memory.readFloat32(output.trendP),
          lost ? [NaN, NaN, NaN] : [1, 1, minimumTrendSliceCount === 0 ? 1 : NaN]
        );
        expectNumericArrayClose(
          memory.readFloat32(output.giZScores),
          lost ? [NaN, NaN, NaN, NaN, NaN, NaN] : [2, 3, -2, -3, NaN, NaN]
        );
        // erfc(2/sqrt(2)) and erfc(3/sqrt(2)), independent normal-tail literals.
        for (const port of [output.pValues, output.adjustedPValues])
          expectNumericArrayClose(
            memory.readFloat32(port),
            lost
              ? [NaN, NaN, NaN, NaN, NaN, NaN]
              : [
                  0.0455002638963584,
                  0.00269979606326019,
                  0.0455002638963584,
                  0.00269979606326019,
                  NaN,
                  NaN
                ],
            {absoluteTolerance: 1e-7}
          );
        expect(Array.from(memory.readUint32(output.trendS))).toEqual(
          lost ? [0, 0, 0] : [1, 0xffffffff, 0]
        );
        expect(Array.from(memory.readFloat32(output.globalStatistics))).toEqual([6, 2, 1, 1]);
        expect(Array.from(memory.readUint32(output.includedCount))).toEqual([6]);
      }
    } finally {
      device.destroy();
    }
  });

it('S4 full-array comparisons reject empty, excess and incorrect third-row reads', () => {
  for (const actual of [[], [NaN, NaN, NaN, NaN], [NaN, NaN, 0]])
    expect(() => expectNumericArrayClose(actual, [NaN, NaN, NaN])).toThrow();
  expect(() => expectNumericArrayClose([0, 0, 99], [0, 0, 0])).toThrow();
});
