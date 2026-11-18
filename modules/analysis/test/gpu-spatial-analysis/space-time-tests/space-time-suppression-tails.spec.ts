// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUKnoxTest,
  GPUMantelTest,
  GPUJacquezTest,
  getGPUSpaceTimeParameterValues,
  GPU_SPACE_TIME_SUMMARY
} from '../../../src/gpu-spatial-analysis/space-time-tests/index';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const kind of ['knox', 'mantel', 'jacquez'] as const)
  it(`final3 ${kind} preserves summary and draw-count spare words across loss and recovery`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const loss = new GPUParameterBuffer(device, {
      id: 'loss',
      format: 'uint32',
      length: 1,
      values: Uint32Array.of(0)
    });
    const summaryBuffer = createOutputBuffer(device, 13),
      countBuffer = createOutputBuffer(device, 2);
    let compiled: ReturnType<typeof rig.graph.compile> | undefined;
    try {
      const summary = importGraphBuffer(rig.graph, 'summary', summaryBuffer, 'float32', 13);
      const permutationCount = importGraphBuffer(rig.graph, 'draws', countBuffer, 'uint32', 2);
      summaryBuffer.write(Float32Array.from({length: 13}, (_, row) => (row === 12 ? 73 : 0)));
      countBuffer.write(Uint32Array.of(0, 71));
      const weights = {
        offsets: rig.input(
          kind === 'jacquez' ? Uint32Array.of(0, 1, 2, 3) : Uint32Array.of(0, 2, 3, 3),
          'uint32'
        ),
        neighbors: rig.input(
          kind === 'jacquez' ? Uint32Array.of(1, 0, 1) : Uint32Array.of(1, 2, 2),
          'uint32'
        ),
        weights: rig.input(Float32Array.of(1, 1, 1), 'float32'),
        distances: rig.input(Float32Array.of(1, 2, 4), 'float32'),
        status: {candidateOverflow: loss.importToGraph(rig.graph)}
      };
      const common = {
        times: rig.input(Float32Array.of(0, 1, 3), 'float32'),
        parameters: rig.input(
          getGPUSpaceTimeParameterValues({seed: 17, permutationCount: 3, timeThreshold: 1}),
          'uint32'
        ),
        maximumPermutations: 3,
        stopAfterExceedances: 3
      };
      const contributor =
        kind === 'mantel'
          ? new GPUMantelTest({
              ...common,
              pairs: weights,
              output: {summary, permutationCount, statistics: rig.output('float32', 4)}
            })
          : kind === 'knox'
            ? new GPUKnoxTest({
                ...common,
                pairs: weights,
                output: {summary, permutationCount, statistics: rig.output('uint32', 4)}
              })
            : new GPUJacquezTest({
                ...common,
                k: 1,
                neighbors: weights,
                output: {summary, permutationCount, statistics: rig.output('uint32', 4)}
              });
      rig.graph.add(contributor);
      compiled = rig.graph.compile();
      for (const flag of [0, 1, 0]) {
        loss.write(Uint32Array.of(flag));
        submitGraph(device, compiled, undefined);
        const values = await readFloat32(summaryBuffer, 13),
          counts = await readUint32(countBuffer, 2);
        expect(values[12]).toBe(73);
        expect(counts).toEqual([flag ? 0 : 3, 71]);
        if (flag) expect(values.slice(0, 12).every(Number.isNaN)).toBe(true);
        else {
          const observed = values[GPU_SPACE_TIME_SUMMARY.observed];
          expect(Number.isFinite(observed)).toBe(true);
          expect(observed).toBeCloseTo(
            kind === 'mantel' ? Math.sqrt(3 / 28) : kind === 'knox' ? 1 : 3,
            5
          );
        }
      }
    } finally {
      compiled?.destroy();
      rig.destroy();
      loss.destroy();
      summaryBuffer.destroy();
      countBuffer.destroy();
    }
  });
