// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import fixture from '../../oracles/fixtures/parity-wp1-skater-lag.json';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';

for (const chunked of [false, true]) {
  it(`spatial lag suppresses candidate-incomplete ${chunked ? 'chunked' : 'whole'} weights and recovers`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const loss = new GPUParameterBuffer(device, {
      id: 'lag-loss',
      format: 'uint32',
      length: 1,
      values: Uint32Array.of(1)
    });
    const offsets = new GPUParameterBuffer(device, {
      id: 'lag-offsets',
      format: 'uint32',
      length: chunked ? 2 : 4,
      values: chunked ? Uint32Array.of(0, 1) : Uint32Array.of(0, 1, 2, 2)
    });
    let compiled: ReturnType<typeof rig.graph.compile> | undefined;
    try {
      const first = {
        offsets: offsets.importToGraph(rig.graph),
        neighbors: rig.input(Uint32Array.of(1, 2, 2), 'uint32'),
        weights: rig.input(Float32Array.of(1, 1, 1), 'float32')
      };
      const status = {candidateOverflow: loss.importToGraph(rig.graph)};
      const weights = chunked
        ? {
            rowCount: 3,
            status,
            chunks: [
              {rowStart: 0, rowCount: 1, weights: first},
              {
                rowStart: 1,
                rowCount: 2,
                weights: {
                  offsets: rig.input(Uint32Array.of(0, 1, 1), 'uint32'),
                  neighbors: rig.input(Uint32Array.of(2), 'uint32'),
                  weights: rig.input(Float32Array.of(1), 'float32')
                }
              }
            ]
          }
        : {...first, status};
      const output = rig.output('float32', 3);
      rig.graph.add(
        new GPUSpatialLag({weights, values: rig.input(Float32Array.of(0, 2, 8), 'float32'), output})
      );
      compiled = rig.graph.compile();
      submitGraph(device, compiled, undefined);
      expect((await rig.readFloat(output)).every(Number.isNaN)).toBe(true);
      offsets.write(chunked ? Uint32Array.of(0, 2) : Uint32Array.of(0, 2, 3, 3));
      loss.write(Uint32Array.of(0));
      submitGraph(device, compiled, undefined);
      expect(await rig.readFloat(output)).toEqual([10, 8, 0]);
    } finally {
      compiled?.destroy();
      rig.destroy();
      loss.destroy();
      offsets.destroy();
    }
  });
}

for (const testCase of fixture.lag) {
  it(`GPUSpatialLag libpysal parity: ${testCase.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const output = rig.output('float32', testCase.n * testCase.columns);
      rig.run(
        new GPUSpatialLag({
          weights: rig.weights(testCase),
          values: rig.input(Float32Array.from(testCase.values), 'float32'),
          columnCount: testCase.columns,
          mask: rig.input(Uint32Array.from(testCase.mask), 'uint32'),
          normalize: testCase.normalize,
          output
        })
      );
      const actual = await rig.readFloat(output);
      // Separate columns keep population scale from hiding errors in rates or counts.
      for (let column = 0; column < testCase.columns; column++) {
        compareParity(
          'GPUSpatialLag',
          testCase,
          `column-${column}`,
          actual.filter((_, index) => index % testCase.columns === column),
          testCase.expected.filter((_, index) => index % testCase.columns === column)
        );
      }
    } finally {
      rig.destroy();
    }
  });
}
