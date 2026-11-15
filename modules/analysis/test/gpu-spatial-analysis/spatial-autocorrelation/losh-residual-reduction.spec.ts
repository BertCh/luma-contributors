// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPULocalSpatialHeteroscedasticityParameterValues,
  GPULocalSpatialHeteroscedasticity
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('LOSH matches hand-derived residual means with padding, an island and missing rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  {
    const rig = new WeightsRig(device);
    const parameters = new GPUParameterBuffer(device, {
      id: 'residual-power',
      format: 'float32',
      length: 2,
      values: new Float32Array([99, 2])
    });
    const imported = parameters.importToGraph(rig.graph);
    const residualPowerView = rig.graph.createDataView(imported.buffer, {
      format: 'float32',
      length: 1,
      byteOffset: 4
    });
    const weights = rig.uploadWeights(
      {
        offsets: [0, 1, 2, 3, 4, 4, 5],
        neighbors: [1, 2, 3, 0, 0],
        weights: [1, 1, 1, 1, 1],
        distances: []
      },
      3
    );
    const statistic = rig.output('float32', 6),
      status = rig.output('uint32', 6),
      normalized = rig.output('float32', 6);
    rig.graph.add(
      new GPULocalSpatialHeteroscedasticity({
        weights,
        residualPower: residualPowerView,
        values: rig.input(new Float32Array([0, 2, 5, 9, 100, NaN]), 'float32', 6),
        output: {
          statistic: statistic.view,
          status: status.view,
          normalizedResiduals: normalized.view
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const residualPower of [1, 2, 1] as const) {
        parameters.write(getGPULocalSpatialHeteroscedasticityParameterValues({residualPower}), 1);
        submitGraph(device, compiled, undefined);
        const residuals = [2, 3, 4, 9].map(value => value ** residualPower);
        const mean = residuals.reduce((sum, value) => sum + value, 0) / 4;
        const actual = await readFloat32(statistic.buffer, 6),
          normalizedValues = await readFloat32(normalized.buffer, 6);
        for (let row = 0; row < 4; row++) {
          expect(actual[row]).toBeCloseTo(residuals[(row + 1) % 4] / mean, 5);
          expect(normalizedValues[row]).toBeCloseTo(residuals[row] / mean, 5);
        }
        expect(actual.slice(4)).toEqual([NaN, NaN]);
        expect(normalizedValues.slice(4)).toEqual([NaN, NaN]);
        expect(await readUint32(status.buffer, 6)).toEqual([0, 0, 0, 0, 1, 1]);
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      rig.destroy();
    }
  }
});

it('LOSH reduces multiple workgroups without overflowing a finite mean and handles zero or absent residuals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 4097; // Partial reduction workgroup; the last row is an island.
  for (const residualPower of [1, 2] as const) {
    const rig = new WeightsRig(device);
    const source = {
      offsets: Array.from({length: rows + 1}, (_, row) => Math.min(row, rows - 1)),
      neighbors: Array.from({length: rows - 1}, (_, row) => row ^ 1),
      weights: Array(rows - 1).fill(1),
      distances: []
    };
    const weights = rig.uploadWeights(source, 2);
    const values = rig.input(new Float32Array(rows), 'float32', rows);
    const mask = rig.input(new Uint32Array(rows).fill(1), 'uint32', rows);
    const statistic = rig.output('float32', rows),
      status = rig.output('uint32', rows),
      normalized = rig.output('float32', rows);
    rig.graph.add(
      new GPULocalSpatialHeteroscedasticity({
        weights,
        values,
        mask,
        residualPower,
        output: {
          statistic: statistic.view,
          status: status.view,
          normalizedResiduals: normalized.view
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const mode of ['ordinary', 'large', 'large-mixed', 'small', 'zero', 'masked'] as const) {
        const amplitude =
          mode === 'large' || mode === 'large-mixed'
            ? residualPower === 1
              ? 1e38
              : 1e18
            : mode === 'small'
              ? residualPower === 1
                ? 2e-38
                : 1e-19
              : mode === 'zero'
                ? 0
                : 1;
        rig.bufferOf(values).write(
          Float32Array.from({length: rows}, (_, row) => {
            const pairScale = mode === 'large-mixed' && Math.floor(row / 2) % 2 === 1 ? 0.5 : 1;
            return (row % 2 === 0 ? amplitude : -amplitude) * pairScale;
          })
        );
        rig.bufferOf(mask).write(new Uint32Array(rows).fill(mode === 'masked' ? 0 : 1));
        submitGraph(device, compiled, undefined);
        const actual = await readFloat32(statistic.buffer, rows),
          states = await readUint32(status.buffer, rows),
          normalizedValues = await readFloat32(normalized.buffer, rows);
        for (let row = 0; row < rows - 1; row++) {
          const label = `power=${residualPower} mode=${mode} row=${row} status=${states[row]} normalized=${normalizedValues[row]}`;
          if (mode === 'zero' || mode === 'masked') {
            expect(actual[row], label).toBeNaN();
            expect(normalizedValues[row], label).toBeNaN();
          } else {
            // Paired rows have equal residuals. Mixed pairs alternate residual powers
            // 1 and (1/2)^a, whose global mean is (1 + (1/2)^a) / 2.
            const pairPower = Math.floor(row / 2) % 2 === 1 ? 0.5 ** residualPower : 1;
            const expected =
              mode === 'large-mixed' ? (2 * pairPower) / (1 + 0.5 ** residualPower) : 1;
            expect(actual[row], label).toBeCloseTo(expected, 5);
            expect(normalizedValues[row], label).toBeCloseTo(expected, 5);
          }
          expect(states[row], label).toBe(mode === 'masked' ? 1 : mode === 'zero' ? 2 : 0);
        }
        expect(actual[rows - 1]).toBeNaN();
        expect(normalizedValues[rows - 1]).toBeNaN();
        expect(states[rows - 1]).toBe(1);
      }
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  }
});
