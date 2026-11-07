// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUVariogramModelFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {fitVariogramModel} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';
import {InterpolationBenchmarkScope} from '../../gpu-spatial-analysis/spatial-interpolation/interpolation-performance-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';

for (const weighting of ['pairs', 'none', 'cressie'] as const) {
  it(`variogram ${weighting} replay clears failures and preserves model outputs after changed bins`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scope = new InterpolationBenchmarkScope(device, `variogram-replay-${weighting}`);
    const distances = Float32Array.from({length: 20}, (_, index) => (index + 1) * 0.2);
    const pairCounts = Uint32Array.from({length: 20}, (_, index) => 10 + index);
    const models = ['gaussian', 'exponential', 'spherical'] as const;
    const observations = Float32Array.from(
      distances,
      distance => 0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 2.2))
    );
    scope.graph.add(
      new GPUVariogramModelFit({
        distances: scope.input('distances', distances, 'float32'),
        semivariances: scope.input('semivariances', observations, 'float32'),
        pairCounts: scope.input('pairs', pairCounts, 'uint32'),
        models,
        weighting,
        output: {
          parameters: scope.output('parameters', 3, 'float32x4'),
          diagnostics: scope.output('diagnostics', 3, 'float32x4'),
          status: scope.output('status', 3, 'uint32')
        }
      })
    );
    const compiled = scope.graph.compile();
    try {
      scope.initialWrites[0].buffer.write(distances);
      for (const scenario of ['initial', 'overflow', 'missing', 'changed', 'initial'] as const) {
        const values = Float32Array.from(observations, (value, index) =>
          scenario === 'overflow'
            ? (1 + (index % 2)) * 1e30
            : scenario === 'changed'
              ? value * 1.5
              : value
        );
        const counts = scenario === 'missing' ? new Uint32Array(20) : pairCounts;
        scope.initialWrites[1].buffer.write(values);
        scope.initialWrites[2].buffer.write(counts);
        submitGraph(device, compiled, undefined);
        const result = await scope.read();
        if (scenario === 'missing' || scenario === 'overflow') {
          expect(result.status).toEqual(Array(3).fill(scenario === 'missing' ? 1 : 2));
          expect(result.parameters.every(Number.isNaN)).toBe(true);
          expect(result.diagnostics.every(Number.isNaN)).toBe(true);
        } else {
          expect(result.status.every(value => value === 0 || value === 3)).toBe(true);
          for (let modelIndex = 0; modelIndex < models.length; modelIndex++) {
            const expected = fitVariogramModel(
              {
                distances: Array.from(distances),
                semivariances: Array.from(values),
                pairCounts: Array.from(counts)
              },
              {model: models[modelIndex], weighting}
            );
            expect(result.diagnostics[modelIndex * 4]).toBe(20);
            expect(
              result.parameters.slice(modelIndex * 4, modelIndex * 4 + 4).every(Number.isFinite)
            ).toBe(true);
            expect(
              Math.abs(result.parameters[modelIndex * 4 + 3] - expected.residual)
            ).toBeLessThan(0.002 * Math.max(1, Math.abs(expected.residual)));
          }
        }
      }
    } finally {
      compiled.destroy();
      scope.destroy();
    }
  });
}
