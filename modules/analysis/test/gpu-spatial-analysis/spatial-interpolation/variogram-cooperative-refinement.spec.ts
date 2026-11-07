// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUVariogramModelFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {
  fitVariogramModel,
  evaluateVariogramModel
} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';
import {InterpolationBenchmarkScope} from './interpolation-performance-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';

for (const weighting of ['none', 'pairs', 'cressie'] as const) {
  for (const binCount of [63, 64, 65, 257, 1025]) {
    it(`cooperative variogram ${weighting} handles ${binCount} bins and failed/recovered replays`, {
      timeout: 120000
    }, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const scope = new InterpolationBenchmarkScope(
        device,
        `cooperative-variogram-${weighting}-${binCount}`
      );
      const models = ['gaussian', 'spherical', 'exponential'] as const;
      const originalDistances = Float32Array.from(
        {length: binCount},
        (_, bin) => 0.05 + (4 * bin) / (binCount - 1)
      );
      const originalValues = originalDistances.map(
        (distance, bin) =>
          0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 2.2)) + 0.025 * Math.sin(bin * 1.7)
      );
      const originalCounts = Uint32Array.from({length: binCount}, (_, bin) => 10 + (bin % 29));
      const distances = originalDistances.slice(),
        values = originalValues.slice(),
        counts = originalCounts.slice();
      scope.graph.add(
        new GPUVariogramModelFit({
          distances: scope.input('distances', distances, 'float32'),
          semivariances: scope.input('values', values, 'float32'),
          pairCounts: scope.input('counts', counts, 'uint32'),
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
      let initial: Awaited<ReturnType<typeof scope.read>> | undefined;
      const close = (actual: number, expected: number, name: string, tolerance = 0.003) =>
        expect(Math.abs(actual - expected), name).toBeLessThanOrEqual(
          tolerance * Math.max(1, Math.abs(expected))
        );
      try {
        for (const scenario of ['initial', 'scaled', 'masked', 'empty', 'overflow', 'initial']) {
          distances.set(originalDistances);
          values.set(originalValues);
          counts.set(originalCounts);
          if (scenario === 'scaled') values.set(originalValues.map(value => value * 2));
          if (scenario === 'masked') {
            distances[0] = 0;
            values[1] = NaN;
            counts[2] = 0;
          }
          if (scenario === 'empty') counts.fill(0);
          if (scenario === 'overflow') values.fill(1e38);
          scope.initialWrites[0].buffer.write(distances);
          scope.initialWrites[1].buffer.write(values);
          scope.initialWrites[2].buffer.write(counts);
          submitGraph(device, compiled, undefined);
          const actual = await scope.read();
          if (scenario === 'empty' || scenario === 'overflow') {
            expect(actual['status']).toEqual(Array(3).fill(scenario === 'empty' ? 1 : 2));
            expect(actual['parameters'].every(Number.isNaN)).toBe(true);
            expect(actual['diagnostics'].every(Number.isNaN)).toBe(true);
            continue;
          }
          expect(actual['status']).toEqual([0, 0, 0]);
          for (const [modelIndex, model] of models.entries()) {
            const expected = fitVariogramModel(
              {distances, semivariances: values, pairCounts: counts},
              {model, weighting}
            );
            const offset = modelIndex * 4;
            const parameters = actual['parameters'].slice(offset, offset + 4);
            [expected.nugget, expected.sill, expected.range, expected.residual].forEach(
              (value, component) =>
                close(parameters[component], value, `${scenario}:${model}:${component}`)
            );
            let included = 0,
              residual = 0;
            for (let bin = 0; bin < binCount; bin++) {
              if (
                !(
                  counts[bin] > 0 &&
                  distances[bin] > 0 &&
                  Number.isFinite(distances[bin]) &&
                  Number.isFinite(values[bin])
                )
              )
                continue;
              included++;
              const difference =
                values[bin] -
                evaluateVariogramModel(
                  {model, nugget: parameters[0], sill: parameters[1], range: parameters[2]},
                  distances[bin]
                );
              residual += difference * difference;
            }
            expect(actual['diagnostics'][offset]).toBe(included);
            close(
              actual['diagnostics'][offset + 1],
              residual,
              `${scenario}:${model}:residual`,
              1e-5
            );
            close(
              actual['diagnostics'][offset + 2],
              included * Math.log(Math.max(residual / included, 1e-30)) + 6,
              `${scenario}:${model}:AIC`,
              1e-5
            );
            expect(actual['diagnostics'][offset + 3]).toBeLessThanOrEqual(1e-5);
          }
          if (scenario === 'initial') {
            if (initial) expect(actual).toEqual(initial);
            else initial = actual;
          }
        }
      } finally {
        compiled.destroy();
        scope.destroy();
      }
    });
  }
}
