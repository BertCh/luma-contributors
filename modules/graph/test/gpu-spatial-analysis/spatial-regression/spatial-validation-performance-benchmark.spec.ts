// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {GPUNestedInterpolationValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation';
import type {GPUSpatialValidationDesign} from '../../../src/gpu-spatial-analysis/spatial-validation/spatial-validation-design';
import {
  measureInterpolationCase,
  type InterpolationBenchmarkCase
} from '../spatial-interpolation/interpolation-performance-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import sourceSnapshot from './spatial-validation-performance-source.json';
const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;
const cases: InterpolationBenchmarkCase[] = [];
for (const mode of ['groups', 'environmental', 'nndm', 'knndm'] as const) {
  for (const rows of mode === 'nndm' ? [16, 64] : [64, 256]) {
    cases.push({
      name: `validation-${mode}-${rows}`,
      shape: {
        rows,
        predictions: rows,
        folds: mode === 'nndm' ? rows : 2,
        candidates: mode === 'knndm' ? 4 : 0
      },
      correctness:
        'Successful design, every row tested exactly once, disjoint training/testing; changed-position/feature replay',
      construct(scope) {
        const coordinates = new Float32Array(
          Array.from({length: rows}, (_, row) => [row % 16, Math.floor(row / 16)]).flat()
        );
        const positions = scope.input(
          'positions',
          coordinates,
          'float32x2',
          coordinates.map(value => value + 0.125)
        );
        const diagnostics = scope.output('diagnostics', mode === 'nndm' ? 4 + rows : 4, 'float32');
        const predictionPositions = scope.input(
          'prediction',
          coordinates.map(value => value + 20),
          'float32x2'
        );
        const design: GPUSpatialValidationDesign =
          mode === 'groups'
            ? {mode, diagnostics}
            : mode === 'environmental'
              ? {
                  mode,
                  features: scope.input(
                    'features',
                    Float32Array.from({length: rows}, (_, row) => row),
                    'float32',
                    Float32Array.from({length: rows}, (_, row) => row + 1)
                  ),
                  featureCount: 1,
                  diagnostics
                }
              : mode === 'nndm'
                ? {mode, predictionPositions, maximumDistance: 2, diagnostics}
                : {mode, predictionPositions, candidateCount: 4, diagnostics};
        const foldCount = mode === 'nndm' ? rows : 2;
        scope.graph.add(
          new GPUSpatialCrossValidation({
            positions,
            blocks: scope.input(
              'blocks',
              Uint32Array.from({length: rows}, (_, row) => row),
              'uint32'
            ),
            foldCount,
            design,
            output: {
              folds: scope.output('folds', rows, 'uint32'),
              training: scope.output('training', rows * foldCount, 'uint32'),
              testing: scope.output('testing', rows * foldCount, 'uint32'),
              counts: scope.output('counts', foldCount * 3, 'uint32'),
              status: scope.output('status', foldCount, 'uint32')
            }
          })
        );
        return actual => {
          expect(actual['diagnostics'][0]).toBe(0);
          for (let row = 0; row < rows; row++) {
            let tests = 0;
            for (let fold = 0; fold < foldCount; fold++) {
              const index = fold * rows + row;
              tests += actual['testing'][index];
              expect(actual['testing'][index] * actual['training'][index]).toBe(0);
            }
            expect(tests).toBe(1);
          }
        };
      }
    });
  }
}
for (const rows of [12, 256]) {
  cases.push({
    name: `nested-interpolation-${rows}`,
    shape: {rows, outerFolds: 2, innerFolds: 2, candidates: 2, neighbors: 4},
    correctness:
      'Constant ordinary-kriging reproduction, model selection, all held-out rows predicted; same graph response2 to3 replay',
    construct(scope) {
      scope.graph.add(
        new GPUNestedInterpolationValidation({
          positions: scope.input(
            'positions',
            new Float32Array(
              Array.from({length: rows}, (_, row) => [row % 8, Math.floor(row / 8)]).flat()
            ),
            'float32x2'
          ),
          response: scope.input(
            'response',
            new Float32Array(rows).fill(2),
            'float32',
            new Float32Array(rows).fill(3)
          ),
          training: scope.input(
            'training',
            Uint32Array.from({length: rows * 2}, (_, index) =>
              Number((index % rows) % 2 !== Math.floor(index / rows))
            ),
            'uint32'
          ),
          testing: scope.input(
            'testing',
            Uint32Array.from({length: rows * 2}, (_, index) =>
              Number((index % rows) % 2 === Math.floor(index / rows))
            ),
            'uint32'
          ),
          innerBlocks: scope.input(
            'blocks',
            Uint32Array.from({length: rows}, (_, row) => Math.floor(row / 2)),
            'uint32'
          ),
          foldCount: 2,
          innerFoldCount: 2,
          neighborCount: 4,
          seed: 2,
          candidates: [
            {
              mode: 'ordinary-kriging',
              covariance: {model: 'exponential', sill: 1, range: 3, nugget: 0.01}
            },
            {mode: 'rbf', epsilon: 1, kernel: 'gaussian', smoothing: 0.1}
          ],
          output: {
            selected: scope.output('selected', 2, 'uint32'),
            scores: scope.output('scores', 12, 'float32'),
            predictions: scope.output('predictions', rows * 2, 'float32'),
            metrics: scope.output('metrics', 12, 'float32'),
            status: scope.output('status', 2, 'uint32')
          }
        })
      );
      return (actual, updated) => {
        expect(actual['status']).toEqual([0, 0]);
        expect(actual['selected']).toEqual([0, 0]);
        for (let fold = 0; fold < 2; fold++) {
          expect(actual['metrics'][fold * 6 + 2]).toBeLessThan(1e-8);
          for (let row = fold; row < rows; row += 2)
            expect(actual['predictions'][fold * rows + row]).toBeCloseTo(updated ? 3 : 2, 4);
        }
      };
    }
  });
}
it.skipIf(environment?.['VITE_GEOSPATIAL_BACKLOG_BENCHMARK'] !== 'true')(
  'spatial validation bounded performance envelopes',
  {timeout: 300000},
  async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    console.log(
      `GEOSPATIAL_VALIDATION_METADATA ${JSON.stringify({sourceSnapshot, recordedAt: new Date().toISOString(), device: device.info, userAgent: navigator.userAgent, seed: 2, method: 'CPU compile/upload/encode-submit-completion/readback timings; five warm and three updated-input samples; shared device caches; no speedup assertion', limits: {maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage, maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize}})}`
    );
    for (const benchmark of cases)
      console.log(
        `GEOSPATIAL_VALIDATION_MEASUREMENT ${JSON.stringify(await measureInterpolationCase(device, benchmark))}`
      );
  }
);
