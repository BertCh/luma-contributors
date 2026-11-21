// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUVariogramModelFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {GPUVariogramModelFitOperation} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit-operation';
import {GPUCoregionalizationFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit';
import {GPUCoregionalizationFitOperation} from '../../../src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit-operation';
import {getGPUCoregionalizationFitParameterValues} from '../../../src/dataframe-analysis/pair-statistics/coregionalization-parameters';
import type {GPUContributorOperationViewGetter} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {makeSpatialFixture, readIntegers, readFloats} from '../../spatial-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

for (const operationForm of [false, true]) {
  for (const maximumIterations of [0, 1, 32]) {
    it(`variogram fit status has an independent log-bracket stopping oracle: operation=${operationForm}, budget=${maximumIterations}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = makeSpatialFixture(device);
      try {
        const lags = Float32Array.from({length: 32}, (_, index) => (index + 1) / 8);
        // Closed-form exponential nugget=.2, partial sill=1.5, range=2.2.
        const ordinates = Float32Array.from(
          lags,
          lag => 0.2 + 1.5 * (1 - Math.exp((-3 * lag) / 2.2))
        );
        const distances = fixture.view('float32', lags);
        const semivariances = fixture.view('float32', ordinates);
        const pairCounts = fixture.view('uint32', new Uint32Array(32).fill(10));
        const parameterWords = fixture.view('float32', new Float32Array(4));
        const diagnosticWords = fixture.view('float32', new Float32Array(4));
        const parameters = fixture.graph.createDataView(parameterWords.view.buffer, {
          format: 'float32x4',
          length: 1
        });
        const diagnostics = fixture.graph.createDataView(diagnosticWords.view.buffer, {
          format: 'float32x4',
          length: 1
        });
        const status = fixture.view('uint32', new Uint32Array([0xffffffff]));
        const options = {
          models: ['exponential'] as const,
          weighting: 'pairs' as const,
          rangeSteps: 200,
          maximumIterations
        };
        if (operationForm) {
          const operation = new GPUVariogramModelFitOperation({
            ...options,
            distances: 'distances',
            semivariances: 'semivariances',
            pairCounts: 'pairCounts',
            output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
          });
          const views = {
            distances: distances.view,
            semivariances: semivariances.view,
            pairCounts: pairCounts.view,
            parameters,
            diagnostics,
            status: status.view
          };
          const getView: GPUContributorOperationViewGetter = port =>
            views[port as keyof typeof views] as never;
          fixture.graph.add(operation.createContributor(getView));
        } else {
          fixture.graph.add(
            new GPUVariogramModelFit({
              ...options,
              distances: distances.view,
              semivariances: semivariances.view,
              pairCounts: pairCounts.view,
              output: {parameters, diagnostics, status: status.view}
            })
          );
        }
        // Domain [1/80,12], 200 log-spaced candidates. The two-neighbor
        // bracket has width W=2*log(960)/199. Each golden-section step
        // contracts it by phi=(sqrt(5)-1)/2. W*phi^0 and W*phi^1
        // exceed 1e-5 (unresolved range); W*phi^32 < 1.5e-8.
        // The known range 2.2 is strictly interior, far from either edge.
        await fixture.run();
        expect(await readIntegers(status.buffer)).toEqual([maximumIterations < 32 ? 3 : 0]);
        const bracket = (await readFloats(diagnosticWords.buffer))[3];
        expect(Number.isFinite(bracket)).toBe(true);
        if (maximumIterations < 32) expect(bracket).toBeGreaterThan(1e-5);
        else {
          expect(bracket).toBeLessThanOrEqual(1e-5);
          const range = (await readFloats(parameterWords.buffer))[2];
          expect(Number.isFinite(range)).toBe(true);
          expect(Math.abs(range - 2.2)).toBeLessThan(0.03);
        }
        // Zero usable bins force code 1, then restoring bins must clear it
        // to the original budget/success code on this same compiled graph.
        pairCounts.buffer.write(new Uint32Array(32));
        await fixture.run();
        expect(await readIntegers(status.buffer)).toEqual([1]);
        pairCounts.buffer.write(new Uint32Array(32).fill(10));
        await fixture.run();
        expect(await readIntegers(status.buffer)).toEqual([maximumIterations < 32 ? 3 : 0]);
      } finally {
        fixture.destroy();
      }
    });
  }
}

it('R6 GPU coregionalization accepts relative change equal to tolerance one', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const distances = fixture.view('float32', new Float32Array([1]));
    const semivariances = fixture.view('float32', new Float32Array([2, -1, 3]));
    const pairCounts = fixture.view('uint32', new Uint32Array([10]));
    const parameters = fixture.view(
      'float32',
      getGPUCoregionalizationFitParameterValues([{model: 'spherical', range: 1, sill: 1}], 1)
    );
    const diagnostics = fixture.view('float32', new Float32Array(4));
    const validity = fixture.view('uint32', new Uint32Array([99]));
    const iterationCount = fixture.view('uint32', new Uint32Array([99]));
    const converged = fixture.view('uint32', new Uint32Array([99]));
    const iterationLimitReached = fixture.view('uint32', new Uint32Array([99]));
    const invalidCount = fixture.view('uint32', new Uint32Array([99]));
    fixture.graph.add(
      new GPUCoregionalizationFit({
        variableCount: 2,
        structureCount: 1,
        maximumIterations: 1,
        distances: distances.view,
        semivariances: semivariances.view,
        pairCounts: pairCounts.view,
        parameters: parameters.view,
        output: {
          matrices: fixture.view('float32', new Float32Array(4)).view,
          loadings: fixture.view('float32', new Float32Array(4)).view,
          diagnostics: fixture.graph.createDataView(diagnostics.view.buffer, {
            format: 'float32x4',
            length: 1
          }),
          status: {
            validity: validity.view,
            iterationCount: iterationCount.view,
            converged: converged.view,
            iterationLimitReached: iterationLimitReached.view,
            invalidCount: invalidCount.view
          }
        }
      })
    );
    // First relative change is 3/3=1, equal to the supported tolerance 1;
    // the inclusive stopping criterion converges without a second sweep.
    await fixture.run();
    expect(await readIntegers(iterationCount.buffer)).toEqual([1]);
    expect(await readIntegers(converged.buffer)).toEqual([1]);
    expect(await readIntegers(iterationLimitReached.buffer)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

for (const operationForm of [false, true]) {
  for (const maximumIterations of [0, 1, 4]) {
    it(`coregionalization status follows a two-sweep PSD fixed point: operation=${operationForm}, budget=${maximumIterations}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = makeSpatialFixture(device);
      try {
        const distances = fixture.view('float32', new Float32Array([1]));
        const semivariances = fixture.view('float32', new Float32Array([2, -1, 3]));
        const pairCounts = fixture.view('uint32', new Uint32Array([10]));
        const parameters = fixture.view(
          'float32',
          getGPUCoregionalizationFitParameterValues([{model: 'spherical', range: 1, sill: 1}])
        );
        const matrices = fixture.view('float32', new Float32Array(4));
        const loadings = fixture.view('float32', new Float32Array(4));
        const diagnosticWords = fixture.view('float32', new Float32Array(4));
        const diagnostics = fixture.graph.createDataView(diagnosticWords.view.buffer, {
          format: 'float32x4',
          length: 1
        });
        const validity = fixture.view('uint32', new Uint32Array([0xffffffff]));
        const iterationCount = fixture.view('uint32', new Uint32Array([0xffffffff]));
        const converged = fixture.view('uint32', new Uint32Array([0xffffffff]));
        const iterationLimitReached = fixture.view('uint32', new Uint32Array([0xffffffff]));
        const invalidCount = fixture.view('uint32', new Uint32Array([0xffffffff]));
        const options = {variableCount: 2, structureCount: 1, maximumIterations};
        const status = {
          validity: validity.view,
          iterationCount: iterationCount.view,
          converged: converged.view,
          iterationLimitReached: iterationLimitReached.view,
          invalidCount: invalidCount.view
        };
        if (operationForm) {
          const operation = new GPUCoregionalizationFitOperation({
            ...options,
            distances: 'distances',
            semivariances: 'semivariances',
            pairCounts: 'pairCounts',
            parameters: 'parameters',
            output: {
              matrices: 'matrices',
              loadings: 'loadings',
              diagnostics: 'diagnostics',
              status: {
                validity: 'validity',
                iterationCount: 'iterationCount',
                converged: 'converged',
                iterationLimitReached: 'iterationLimitReached',
                invalidCount: 'invalidCount'
              }
            }
          });
          const views = {
            distances: distances.view,
            semivariances: semivariances.view,
            pairCounts: pairCounts.view,
            parameters: parameters.view,
            matrices: matrices.view,
            loadings: loadings.view,
            diagnostics,
            ...status
          };
          const getView: GPUContributorOperationViewGetter = port =>
            views[port as keyof typeof views] as never;
          fixture.graph.add(operation.createContributor(getView));
        } else {
          fixture.graph.add(
            new GPUCoregionalizationFit({
              ...options,
              distances: distances.view,
              semivariances: semivariances.view,
              pairCounts: pairCounts.view,
              parameters: parameters.view,
              output: {matrices: matrices.view, loadings: loadings.view, diagnostics, status}
            })
          );
        }
        const readStatus = async () => [
          ...(await readIntegers(validity.buffer)),
          ...(await readIntegers(iterationCount.buffer)),
          ...(await readIntegers(converged.buffer)),
          ...(await readIntegers(iterationLimitReached.buffer)),
          ...(await readIntegers(invalidCount.buffer))
        ];
        // Unit spherical lag has shape 1. The target B=[[2,-1],[-1,3]]
        // is PSD: eigenvalues (5±sqrt(5))/2>0. Sweep 1 sets B exactly;
        // sweep 2 leaves it unchanged and converges. After zero sweeps B
        // remains zero; after one sweep the measured change is still 3,
        // exceeding tolerance*3, so convergence has NOT been established.
        const expected =
          maximumIterations === 0
            ? [1, 0, 0, 1, 0]
            : maximumIterations === 1
              ? [1, 1, 0, 1, 0]
              : [1, 2, 1, 0, 0];
        await fixture.run();
        expect(await readStatus()).toEqual(expected);
        // One nonfinite matrix invalidates its entire lag: no usable data,
        // no iterations/convergence/budget flag, exactly one rejected lag.
        semivariances.buffer.write(new Float32Array([2, NaN, 3]));
        await fixture.run();
        expect(await readStatus()).toEqual([0, 0, 0, 0, 1]);
        semivariances.buffer.write(new Float32Array([2, -1, 3]));
        await fixture.run();
        expect(await readStatus()).toEqual(expected);
      } finally {
        fixture.destroy();
      }
    });
  }
}
