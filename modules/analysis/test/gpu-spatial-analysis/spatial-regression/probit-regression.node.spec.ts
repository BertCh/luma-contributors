// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUProbitRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

function getProbitKernels(rows: number) {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `probit-${rows}-${serial++}`, format, length);
    const probit = new GPUProbitRegression({
      predictors: view('float32', rows * 2),
      predictorCount: 2,
      response: view('float32', rows),
      output: {
        coefficients: view('float32', 3),
        standardErrors: view('float32', 3),
        fitted: view('float32', rows),
        marginalEffects: view('float32', 2),
        summary: view('float32', 4),
        status: view('uint32', 1)
      }
    });
    const nodes = probit.getCommandNodes(graph);
    expect(nodes.length).toBe(probit.estimatedCommandNodeCount);
    getParsedKernelSources(nodes);
    return getGPUCommandNodeKernelSources(nodes);
  } finally {
    device.destroy();
  }
}

// Behaviour change on purpose: the single-invocation fit kernel (which looped over every row in
// every iteration) is replaced by row tiles and unrolled iterations, so this pins the new shape.
it('probit row passes take the row count as a pipeline override and no single invocation loops over rows', () => {
  const small = getProbitKernels(40);
  const large = getProbitKernels(60000);
  expect(large.length).toBe(small.length);
  for (const [index, kernel] of small.entries()) {
    expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    if (kernel.id.endsWith('-tiles')) {
      expect(kernel.constants['ROW_COUNT']).toBe(40);
      expect(large[index].constants['ROW_COUNT']).toBe(60000);
      // Only the override value differs: recompiling per row count is avoided.
      expect(large[index].source).toBe(kernel.source);
    }
    // At 60,000 rows every row pass has many tiles; one-invocation kernels never see the rows.
    if (large[index].constants['INVOCATION_COUNT'] === 1) {
      expect(large[index].source, kernel.id).not.toContain('ROW_COUNT');
    }
  }
});

it('probit fit merges the likelihood pairwise and damps overshoot before flagging separation', () => {
  const source = getProbitKernels(40)
    .map(kernel => kernel.source)
    .join('\n');
  // Pairwise merges of the log-likelihood and score tile sums. The former Neumaier compensation
  // is folded to a plain sum by the shader compiler of the test GPU
  // (math-review-tile-merge-sums.spec.ts), so no float compensation may come back.
  expect(source).toContain('sumLevels');
  expect(source).not.toContain('compensation');
  // The likelihood, score and information go through the tail probabilities: no 1 - p of a
  // probability floored at 1e-7, which gave a row beyond float32 rounding a flat likelihood.
  expect(source).toContain('log(tailProbability(signedLinear))');
  expect(source).not.toContain('log(1.0 - fitted)');
  // Only rows on the wrong side of the fit count towards the bound of the line search.
  expect(source).toContain('signedLinear < 0.0');
  // Separation is not decided by a threshold on the linear predictor: a few rows separated by
  // an indicator never pass it before the fit stops, and a legitimate unfinished fit does
  // (frc-regression-03 and -05 of the fix review; the behaviour is pinned on the GPU by
  // math-review-probit-bound.spec.ts). The flag of a separating direction decides instead.
  expect(source).not.toContain('S_LARGEST) > 8.0');
  expect(source).not.toContain('largestLinear > 8.0');
  expect(source).toContain('S_SEPARATING');
  // A certificate has no tolerance (frc2-regression-03): the moves of a row are flags decided
  // against the float32 evaluation error bound, or exact comparisons with a threshold; the
  // ratios that accepted moves towards the wrong side up to 1e-3 are gone.
  expect(source).not.toContain('wrongSideMove');
  expect(source).not.toContain('rightSideMove');
  expect(source).toContain('signedMove < -errorBound');
  expect(source).toContain('value > thresholdValue');
  expect(source).toContain('S_UNDECIDED');
  // A step shortened by the bound or by the likelihood cannot report convergence (verdict 1).
  expect(source).toContain('if (verdict == 1u) { state[stateOffset + S_DONE] = 1.0;');
  // The line search rejects unbounded trial steps by halving, instead of aborting on the first.
  expect(source).toContain('lastLargestLinear <= 12.0');
  expect(source).not.toContain('abs(linear) > 12.0');
  // Absolute 1e-6 acceptance tolerance is replaced by a likelihood-relative one.
  expect(source).not.toContain('current - 1e-6');
  expect(source).toContain('noiseScale * (abs(current) + 1.0)');
});

it('probit predictions, marginal effects and diagnostic residuals use the centered design, not the published raw coefficients', () => {
  // frc2-regression-04: the raw intercept of a predictor with a large offset is a float32
  // difference of large numbers; a linear predictor rebuilt from it loses the fit.
  const kernels = getProbitKernels(40);
  for (const suffix of ['-predict', '-effects-tiles']) {
    const kernel = kernels.find(candidate => candidate.id.endsWith(suffix))!;
    expect(kernel.source, kernel.id).toContain('getDesign(');
    expect(kernel.source, kernel.id).toContain('S_ESTIMATE');
    expect(kernel.source, kernel.id).not.toContain('coefficients[');
  }
  // The raw slopes still scale the mean density into the published marginal effects.
  const effects = kernels.find(candidate => candidate.id.endsWith('-effects'))!;
  expect(effects.source).toContain('coefficients[coefficientsOffset + index + 1u]');
});
