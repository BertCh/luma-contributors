// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  getViewBindingRange,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {createNullWebGPUDevice, createOutputBuffer} from '../../utils/gpu-contributor-test-utils';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {applyFinalGeneralizedCase, type FinalGeneralizedCase} from './local-regression-w5-fixtures';
import {
  GENERALIZED_AUDIT_MODELS,
  createGeneralizedAuditFixture,
  applyGeneralizedAuditReviewCase,
  type GeneralizedAuditReviewCase,
  type GeneralizedAuditModel
} from './generalized-gwr-audit-fixtures';

function runEmittedFit(
  model: GeneralizedAuditModel,
  rescaled = false,
  criterion?: 'aicc' | 'loo-deviance',
  queryOffset?: number,
  reviewCase?: GeneralizedAuditReviewCase,
  finalCase?: FinalGeneralizedCase,
  leaveOneOut = false
) {
  const fixture = createGeneralizedAuditFixture(model, rescaled);
  applyGeneralizedAuditReviewCase(fixture, reviewCase);
  applyFinalGeneralizedCase(fixture, finalCase);
  if (queryOffset !== undefined) fixture.offset[12] = queryOffset;
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const memory = new Map<GraphBufferHandle, ArrayBuffer>();
  let serial = 0;
  const port = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    values: number[]
  ) => {
    const view = createTransientView(
      graph,
      `audit-${serial++}`,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
    const array = format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values);
    memory.set(view.buffer, array.buffer);
    return view;
  };
  const outputs = {
    coefficients: port('float32', Array(28).fill(-999)),
    fitted: port('float32', Array(14).fill(-999)),
    codes: port('uint32', Array(14).fill(999)),
    iterationCount: port('uint32', Array(14).fill(999)),
    standardErrors: port('float32', Array(28).fill(-999)),
    influence: port('float32', Array(14).fill(-999))
  };
  const selection = {
    scores: port('float32', [-999, -999]),
    bandwidth: port('float32', [-999]),
    status: port('uint32', [999]),
    evaluatedBandwidths: port('float32', [-999, -999]),
    iterationCount: port('uint32', [999]),
    converged: port('uint32', [999]),
    iterationLimitReached: port('uint32', [999])
  };
  const getMemory = (view: GraphDataView) => {
    if (!memory.has(view.buffer)) memory.set(view.buffer, new ArrayBuffer(view.buffer.byteLength));
    return memory.get(view.buffer)!;
  };
  try {
    const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
      family: model.family,
      kernel: model.kernel,
      bandwidthMode: model.bandwidthMode,
      positions: port('float32x2', fixture.positions),
      predictors: port('float32', fixture.predictors),
      predictorCount: 1,
      response: port('float32', fixture.response),
      offset: port('float32', fixture.offset),
      mask: port('uint32', fixture.mask),
      parameters: port('float32', fixture.parameters),
      predictExcluded: true,
      maximumIterations: 64,
      leaveOneOut,
      bandwidthSelection: criterion
        ? {
            criterion,
            bandwidths: reviewCase === 'large-counts' ? [10, 0.5] : [2.5, 3],
            output: {
              scores: selection.scores,
              bandwidth: selection.bandwidth,
              status: selection.status,
              evaluatedBandwidths: selection.evaluatedBandwidths
            },
            status: {
              iterationCount: selection.iterationCount,
              converged: selection.converged,
              iterationLimitReached: selection.iterationLimitReached
            }
          }
        : undefined,
      inference: {standardErrors: outputs.standardErrors, influence: outputs.influence},
      output: {
        coefficients: outputs.coefficients,
        fitted: outputs.fitted,
        status: {codes: outputs.codes, iterationCount: outputs.iterationCount}
      }
    });
    for (const node of contributor.getCommandNodes(graph)) {
      const record = getWGSLKernelNodeRecord(graph, node.id)!;
      const bindings = Object.fromEntries(
        record.bindings.map((binding, index) => {
          const range = getViewBindingRange(binding.view);
          return [index, new Uint8Array(getMemory(binding.view), range.offset, range.size)];
        })
      );
      // The interpreter cannot copy arrays through a pointer or into a struct constructor.
      // Expand those copies elementwise; keep the emitted solver/fit arithmetic unchanged.
      const source = record.source
        .replace(/^\s*_\s*=.*;$/gm, '')
        .replace(/@workgroup_size\((\d+)\)/, '@workgroup_size($1, 1, 1)')
        .replace(
          '*scale = factorization.scale;',
          'for (var copied = 0u; copied < 2u; copied++) { (*scale)[copied] = factorization.scale[copied]; }'
        )
        .replace(
          'var factorization = DenseCholesky_2(*scale, 2u, 0u, 1.0);',
          'var factorization: DenseCholesky_2; factorization.size = 2u; factorization.smallestPivotRatio = 1.0; for (var copied = 0u; copied < 2u; copied++) { factorization.scale[copied] = (*scale)[copied]; }'
        );
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
        'main',
        [...record.workgroups],
        {0: bindings},
        {constants: record.constants}
      );
    }
    const read = (view: GraphDataView) =>
      Array.from(
        view.format === 'uint32'
          ? new Uint32Array(getMemory(view))
          : new Float32Array(getMemory(view))
      );
    return {
      fit: Object.fromEntries(
        Object.entries(outputs).map(([name, view]) => [name, read(view)])
      ) as typeof fixture.expected,
      selection: Object.fromEntries(
        Object.entries(selection).map(([name, view]) => [name, read(view)])
      ) as Record<keyof typeof selection, number[]>
    };
  } finally {
    device.destroy();
  }
}

it('W5-51 heterogeneous Poisson exposure units share a feasible initialization', () => {
  const fits = (['feasible-exposure', 'shifted-feasible-exposure'] as const).map(
    finalCase =>
      runEmittedFit(GENERALIZED_AUDIT_MODELS[0], false, undefined, undefined, undefined, finalCase)
        .fit
  );
  for (const [index, fit] of fits.entries()) {
    expect(fit.codes).toEqual(Array(14).fill(0));
    // The positive-count categories have means 1 and log means 0; hence b1=-5,
    // b0=-5 (or -1 when all log exposures decrease by 4). Row zero has mean exp(-23).
    for (let row = 0; row < 14; row++) {
      expect(fit.coefficients[2 * row]).toBeCloseTo(index === 0 ? -5 : -1, 4);
      expect(fit.coefficients[2 * row + 1]).toBeCloseTo(-5, 4);
      expect(Number.isFinite(fit.fitted[row])).toBe(true);
      if (row === 0) expect(fit.fitted[row] / Math.exp(-23)).toBeCloseTo(1, 4);
      else expect(fit.fitted[row]).toBeCloseTo(1, 5);
    }
  }
});

it('B1 fractional exposure shift keeps reconstructed float32 endpoint links feasible', () => {
  for (const [scenario, shift] of [
    ['fractional-exposure', 0],
    ['shifted-fractional-exposure', 2.004]
  ] as const) {
    const {fit} = runEmittedFit(
      GENERALIZED_AUDIT_MODELS[0],
      false,
      undefined,
      undefined,
      undefined,
      scenario
    );
    expect(fit.codes.slice(0, 6)).toEqual(Array(6).fill(0));
    // Category MLE: log rate -23 at x=0 and -33 at x=1, to float32 precision.
    expect(fit.coefficients[0]).toBeCloseTo(-23 - shift, 4);
    expect(fit.coefficients[1]).toBeCloseTo(-10, 4);
    expect(fit.fitted[0] / Math.exp(-23)).toBeCloseTo(1, 4);
  }
});

it('W5-52 explicit LOO and selection anchor on positive-weight training rows', () => {
  const {fit} = runEmittedFit(
    GENERALIZED_AUDIT_MODELS[0],
    false,
    undefined,
    undefined,
    undefined,
    'eligible-extrapolation',
    true
  );
  expect(fit.codes[12]).toBe(0);
  expect(fit.fitted[12]).toBeCloseTo(2.90666e-6, 9);
  const {selection} = runEmittedFit(
    GENERALIZED_AUDIT_MODELS[0],
    false,
    'loo-deviance',
    undefined,
    undefined,
    'eligible-extrapolation'
  );
  // Independent SciPy weighted Poisson fits, audit-gwr-b R4.
  for (const [index, expected] of [3.2169481, 3.20898537].entries()) {
    expect(Number.isFinite(selection.scores[index])).toBe(true);
    expect(selection.scores[index]).toBeCloseTo(expected, 4);
  }
  expect(selection.bandwidth).toEqual([3]);
});

it('R1 Poisson LOO distinguishes small deviances of large counts', () => {
  const {selection} = runEmittedFit(
    GENERALIZED_AUDIT_MODELS[0],
    false,
    'loo-deviance',
    undefined,
    'large-counts'
  );
  expect(selection.scores[0]).toBeCloseTo(0.000754351, 5);
  expect(selection.scores[1]).toBeCloseTo(0.000374729, 5);
  expect(selection.bandwidth).toEqual([0.5]);
});

it('R3 changing Poisson exposure units shifts only the intercept', () => {
  const {fit} = runEmittedFit(
    GENERALIZED_AUDIT_MODELS[0],
    false,
    undefined,
    undefined,
    'offset-shift'
  );
  expect(fit.codes).toEqual(Array(14).fill(0));
  expect(fit.coefficients[0]).toBeCloseTo(14.537205116576097, 4);
  expect(fit.coefficients[1]).toBeCloseTo(0.0259754962002971, 4);
  expect(fit.fitted[0]).toBeCloseTo(1.8893438894752617, 4);
});

it('R3 exposure initialization stays finite for offsets shifted by either 100', () => {
  for (const [reviewCase, shift] of [
    ['exposure-shift-up', 100],
    ['exposure-shift-down', -100]
  ] as const) {
    const {fit} = runEmittedFit(
      GENERALIZED_AUDIT_MODELS[0],
      false,
      undefined,
      undefined,
      reviewCase
    );
    expect(fit.codes).toEqual(Array(14).fill(0));
    expect(fit.coefficients[0]).toBeCloseTo(0.5372051165760972 - shift, 3);
    expect(fit.coefficients[1]).toBeCloseTo(0.0259754962002971, 4);
    expect(fit.fitted[0]).toBeCloseTo(1.8893438894752617, 4);
  }
});

it('R4 held-out extrapolation does not change the training fit rank', () => {
  const {fit} = runEmittedFit(
    GENERALIZED_AUDIT_MODELS[0],
    false,
    undefined,
    undefined,
    'extrapolation'
  );
  expect(fit.codes.slice(12)).toEqual([0, 0]);
  for (const row of [12, 13]) {
    expect(fit.coefficients[2 * row]).toBeCloseTo(0.5547189562170504, 5);
    expect(fit.coefficients[2 * row + 1]).toBeCloseTo(-0.013428224342895145, 5);
    expect(fit.influence[row]).toBe(0);
  }
  expect(fit.fitted[12]).toBeCloseTo(0.00000290666, 9);
  expect(fit.fitted[13]).toBeCloseTo(2.5, 5);
});

it('R5 a singular first information matrix completes zero IRLS updates', () => {
  const {fit} = runEmittedFit(GENERALIZED_AUDIT_MODELS[0], false, undefined, undefined, 'singular');
  expect(fit.codes).toEqual(Array(14).fill(2));
  expect(fit.iterationCount).toEqual(Array(14).fill(0));
});

it('R7 rejects output handles imported twice from overlapping physical bytes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const shared = createOutputBuffer(device, 14);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `alias-${serial++}`, format, length);
  try {
    const fitted = importGraphBuffer(graph, 'first-import', shared, 'float32', 14);
    const influence = importGraphBuffer(graph, 'second-import', shared, 'float32', 14);
    expect(fitted.buffer).not.toBe(influence.buffer);
    expect(
      () =>
        new GPUGeneralizedGeographicallyWeightedRegression({
          family: 'poisson',
          positions: view('float32x2', 14),
          predictors: view('float32', 14),
          predictorCount: 1,
          response: view('float32', 14),
          parameters: view('float32', 2),
          inference: {standardErrors: view('float32', 28), influence},
          output: {
            coefficients: view('float32', 28),
            fitted,
            status: {codes: view('uint32', 14), iterationCount: view('uint32', 14)}
          }
        })
    ).toThrow(/overlap/);
  } finally {
    shared.destroy();
    device.destroy();
  }
});

it('W5-55 snapshots nested inference and output maps before caller mutation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `mutation-${serial++}`, format, length);
  const inference = {standardErrors: view('float32', 28), influence: view('float32', 14)};
  const output = {
    coefficients: view('float32', 28),
    fitted: view('float32', 14),
    status: {codes: view('uint32', 14), iterationCount: view('uint32', 14)}
  };
  const originalInfluence = inference.influence;
  const originalFitted = output.fitted;
  try {
    const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
      family: 'poisson',
      positions: view('float32x2', 14),
      predictors: view('float32', 14),
      predictorCount: 1,
      response: view('float32', 14),
      parameters: view('float32', 2),
      inference,
      output
    });
    const selectionOutput = {
      scores: view('float32', 2),
      bandwidth: view('float32', 1),
      status: view('uint32', 1)
    };
    for (const method of ['grid', 'golden-section'] as const) {
      let methodReads = 0;
      const selection = Object.create(
        method === 'grid'
          ? {
              get parameters() {
                throw new Error('inactive golden parameters');
              }
            }
          : {
              get bandwidths() {
                throw new Error('inactive grid bandwidths');
              }
            }
      );
      Object.defineProperty(selection, 'method', {
        get() {
          methodReads++;
          return method;
        }
      });
      Object.assign(
        selection,
        {output: selectionOutput},
        method === 'grid'
          ? {bandwidths: [2, 3]}
          : {parameters: view('float32', 3), maximumIterations: 1}
      );
      expect(
        () =>
          new GPUGeneralizedGeographicallyWeightedRegression({
            ...contributor.props,
            bandwidthSelection: selection
          })
      ).not.toThrow();
      expect(methodReads).toBe(1);
    }
    inference.influence = output.fitted;
    output.fitted = view('float32', 14);
    expect(contributor.props.inference!.influence).toBe(originalInfluence);
    expect(contributor.props.output.fitted).toBe(originalFitted);
    const nodes = contributor.getCommandNodes(graph);
    const records = nodes.map(node => getWGSLKernelNodeRecord(graph, node.id)!);
    expect(
      records.some(record =>
        record.bindings.some(
          binding => binding.view === originalInfluence && binding.access === 'read_write'
        )
      )
    ).toBe(true);
    expect(
      records.some(record =>
        record.bindings.some(
          binding => binding.view === originalFitted && binding.access === 'read_write'
        )
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('R9 a prediction-only logistic offset of 100 remains a valid saturated probability', () => {
  const {fit} = runEmittedFit(GENERALIZED_AUDIT_MODELS[4], false, undefined, 100);
  expect(fit.codes[12]).toBe(0);
  expect(fit.fitted[12]).toBe(1);
  expect(fit.influence[12]).toBe(0);
  expect(fit.coefficients[24]).toBeCloseTo(0, 5);
  expect(fit.coefficients[25]).toBeCloseTo(-0.25, 5);
});

it('emitted Poisson GLM gives masked queries zero diagonal influence against the category MLE oracle', () => {
  const {fit: result} = runEmittedFit(GENERALIZED_AUDIT_MODELS[0]);
  expect(result.coefficients[0]).toBeCloseTo(0.5372051165760972, 4);
  expect(result.codes).toEqual(Array(14).fill(0));
  expect(result.influence.slice(12)).toEqual([0, 0]);
});

it('emitted Poisson query overflow reports numerical failure and suppresses dependent outputs', () => {
  const {fit} = runEmittedFit(GENERALIZED_AUDIT_MODELS[0], false, undefined, 100);
  expect(fit.fitted[12]).toBeNaN();
  expect(fit.codes[12]).toBe(4);
  expect(fit.iterationCount[12]).toBe(5);
  expect(fit.coefficients.slice(24, 26).every(Number.isNaN)).toBe(true);
  expect(fit.standardErrors.slice(24, 26).every(Number.isNaN)).toBe(true);
  expect(fit.influence[12]).toBeNaN();
});

it.each(
  GENERALIZED_AUDIT_MODELS
)('emitted $family/$kernel/$bandwidthMode matches every category MLE output, including rescaled inputs', model => {
  const maximumErrors: Record<string, number> = {};
  for (const rescaled of [false, true]) {
    const {fit: actual} = runEmittedFit(model, rescaled);
    const {expected} = createGeneralizedAuditFixture(model, rescaled);
    expect(actual.codes).toEqual(expected.codes);
    expect(actual.iterationCount).toEqual(expected.iterationCount);
    for (const name of ['coefficients', 'fitted', 'standardErrors', 'influence'] as const) {
      expect(actual[name]).toHaveLength(expected[name].length);
      actual[name].forEach((value, index) => {
        expect(Number.isFinite(value), `${name}[${index}]`).toBe(true);
        const difference = Math.abs(value - expected[name][index]);
        maximumErrors[name] = Math.max(maximumErrors[name] ?? 0, difference);
        // 1e-4 covers the short f32 reductions, Newton stopping and Cholesky rounding.
        expect(difference, `${name}[${index}]`).toBeLessThan(1e-4);
      });
    }
  }
  console.info(
    `${model.family}/${model.kernel}/${model.bandwidthMode} maximum absolute differences`,
    maximumErrors
  );
});

it.each(
  GENERALIZED_AUDIT_MODELS.filter(model => model.bandwidthMode === 'fixed')
)('emitted $family/$kernel selection scores and statuses match independent category deviances', model => {
  for (const criterion of ['aicc', 'loo-deviance'] as const) {
    const {selection, fit} = runEmittedFit(model, false, criterion);
    const expected = criterion === 'aicc' ? model.aicc : model.loo;
    console.info(
      `${model.family}/${model.kernel}/${criterion} maximum absolute score difference`,
      Math.max(...selection.scores.map((value, index) => Math.abs(value - expected[index])))
    );
    selection.scores.forEach((value, index) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value - expected[index])).toBeLessThan(1e-4);
    });
    expect(selection.bandwidth).toEqual([expected[0] < expected[1] ? 2.5 : 3]);
    expect(selection.evaluatedBandwidths).toEqual([2.5, 3]);
    expect(selection.status).toEqual([0]);
    expect(selection.iterationCount).toEqual([1]);
    expect(selection.converged).toEqual([1]);
    expect(selection.iterationLimitReached).toEqual([0]);
    expect(fit.codes).toEqual(Array(14).fill(0));
  }
});
