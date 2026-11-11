// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNestedInterpolationValidationOnCPU,
  CPUNestedInterpolationValidationOperation
} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-nested-interpolation-validation-operation';
import {GPUNestedInterpolationValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation-operation';
import {computeSpatialCrossValidationOnCPU} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-spatial-cross-validation-operation';
import {
  constructSamplingContributor,
  createSamplingCPUInputs
} from '../point-process/validation-sampling-operation-test-utils';
import originalFixture from '../../oracles/fixtures/nested-validation-oracle.json';
import {
  createNestedInput,
  createNestedProps,
  createNestedGraphInputs,
  expectNestedRows,
  fixture
} from './nested-interpolation-validation-operation-test-utils';

it('matches the committed independent nested kriging/Gaussian RBF oracle', () => {
  const input = {
    positions: new Float32Array(originalFixture.positions),
    response: new Float32Array(originalFixture.response),
    training: new Uint32Array(originalFixture.folds.flatMap(fold => fold.training)),
    testing: new Uint32Array(originalFixture.folds.flatMap(fold => fold.testing)),
    innerBlocks: new Uint32Array(originalFixture.blocks)
  };
  const result = computeNestedInterpolationValidationOnCPU(input, {
    foldCount: 2,
    innerFoldCount: 2,
    seed: 8,
    neighborCount: 4,
    candidates: [
      {
        mode: 'ordinary-kriging',
        covariance: {model: 'exponential', sill: 1, range: 3, nugget: 0.01}
      },
      {mode: 'rbf', kernel: 'gaussian', epsilon: 0.5, smoothing: 0.1},
      {mode: 'rbf', kernel: 'gaussian', epsilon: 1, smoothing: 0.1}
    ]
  });
  expect(Array.from(result.selected)).toEqual([0, 0]);
  expect(Array.from(result.status)).toEqual([0, 0]);
  expectNestedRows(
    result.scores,
    originalFixture.folds.flatMap(fold => fold.scores.flat())
  );
  expectNestedRows(
    result.predictions,
    originalFixture.folds.flatMap(fold => fold.predictions)
  );
  expectNestedRows(
    result.metrics,
    originalFixture.folds.flatMap(fold => fold.metrics)
  );
});

it('matches all thirteen SciPy/LAPACK candidate scores and each candidate as the outer predictor', () => {
  const input = createNestedInput();
  const props = createNestedProps();
  const result = computeNestedInterpolationValidationOnCPU(input, props);
  expect(Array.from(result.selected)).toEqual(fixture.folds.map(fold => fold.selected));
  expect(Array.from(result.status)).toEqual([0, 0, 0]);
  expectNestedRows(
    result.scores,
    fixture.folds.flatMap(fold => fold.scores.flat())
  );
  expectNestedRows(
    result.predictions,
    fixture.folds.flatMap(fold => fold.predictions)
  );
  expectNestedRows(
    result.metrics,
    fixture.folds.flatMap(fold => fold.metrics)
  );
  expect(result.selected.some(selected => selected > 0)).toBe(true);
  props.candidates.forEach((candidate, candidateIndex) => {
    const single = computeNestedInterpolationValidationOnCPU(input, {
      ...props,
      candidates: [candidate]
    });
    expect(Array.from(single.selected)).toEqual([0, 0, 0]);
    expect(Array.from(single.status)).toEqual([0, 0, 0]);
    expectNestedRows(
      single.scores,
      fixture.folds.flatMap(fold => fold.scores[candidateIndex])
    );
    expectNestedRows(
      single.predictions,
      fixture.folds.flatMap(fold => fold.candidatePredictions[candidateIndex])
    );
  });
  for (let fold = 0; fold < props.foldCount; fold++) {
    const inner = computeSpatialCrossValidationOnCPU(
      {
        positions: input.positions,
        blocks: input.innerBlocks,
        mask: input.training.subarray(fold * props.rowCount, (fold + 1) * props.rowCount)
      },
      {
        foldCount: props.innerFoldCount!,
        seed: props.seed,
        bufferDistance: props.innerBufferDistance
      }
    );
    expect(Array.from(inner.folds)).toEqual(
      fixture.folds[fold].innerFolds.map(value => (value < 0 ? 0xffffffff : value))
    );
    expect(Array.from(inner.counts).filter((_, index) => index % 3 === 2)).toEqual(
      fixture.folds[fold].bufferedCounts
    );
  }
  expect(fixture.folds.some(fold => fold.bufferedCounts.some(count => count > 0))).toBe(true);
});

it('unpacks every CPU graph port and preserves renamed graph values and fixed row counts', async () => {
  const props = createNestedProps();
  props.output = {
    selected: 'chosen',
    scores: 'losses',
    predictions: 'estimates',
    metrics: 'summary',
    status: 'flags'
  };
  const input = createNestedInput();
  const result = await new CPUNestedInterpolationValidationOperation(props).execute({
    inputs: createSamplingCPUInputs(createNestedGraphInputs(input))
  });
  const expected = computeNestedInterpolationValidationOnCPU(input, props);
  expect(result).toEqual({
    chosen: expected.selected,
    losses: expected.scores,
    estimates: expected.predictions,
    summary: expected.metrics,
    flags: expected.status
  });
  expect(() =>
    new CPUNestedInterpolationValidationOperation({...props, rowCount: props.rowCount + 1}).execute(
      {inputs: createSamplingCPUInputs(createNestedGraphInputs(input))}
    )
  ).toThrow(/response/);
});

it('never uses outer held-out responses to select a model or fit that fold', () => {
  const input = createNestedInput();
  const props = createNestedProps();
  const before = computeNestedInterpolationValidationOnCPU(input, props);
  input.response = Float32Array.from(input.response, (value, row) =>
    input.testing[row] ? value + 20 : value
  );
  const after = computeNestedInterpolationValidationOnCPU(input, props);
  expect(after.selected[0]).toBe(before.selected[0]);
  expect(after.scores.subarray(0, props.candidates.length * 3)).toEqual(
    before.scores.subarray(0, props.candidates.length * 3)
  );
  expect(after.predictions.subarray(0, props.rowCount)).toEqual(
    before.predictions.subarray(0, props.rowCount)
  );
  expect(after.metrics[2]).toBeGreaterThan(100);
});

it('matches explicit defaults and distinguishes partial inner coverage from a valid score', () => {
  const input = createNestedInput();
  const props = createNestedProps();
  const candidate = {mode: 'rbf' as const, epsilon: 1};
  const defaults = {
    ...props,
    innerFoldCount: undefined,
    neighborCount: undefined,
    seed: undefined,
    innerBufferDistance: undefined,
    candidates: [candidate]
  };
  expect(computeNestedInterpolationValidationOnCPU(input, defaults)).toEqual(
    computeNestedInterpolationValidationOnCPU(input, {
      ...defaults,
      innerFoldCount: 3,
      neighborCount: 16,
      seed: 0,
      innerBufferDistance: 0,
      candidates: [{...candidate, kernel: 'thin-plate-spline', degree: 1, smoothing: 0}]
    })
  );
  // Seven rows in one block, one in another. The quadratic model can fit only the split
  // with seven training rows; a finite one-row error must never become a selectable score.
  const small = {
    positions: input.positions.slice(0, 24),
    response: input.response.slice(0, 12),
    innerBlocks: new Uint32Array([0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 3]),
    training: new Uint32Array([
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0
    ]),
    testing: new Uint32Array([
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1
    ])
  };
  const partial = computeNestedInterpolationValidationOnCPU(small, {
    ...props,
    foldCount: 2,
    innerFoldCount: 2,
    innerBufferDistance: 0,
    candidates: [{mode: 'rbf', kernel: 'quintic', epsilon: 1, smoothing: 0.1}]
  });
  expect(Array.from(partial.scores)).toEqual([NaN, 1, 7, NaN, 1, 7]);
  expect(Array.from(partial.selected)).toEqual([0xffffffff, 0xffffffff]);
  expect(Array.from(partial.status)).toEqual([6, 6]);
});

it('rejects overlapping folds and distinguishes empty held-out sets from missing predictions', () => {
  const props = {
    ...createNestedProps(),
    foldCount: 3,
    candidates: [{mode: 'rbf' as const, kernel: 'gaussian' as const, epsilon: 1, smoothing: 0.1}]
  };
  const input = createNestedInput();
  input.training.fill(0);
  input.testing.fill(0);
  input.training.fill(2, 0, props.rowCount);
  input.testing.fill(3, 0, props.rowCount);
  input.testing[2 * props.rowCount] = 1;
  const result = computeNestedInterpolationValidationOnCPU(input, props);
  expect(Array.from(result.selected)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
  expect(Array.from(result.status)).toEqual([7, 10, 6]);
  expect(Array.from(result.metrics.subarray(0, 2))).toEqual([0, props.rowCount]);
  expect(Array.from(result.metrics.subarray(6, 8))).toEqual([0, 0]);
  expect(Array.from(result.metrics.subarray(12, 14))).toEqual([0, 1]);
  expect(result.predictions.every(Number.isNaN)).toBe(true);
  expect(Array.from(result.scores)).toEqual([NaN, 0, 0, NaN, 0, 0, NaN, 0, 0]);
});

it('keeps first-candidate ties, constant-response R², nonfinite row filtering and exact count words', () => {
  const props = createNestedProps();
  const candidate = {
    mode: 'ordinary-kriging' as const,
    covariance: {model: 'spherical' as const, range: 1, sill: 1}
  };
  props.candidates = [candidate, candidate];
  const input = createNestedInput();
  input.response.fill(7);
  // Both nonfinite response and coordinate paths have defined missing-data behavior.
  input.response[0] = NaN;
  input.positions[2] = Infinity;
  const result = computeNestedInterpolationValidationOnCPU(input, props);
  expect(Array.from(result.selected)).toEqual([0, 0, 0]);
  expect(Array.from(result.status)).toEqual([4, 0, 0]);
  expect(Array.from(result.metrics.subarray(0, 2))).toEqual([8, 2]);
  expect(
    Array.from(result.scores)
      .filter((_, index) => index % 3 === 2)
      .every(value => value === 0)
  ).toBe(true);
  for (let fold = 0; fold < props.foldCount; fold++) {
    expect(Number.isNaN(result.metrics[fold * 6 + 5])).toBe(true);
    expect(result.metrics[fold * 6 + 2]).toBeLessThan(1e-25);
    expect(result.scores[fold * 6 + 1]).toBe(20 - (fold === 0 ? 0 : 2));
  }
});

it('rejects deficient RBF polynomial rank and coincident-location buffering', () => {
  const props = {
    ...createNestedProps(),
    rowCount: 6,
    foldCount: 2,
    innerFoldCount: 2,
    innerBufferDistance: 0,
    neighborCount: 4
  };
  const input = {
    positions: new Float32Array([0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0]),
    response: new Float32Array([1, 2, 3, 4, 5, 6]),
    training: new Uint32Array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]),
    testing: new Uint32Array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]),
    innerBlocks: new Uint32Array([0, 1, 2, 3, 4, 5])
  };
  const failed = computeNestedInterpolationValidationOnCPU(input, {
    ...props,
    candidates: [{mode: 'rbf', kernel: 'cubic', epsilon: 1}]
  });
  expect(Array.from(failed.status)).toEqual([6, 10]);
  expect(Array.from(failed.scores)).toEqual([NaN, 0, 4, NaN, 0, 4]);
  const usable = {mode: 'rbf' as const, kernel: 'gaussian' as const, epsilon: 1, smoothing: 0.1};
  const recovered = computeNestedInterpolationValidationOnCPU(input, {
    ...props,
    candidates: [{mode: 'rbf', kernel: 'cubic', epsilon: 1}, usable]
  });
  expect(Array.from(recovered.selected)).toEqual([1, 1]);
  // All training positions collapse; zero-radius exclusion removes all inner training rows.
  input.positions.fill(0);
  const duplicates = computeNestedInterpolationValidationOnCPU(input, {
    ...props,
    candidates: [{mode: 'ordinary-kriging', covariance: {model: 'exponential', range: 1, sill: 1}}]
  });
  expect(Array.from(duplicates.selected)).toEqual([0xffffffff, 0xffffffff]);
  expect(Array.from(duplicates.scores)).toEqual([NaN, 0, 4, NaN, 0, 4]);
});

it('rejects singular duplicate kriging systems without silently adding jitter', () => {
  const input = createNestedInput();
  const props = {
    ...createNestedProps(),
    foldCount: 2,
    innerFoldCount: 2,
    innerBufferDistance: 0,
    candidates: [
      {
        mode: 'ordinary-kriging' as const,
        covariance: {model: 'exponential' as const, range: 1, sill: 1}
      }
    ]
  };
  input.training = new Uint32Array(props.rowCount * 2);
  input.testing = new Uint32Array(props.rowCount * 2);
  for (let row = 0; row < props.rowCount; row++) {
    if (row % 2 === 1)
      input.positions.set(input.positions.subarray((row - 1) * 2, row * 2), row * 2);
    input.innerBlocks[row] = Math.floor(row / 2);
    for (let fold = 0; fold < 2; fold++) {
      input.training[fold * props.rowCount + row] = Number(row < 20);
      input.testing[fold * props.rowCount + row] = Number(row >= 20);
    }
  }
  const result = computeNestedInterpolationValidationOnCPU(input, props);
  expect(Array.from(result.scores)).toEqual([NaN, 0, 20, NaN, 0, 20]);
  expect(Array.from(result.selected)).toEqual([0xffffffff, 0xffffffff]);
  expect(Array.from(result.status)).toEqual([6, 6]);
});

it('removes GPU work caps while keeping float32 counts and uint32 lengths exact', () => {
  const props = createNestedProps();
  expect(
    () =>
      new CPUNestedInterpolationValidationOperation({
        ...props,
        rowCount: 257,
        foldCount: 17,
        innerFoldCount: 9,
        neighborCount: 33,
        candidates: Array(17).fill(props.candidates[0])
      })
  ).not.toThrow();
  expect(
    () => new CPUNestedInterpolationValidationOperation({...props, rowCount: 2 ** 24 + 1})
  ).toThrow(/nested-test.*rowCount/);
  expect(
    () => new CPUNestedInterpolationValidationOperation({...props, foldCount: 0xffffffff})
  ).toThrow(/nested-test.*length/);
  const input = createNestedInput();
  input.response = Float32Array.from(input.response, value => value * 1e20);
  const result = computeNestedInterpolationValidationOnCPU(input, props);
  // f64 scoring still selects, even when the final f32 MSE overflows. No f64 arithmetic failed.
  expect(Array.from(result.status)).toEqual([0, 0, 0]);
  expect(result.selected.every(value => value !== 0xffffffff)).toBe(true);
  expect(result.scores.some(value => value === Infinity)).toBe(true);
});

it('reproduces an affine plane at very large and small coordinate scales', () => {
  // Polynomial reproduction is independent of the radial formula and dense solver. Cubic
  // distances cubed at the large scale would overflow float32 without local normalization.
  for (const scale of [1e30, 1e-10]) {
    const input = createNestedInput();
    const props = {
      ...createNestedProps(),
      innerBufferDistance: 0,
      candidates: [{mode: 'rbf' as const, kernel: 'cubic' as const, epsilon: 1}]
    };
    input.positions = Float32Array.from(input.positions, value => value * scale);
    const plane = (row: number) =>
      2 + input.positions[row * 2] / scale - (3 * input.positions[row * 2 + 1]) / scale;
    input.response = Float32Array.from(input.response, (_, row) => plane(row));
    const result = computeNestedInterpolationValidationOnCPU(input, props);
    expect(Array.from(result.selected)).toEqual([0, 0, 0]);
    expect(Array.from(result.status)).toEqual([0, 0, 0]);
    for (let fold = 0; fold < props.foldCount; fold++) {
      expect(result.scores[fold * 3]).toBeLessThan(1e-10);
      for (let row = 0; row < props.rowCount; row++) {
        const index = fold * props.rowCount + row;
        if (input.testing[index])
          expect(Math.abs(result.predictions[index] - plane(row))).toBeLessThan(2e-6);
        else expect(Number.isNaN(result.predictions[index])).toBe(true);
      }
    }
  }
});

it('validates malformed arrays and CPU props immediately; GPU props validate at contributor construction', () => {
  const input = createNestedInput();
  const props = createNestedProps();
  for (const name of ['positions', 'response', 'training', 'testing', 'innerBlocks'] as const) {
    expect(() =>
      computeNestedInterpolationValidationOnCPU({...input, [name]: input[name].subarray(1)}, props)
    ).toThrow(/positions|response|training|testing|innerBlocks/);
  }
  const shapes = Object.fromEntries(
    Object.entries(createNestedGraphInputs(input)).map(([name, value]) => [
      name,
      {
        format: value.format,
        length: name === 'positions' ? value.values.length / 2 : value.values.length
      }
    ])
  );
  expect(() =>
    constructSamplingContributor(new GPUNestedInterpolationValidationOperation(props), shapes)
  ).not.toThrow();
  for (const change of [
    {foldCount: 1},
    {innerFoldCount: 1},
    {neighborCount: 1},
    {candidates: []}
  ]) {
    expect(() => new CPUNestedInterpolationValidationOperation({...props, ...change})).toThrow(
      /nested-test/
    );
    expect(() =>
      constructSamplingContributor(
        new GPUNestedInterpolationValidationOperation({...props, ...change}),
        shapes
      )
    ).toThrow(/nested-test/);
  }
  for (const change of [
    {seed: -1},
    {seed: 1.5},
    {innerBufferDistance: -1},
    {innerBufferDistance: Infinity},
    {candidates: [{mode: 'rbf' as const, epsilon: 0}]},
    {
      candidates: [{mode: 'rbf' as const, kernel: 'cubic' as const, degree: 0 as const, epsilon: 1}]
    },
    {
      candidates: [
        {
          mode: 'ordinary-kriging' as const,
          covariance: {model: 'gaussian' as const, range: 0, sill: 1}
        }
      ]
    }
  ])
    expect(() => new CPUNestedInterpolationValidationOperation({...props, ...change})).toThrow(
      /nested-test/
    );
  for (const Operation of [
    CPUNestedInterpolationValidationOperation,
    GPUNestedInterpolationValidationOperation
  ]) {
    expect(
      () => new Operation({...props, output: {...props.output, selected: props.positions}})
    ).toThrow(/nested-test/);
    expect(
      () => new Operation({...props, output: {...props.output, status: props.output.selected}})
    ).toThrow(/nested-test/);
  }
});
