// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, expectTypeOf, it} from 'vitest';
import {
  computeEnsembleScoresOnCPU,
  type CPUEnsembleScoresOptions,
  type CPUEnsembleScoresResult,
  GPUEnsembleScoresOperation,
  type EnsembleScoresOperationProps
} from '../../../src/gpu-spatial-analysis/scoring-rules';

type ScoreDescriptor<Value extends string> = Readonly<{
  value: Value;
  format: 'float32';
  length: number;
}>;

const input = {ensemble: Float32Array.of(1, 3), observations: Float32Array.of(2)};

it('TYP-08 returns the literal score selection and preserves selected operation ports', () => {
  const result = computeEnsembleScoresOnCPU(input, {
    memberCount: 2,
    dimension: 1,
    scores: ['crps']
  });
  expectTypeOf(result).toEqualTypeOf<{crps: Float32Array}>();
  expect(Array.from(result.crps)).toEqual([0.5]);
  expect(Object.keys(result)).toEqual(['crps']);
  const operation = new GPUEnsembleScoresOperation({
    ensemble: 'members',
    observations: 'observed',
    memberCount: 2,
    dimension: 1,
    output: {crps: 'crpsValue'}
  });
  const descriptor = operation.getOutput('crps');
  expectTypeOf(descriptor.value).toEqualTypeOf<'crpsValue'>();
  expectTypeOf(descriptor.format).toEqualTypeOf<'float32'>();
  expect(descriptor.value).toBe('crpsValue');
  expectTypeOf(operation.getOutput('energyScore')).toEqualTypeOf<undefined>();
  expect(operation.getOutput('energyScore')).toBeUndefined();
  const omitted = new GPUEnsembleScoresOperation({
    ensemble: 'members',
    observations: 'observed',
    memberCount: 2,
    dimension: 1,
    output: {crps: 'crpsValue', energyScore: undefined}
  });
  expectTypeOf(omitted.getOutput('energyScore')).toEqualTypeOf<undefined>();
  expect(omitted.getOutput('energyScore')).toBeUndefined();
});

// Compiled by the lead's test typecheck; never invoked at runtime.
export function assertEnsembleScoreSelectionTypes(
  options: CPUEnsembleScoresOptions,
  props: EnsembleScoresOperationProps,
  port: string,
  selectEnergy: boolean
): void {
  const selected = computeEnsembleScoresOnCPU(input, {
    memberCount: 2,
    dimension: 1,
    scores: ['energyScore', 'variogramScore'] as const
  });
  expectTypeOf(selected.energyScore).toEqualTypeOf<Float32Array>();
  expectTypeOf(selected.variogramScore).toEqualTypeOf<Float32Array>();
  expectTypeOf(selected).not.toHaveProperty('crps');
  const dynamic = computeEnsembleScoresOnCPU(input, options);
  expectTypeOf(dynamic).toEqualTypeOf<CPUEnsembleScoresResult>();
  const scores: NonNullable<CPUEnsembleScoresOptions['scores']> = options.scores ?? ['crps'];
  const dynamicScores = computeEnsembleScoresOnCPU(input, {...options, scores});
  expectTypeOf(dynamicScores).toEqualTypeOf<CPUEnsembleScoresResult>();
  expectTypeOf(dynamicScores.crps).toEqualTypeOf<Float32Array | undefined>();
  const operation = new GPUEnsembleScoresOperation(props);
  const possible = operation.getOutput('crps');
  expectTypeOf(possible).toEqualTypeOf<ScoreDescriptor<string> | undefined>();
  if (possible) expectTypeOf(possible.value).toEqualTypeOf<string>();
  const literal = new GPUEnsembleScoresOperation({...props, output: {energyScore: 'energy'}});
  expectTypeOf(literal.getOutput('energyScore').value).toEqualTypeOf<'energy'>();
  expectTypeOf(literal.getOutput('crps')).toEqualTypeOf<undefined>();
  const conditional = new GPUEnsembleScoresOperation({
    ...props,
    output: {
      crps: 'siteScore',
      energyScore: selectEnergy ? 'energy' : undefined
    }
  });
  const possibleEnergy = conditional.getOutput('energyScore');
  expectTypeOf(possibleEnergy).toEqualTypeOf<ScoreDescriptor<'energy'> | undefined>();
  if (possibleEnergy) expectTypeOf(possibleEnergy.value).toEqualTypeOf<'energy'>();
  const arbitrary = literal.getOutput(port);
  if (arbitrary) expectTypeOf(arbitrary.value).toEqualTypeOf<'energy'>();
}

it('B4 returns only the actual key from a union-valued tuple element', () => {
  for (const flag of [false, true]) {
    const result = computeEnsembleScoresOnCPU(input, {
      memberCount: 2,
      dimension: 1,
      scores: [flag ? 'crps' : 'energyScore'] as const
    });
    expect(Object.keys(result)).toEqual([flag ? 'crps' : 'energyScore']);
  }
});

it('B5 does not invent output descriptors from a string index signature', () => {
  const output: Record<string, string> & {crps: string} = {crps: 'primary'};
  const operation = new GPUEnsembleScoresOperation({
    ensemble: 'members',
    observations: 'observed',
    memberCount: 2,
    dimension: 1,
    output
  });
  expect(operation.getOutput('crps').value).toBe('primary');
  expect(operation.getOutput('energyScore')).toBeUndefined();
});

// Compile-only probes require absence explicitly instead of relying on diagnostic suppression.
export function assertReviewSelectionTypes(flag: boolean): void {
  const uncertain = computeEnsembleScoresOnCPU(input, {
    memberCount: 2,
    dimension: 1,
    scores: [flag ? 'crps' : 'energyScore'] as const
  });
  expectTypeOf(uncertain.crps).toEqualTypeOf<Float32Array | undefined>();
  expectTypeOf(uncertain.energyScore).toEqualTypeOf<Float32Array | undefined>();
  expectTypeOf(uncertain).not.toMatchTypeOf<{crps: Float32Array}>();
  expectTypeOf(uncertain).not.toMatchTypeOf<{energyScore: Float32Array}>();
  const selection = flag ? (['crps'] as const) : (['energyScore'] as const);
  const alternative = computeEnsembleScoresOnCPU(input, {
    memberCount: 2,
    dimension: 1,
    scores: selection
  });
  expectTypeOf(alternative.crps).toEqualTypeOf<Float32Array | undefined>();
  expectTypeOf(alternative.energyScore).toEqualTypeOf<Float32Array | undefined>();
  const withGuaranteed = flag ? (['crps'] as const) : (['crps', 'energyScore'] as const);
  const guaranteed = computeEnsembleScoresOnCPU(input, {
    memberCount: 2,
    dimension: 1,
    scores: withGuaranteed
  });
  expectTypeOf(guaranteed.crps).toEqualTypeOf<Float32Array>();
  expectTypeOf(guaranteed.energyScore).toEqualTypeOf<Float32Array | undefined>();
  const output: Record<string, string> & {crps: string} = {crps: 'primary'};
  const operation = new GPUEnsembleScoresOperation({
    ensemble: 'members',
    observations: 'observed',
    memberCount: 2,
    dimension: 1,
    output
  });
  expectTypeOf(operation.getOutput('crps').value).toEqualTypeOf<string>();
  const maybeEnergy = operation.getOutput('energyScore');
  expectTypeOf(maybeEnergy).toEqualTypeOf<ScoreDescriptor<string> | undefined>();
  if (maybeEnergy) expectTypeOf(maybeEnergy.value).toEqualTypeOf<string>();
}
