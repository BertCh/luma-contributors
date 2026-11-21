// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpatialRegressionDiagnosticsOnCPU,
  CPUSpatialRegressionDiagnosticsOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-regression-diagnostics-operation';
import {
  GPUSpatialRegressionDiagnosticsOperation,
  type SpatialRegressionDiagnosticsOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics-operation';
import fixtures from './spatial-regression-diagnostics-operation-fixtures.json';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';

const PROPS: SpatialRegressionDiagnosticsOperationProps = {
  id: 'diagnostics-test',
  rowCount: 30,
  predictorCount: 2,
  weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
  predictors: 'predictors',
  response: 'response',
  residuals: 'residuals',
  mask: 'mask',
  output: {tests: 'tests', summary: 'summary', status: {code: 'code'}}
};
function getInput(fixture = fixtures[0]) {
  return {
    weights: {
      offsets: Uint32Array.from(fixture.weights.offsets),
      neighbors: Uint32Array.from(fixture.weights.neighbors),
      weights: Float32Array.from(fixture.weights.weights)
    },
    predictors: Float32Array.from(fixture.predictors),
    response: Float32Array.from(fixture.response),
    residuals: Float32Array.from(fixture.residuals),
    mask: Uint32Array.from(fixture.mask)
  };
}

it('G3 operations retain declared loss inputs after caller status and output maps mutate', async () => {
  const props = {
    ...PROPS,
    weights: {...PROPS.weights, status: {candidateOverflow: 'flag' as string | undefined}},
    output: {...PROPS.output, status: {...PROPS.output.status}}
  };
  const cpu = new CPUSpatialRegressionDiagnosticsOperation(props);
  const gpu = new GPUSpatialRegressionDiagnosticsOperation(props);
  const input = getInput();
  const vector = <Values extends Float32Array | Uint32Array>(values: Values) => ({
    format: (values instanceof Float32Array ? 'float32' : 'uint32') as 'float32' | 'uint32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const inputs = {
    offsets: vector(input.weights.offsets),
    neighbors: vector(input.weights.neighbors),
    weights: vector(input.weights.weights),
    predictors: vector(input.predictors),
    response: vector(input.response),
    residuals: vector(input.residuals),
    mask: vector(input.mask),
    weightsCandidateOverflow: vector(Uint32Array.of(1))
  };
  expect((await cpu.execute({inputs}))['code']).toEqual(Uint32Array.of(3));
  delete props.weights.status.candidateOverflow;
  props.output.status.code = 'replacement';
  expect((await cpu.execute({inputs}))['code']).toEqual(Uint32Array.of(3));
  const contributor = constructRegressionOperationContributor(gpu, {
    offsets: {format: 'uint32', length: 31},
    neighbors: {format: 'uint32', length: 90},
    weights: {format: 'float32', length: 90},
    predictors: {format: 'float32', length: 60},
    response: {format: 'float32', length: 30},
    residuals: {format: 'float32', length: 30},
    mask: {format: 'uint32', length: 30},
    weightsCandidateOverflow: {format: 'uint32', length: 1}
  });
  inputs.weightsCandidateOverflow.pack()[0] = 0;
  const recovered = (await cpu.execute({inputs})) as Record<string, Float32Array | Uint32Array>;
  expect(recovered['code']).toEqual(Uint32Array.of(0));
  expect(Number.isFinite(recovered['tests'][0])).toBe(true);
  expect(contributor).toHaveProperty('props.weights.status.candidateOverflow');
  expect(
    (contributor as unknown as {props: {weights: {status: {candidateOverflow?: unknown}}}}).props
      .weights.status.candidateOverflow
  ).toBeDefined();
});

it.each([
  'overflow',
  'candidateOverflow'
] as const)('F2 suppresses CPU diagnostics on weights %s and recovers', flag => {
  const input = getInput();
  const loss = Uint32Array.of(1);
  const weights = {...input.weights, status: {[flag]: loss}};
  const result = computeSpatialRegressionDiagnosticsOnCPU({...input, weights}, {predictorCount: 2});
  expect(result.status[0]).toBe(3);
  for (let test = 0; test < 6; test++) {
    expect(result.tests[3 * test]).toBeNaN();
    expect(result.tests[3 * test + 2]).toBeNaN();
  }
  loss[0] = 0;
  expect(
    computeSpatialRegressionDiagnosticsOnCPU({...input, weights}, {predictorCount: 2}).status[0]
  ).toBe(0);
});

for (const fixture of fixtures) {
  it(`matches independent dense SciPy annihilator traces and tests: ${fixture.name}`, () => {
    const result = computeSpatialRegressionDiagnosticsOnCPU(getInput(fixture), fixture);
    expect(result.status[0]).toBe(0);
    expectRegressionRowsClose(result.tests, fixture.expected.tests, 2e-6);
    expectRegressionRowsClose(result.summary, fixture.expected.summary, 2e-6);
    expect(result.tests[2]).toBeGreaterThan(1e-4);
    expect(result.tests[2]).toBeLessThan(0.99);
  });
}

it('preserves every failure layout and status precedence', () => {
  const input = getInput();
  for (const [changes, expectedStatus, expectedCount] of [
    [{mask: new Uint32Array(30)}, 2, 0],
    [{predictors: new Float32Array(60)}, 1, 30],
    [{weights: {...input.weights, weights: new Float32Array(90)}}, 3, 30],
    [{residuals: new Float32Array(30)}, 4, 30]
  ] as const) {
    const result = computeSpatialRegressionDiagnosticsOnCPU(
      {...input, ...changes},
      {predictorCount: 2}
    );
    expect(result.status[0]).toBe(expectedStatus);
    expect(result.summary[0]).toBe(expectedCount);
    expect(result.summary[12]).toBe(30 - expectedCount);
    expect(result.summary.subarray(1, 12).every(Number.isNaN)).toBe(true);
    for (let test = 0; test < 6; test++) {
      expect(result.tests[test * 3]).toBeNaN();
      expect(result.tests[test * 3 + 2]).toBeNaN();
      expect(result.tests[test * 3 + 1]).toBe(test === 5 ? 0 : test === 4 ? 2 : 1);
    }
  }
  const nonFinite = input.weights.weights.slice();
  nonFinite[0] = NaN;
  expect(
    computeSpatialRegressionDiagnosticsOnCPU(
      {...input, weights: {...input.weights, weights: nonFinite}},
      {predictorCount: 2}
    ).status[0]
  ).toBe(4);
});

it('checks array consistency and both operation constructions', () => {
  const input = getInput();
  for (const changes of [
    {predictors: new Float32Array(59)},
    {response: new Float32Array(29)},
    {residuals: new Float32Array(29)},
    {mask: new Uint32Array(29)}
  ])
    expect(() =>
      computeSpatialRegressionDiagnosticsOnCPU({...input, ...changes}, {predictorCount: 2})
    ).toThrow(new RegExp(Object.keys(changes)[0]));
  expect(() =>
    computeSpatialRegressionDiagnosticsOnCPU(
      {...input, weights: {...input.weights, offsets: Uint32Array.of(0, 91)}},
      {predictorCount: 2}
    )
  ).toThrow(/offsets/);
  const shapes = {
    offsets: {format: 'uint32' as const, length: 31},
    neighbors: {format: 'uint32' as const, length: 90},
    weights: {format: 'float32' as const, length: 90},
    predictors: {format: 'float32' as const, length: 60},
    response: {format: 'float32' as const, length: 30},
    residuals: {format: 'float32' as const, length: 30},
    mask: {format: 'uint32' as const, length: 30}
  };
  for (const changes of [{predictorCount: 0}, {tileRowCount: 0}]) {
    expect(() => new CPUSpatialRegressionDiagnosticsOperation({...PROPS, ...changes})).toThrow(
      /diagnostics-test/
    );
    expect(() =>
      constructRegressionOperationContributor(
        new GPUSpatialRegressionDiagnosticsOperation({...PROPS, ...changes}),
        shapes
      )
    ).toThrow(/diagnostics-test/);
  }
  for (const Operation of [
    CPUSpatialRegressionDiagnosticsOperation,
    GPUSpatialRegressionDiagnosticsOperation
  ])
    expect(() => new Operation({...PROPS, output: {...PROPS.output, summary: 'response'}})).toThrow(
      /diagnostics-test/
    );
  expect(new CPUSpatialRegressionDiagnosticsOperation(PROPS).contract.outputs).toEqual(
    new GPUSpatialRegressionDiagnosticsOperation(PROPS).contract.outputs
  );
  expect(() =>
    constructRegressionOperationContributor(
      new GPUSpatialRegressionDiagnosticsOperation(PROPS),
      shapes
    )
  ).not.toThrow();
  const withStatus = {
    ...PROPS,
    weights: {...PROPS.weights, status: {overflow: 'loss', candidateOverflow: 'candidate-loss'}}
  };
  for (const Operation of [
    CPUSpatialRegressionDiagnosticsOperation,
    GPUSpatialRegressionDiagnosticsOperation
  ]) {
    const operation = new Operation(withStatus);
    expect(operation.inputs).toMatchObject({
      weightsOverflow: 'loss',
      weightsCandidateOverflow: 'candidate-loss'
    });
    if ('destroy' in operation) operation.destroy();
  }
  const contributor = constructRegressionOperationContributor(
    new GPUSpatialRegressionDiagnosticsOperation(withStatus),
    {
      ...shapes,
      weightsOverflow: {format: 'uint32', length: 1},
      weightsCandidateOverflow: {format: 'uint32', length: 1}
    }
  );
  expect(contributor).toMatchObject({
    props: {weights: {status: {overflow: expect.anything(), candidateOverflow: expect.anything()}}}
  });
});

it('BC2 operations retain complete certificates after caller status and output maps mutate', async () => {
  const props = {
    ...PROPS,
    weights: {...PROPS.weights, status: {complete: 'flag' as string | undefined}},
    output: {...PROPS.output, status: {...PROPS.output.status}}
  };
  const cpu = new CPUSpatialRegressionDiagnosticsOperation(props);
  const gpu = new GPUSpatialRegressionDiagnosticsOperation(props);
  const input = getInput();
  const vector = <Values extends Float32Array | Uint32Array>(values: Values) => ({
    format: (values instanceof Float32Array ? 'float32' : 'uint32') as 'float32' | 'uint32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const inputs = {
    offsets: vector(input.weights.offsets),
    neighbors: vector(input.weights.neighbors),
    weights: vector(input.weights.weights),
    predictors: vector(input.predictors),
    response: vector(input.response),
    residuals: vector(input.residuals),
    mask: vector(input.mask),
    complete: vector(Uint32Array.of(0))
  };
  expect((await cpu.execute({inputs}))['code']).toEqual(Uint32Array.of(3));
  delete props.weights.status.complete;
  props.output.status.code = 'replacement';
  expect((await cpu.execute({inputs}))['code']).toEqual(Uint32Array.of(3));
  const contributor = constructRegressionOperationContributor(gpu, {
    offsets: {format: 'uint32', length: 31},
    neighbors: {format: 'uint32', length: 90},
    weights: {format: 'float32', length: 90},
    predictors: {format: 'float32', length: 60},
    response: {format: 'float32', length: 30},
    residuals: {format: 'float32', length: 30},
    mask: {format: 'uint32', length: 30},
    complete: {format: 'uint32', length: 1}
  });
  inputs.complete.pack()[0] = 1;
  const recovered = (await cpu.execute({inputs})) as Record<string, Float32Array | Uint32Array>;
  expect(recovered['code']).toEqual(Uint32Array.of(0));
  expect(Number.isFinite(recovered['tests'][0])).toBe(true);
  expect(contributor).toHaveProperty('props.weights.status.complete');
  expect(
    (contributor as unknown as {props: {weights: {status: {complete?: unknown}}}}).props.weights
      .status.complete
  ).toBeDefined();
});
