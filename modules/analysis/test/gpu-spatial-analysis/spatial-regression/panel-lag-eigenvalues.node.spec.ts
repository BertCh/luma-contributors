// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {
  computeSpatialPanelLagMaximumLikelihoodOnCPU,
  CPUSpatialPanelLagMaximumLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation';
import {
  GPUSpatialPanelLagMaximumLikelihoodOperation,
  type SpatialPanelLagMaximumLikelihoodOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood-operation';
import {fitLeastSquares} from './math-review-regression-test-utils';
import {
  expectPanelLagCovarianceClose,
  expectPanelLagPredictionsClose
} from './panel-lag-comparisons.test-utils';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';
import fixtures from './spatial-panel-lag-maximum-likelihood-operation-fixtures.json';

const FIXTURE = fixtures[1];
const INPUT = {
  weights: {
    offsets: Uint32Array.from(FIXTURE.weights.offsets),
    neighbors: Uint32Array.from(FIXTURE.weights.neighbors),
    weights: new Float32Array(FIXTURE.weights.weights.length).fill(0.5)
  },
  predictors: Float32Array.from(FIXTURE.predictors),
  response: Float32Array.from(FIXTURE.response)
};
const OPTIONS = {
  periodCount: FIXTURE.periodCount,
  predictorCount: FIXTURE.predictorCount,
  effects: 'unit' as const
};
const PROPS: SpatialPanelLagMaximumLikelihoodOperationProps = {
  ...OPTIONS,
  entityCount: FIXTURE.entityCount,
  eigenvalues: 'eigenvalues',
  weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
  predictors: 'predictors',
  response: 'response',
  output: {
    coefficients: 'coefficients',
    covariance: 'covariance',
    summary: 'summary',
    status: 'status',
    fitted: 'fitted',
    residuals: 'residuals'
  }
};

it.each([
  {scene: 'valid', index: 9, relativeError: 0.005753996032898938},
  {scene: 'zero', index: 6, relativeError: 0.012433731481034325}
])('normalizes the lead GPU covariance failure by standard errors: $scene', ({
  scene,
  index,
  relativeError
}) => {
  const eigenvalues = scene === 'zero' ? new Float32Array(FIXTURE.entityCount) : undefined;
  const expected = computeSpatialPanelLagMaximumLikelihoodOnCPU({...INPUT, eigenvalues}, OPTIONS);
  const width = FIXTURE.predictorCount + 2;
  const actual = expected.covariance.slice();
  actual[index] *= 1 + relativeError;
  // Replay the magnitude of the real GPU discrepancy supplied by the lead. A small
  // off-diagonal covariance must be scaled by the two marginal standard errors.
  const row = Math.floor(index / width);
  const column = index % width;
  const scale = Math.sqrt(
    expected.covariance[row * width + row] * expected.covariance[column * width + column]
  );
  expect(Math.abs(actual[index] - expected.covariance[index]) / scale).toBeLessThan(0.001);
  expectPanelLagCovarianceClose(actual, expected.covariance, width);
});

it('covariance comparisons reject variance errors and unexpected NaN', () => {
  const expected = [1, 0, 0, 4];
  expect(() => expectPanelLagCovarianceClose([1.01, 0, 0, 4], expected, 2)).toThrow();
  expect(() => expectPanelLagCovarianceClose([1, NaN, 0, 4], expected, 2)).toThrow();
});

it('prediction comparisons use response scale near zero and reject errors and NaN', () => {
  expectPanelLagPredictionsClose([1e-6], [0], [1]);
  expect(() => expectPanelLagPredictionsClose([0.001], [0], [1])).toThrow();
  expect(() => expectPanelLagPredictionsClose([NaN], [0], [1])).toThrow();
});

async function executeCPU(
  eigenvalues: Float32Array,
  operation = new CPUSpatialPanelLagMaximumLikelihoodOperation(PROPS),
  input = INPUT,
  weightsStatus: Record<string, Uint32Array> = {}
) {
  const makeRows = (values: Float32Array | Uint32Array) => ({
    format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  return operation.execute({
    inputs: {
      offsets: makeRows(input.weights.offsets),
      neighbors: makeRows(input.weights.neighbors),
      weights: makeRows(input.weights.weights),
      predictors: makeRows(input.predictors),
      response: makeRows(input.response),
      eigenvalues: makeRows(eigenvalues),
      ...Object.fromEntries(
        Object.entries(weightsStatus).map(([field, values]) => [
          `weightsStatus_${field}`,
          makeRows(values)
        ])
      )
    }
  });
}

it.each([
  ['reference', false],
  ['tiled', false],
  ['reference', true],
  ['tiled', true]
] as const)('refuses candidate-only incomplete panel weights and recovers on the next CPU execution: %s getter=%s', async (fitMode, getterStatus) => {
  const eigenvalues = Float32Array.from({length: FIXTURE.entityCount}, (_, index) =>
    Math.cos((2 * Math.PI * index) / FIXTURE.entityCount)
  );
  const options = {...OPTIONS, fitMode};
  const input = {...INPUT, eigenvalues: fitMode === 'tiled' ? eigenvalues : undefined};
  const status = {overflow: Uint32Array.of(0), candidateOverflow: Uint32Array.of(1)};
  let reads = 0;
  let flagName = 'candidateOverflow';
  const statusNames = getterStatus
    ? Object.defineProperty({overflow: 'overflow'}, 'candidateOverflow', {
        enumerable: false,
        get() {
          reads++;
          return flagName;
        }
      })
    : {overflow: 'overflow', candidateOverflow: 'candidateOverflow'};
  const operation = new CPUSpatialPanelLagMaximumLikelihoodOperation({
    ...PROPS,
    ...options,
    eigenvalues: fitMode === 'tiled' ? 'eigenvalues' : undefined,
    weights: {
      ...PROPS.weights,
      status: statusNames
    }
  });
  flagName = 'lost';
  const expected = computeSpatialPanelLagMaximumLikelihoodOnCPU(input, options);
  expect(expected.status[0]).toBe(0);
  for (const candidateOverflow of [1, 0]) {
    status.candidateOverflow[0] = candidateOverflow;
    const pure = computeSpatialPanelLagMaximumLikelihoodOnCPU(
      {...input, weights: {...input.weights, status}},
      options
    );
    const graph = await executeCPU(eigenvalues, operation, INPUT, status);
    for (const actual of [pure, graph]) {
      expect(Array.from(actual['status'] as Uint32Array)).toEqual([candidateOverflow ? 1 : 0]);
      for (const port of [
        'coefficients',
        'covariance',
        'summary',
        'fitted',
        'residuals'
      ] as const) {
        const values = actual[port] as Float32Array;
        if (candidateOverflow) expect(values.every(Number.isNaN), port).toBe(true);
        else expectRegressionRowsClose(values, expected[port], 0);
      }
    }
  }
  if (getterStatus) {
    expect(reads).toBe(1);
    expect(operation.inputs['weightsStatus_candidateOverflow']).toBe('candidateOverflow');
  }
});

for (const fitMode of ['reference', 'tiled'] as const) {
  it.each([
    'operation',
    'contributor'
  ] as const)(`retains the getter-backed loss reader after construction on ${fitMode}: %s`, form => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const candidate = view('candidate', 'uint32', 1);
    const replacement = view('replacement', 'uint32', 1);
    let reads = 0;
    let current = form === 'operation' ? 'candidate' : candidate;
    const status = Object.create({
      get candidateOverflow() {
        reads++;
        return current;
      }
    });
    const views = {
      offsets: view('offsets', 'uint32', INPUT.weights.offsets.length),
      neighbors: view('neighbors', 'uint32', INPUT.weights.neighbors.length),
      weights: view('weights', 'float32', INPUT.weights.weights.length),
      eigenvalues: view('eigenvalues', 'float32', FIXTURE.entityCount),
      predictors: view('predictors', 'float32', INPUT.predictors.length),
      response: view('response', 'float32', INPUT.response.length),
      coefficients: view('coefficients', 'float32', FIXTURE.predictorCount + 3),
      covariance: view('covariance', 'float32', (FIXTURE.predictorCount + 2) ** 2),
      summary: view('summary', 'float32', 6),
      status: view('status', 'uint32', 1),
      fitted: view('fitted', 'float32', INPUT.response.length),
      residuals: view('residuals', 'float32', INPUT.response.length),
      weightsStatus_candidateOverflow: candidate
    };
    const output = {
      coefficients: views.coefficients,
      covariance: views.covariance,
      summary: views.summary,
      status: views.status,
      fitted: views.fitted,
      residuals: views.residuals
    };
    let operation: GPUSpatialPanelLagMaximumLikelihoodOperation | undefined;
    try {
      let contributor: GPUSpatialPanelLagMaximumLikelihood;
      if (form === 'operation') {
        operation = new GPUSpatialPanelLagMaximumLikelihoodOperation({
          ...PROPS,
          fitMode,
          eigenvalues: fitMode === 'tiled' ? 'eigenvalues' : undefined,
          weights: {...PROPS.weights, status}
        });
        current = 'replacement';
        contributor = operation.createContributor(
          name => views[name as keyof typeof views] as never
        ) as GPUSpatialPanelLagMaximumLikelihood;
      } else {
        contributor = new GPUSpatialPanelLagMaximumLikelihood({
          ...OPTIONS,
          fitMode,
          eigenvalues: fitMode === 'tiled' ? views.eigenvalues : undefined,
          predictors: views.predictors,
          response: views.response,
          weights: {
            offsets: views.offsets,
            neighbors: views.neighbors,
            weights: views.weights,
            status
          },
          output
        });
        current = replacement;
      }
      const nodes = contributor.getCommandNodes(graph);
      const bindings = nodes.flatMap(
        node => getWGSLKernelNodeRecord(graph, node.id)?.bindings ?? []
      );
      expect(
        bindings.filter(binding => binding.view === candidate && binding.access === 'read').length
      ).toBeGreaterThan(0);
      expect(bindings.filter(binding => binding.view === replacement)).toHaveLength(0);
      expect(reads).toBe(1);
    } finally {
      operation?.destroy();
      device.destroy();
    }
  });
}

it('snapshots spectrum presence and fit options but rereads spectrum contents on every execution (R1)', async () => {
  const props: SpatialPanelLagMaximumLikelihoodOperationProps = {
    ...PROPS,
    eigenvalues: 'spectrum',
    output: {...PROPS.output}
  };
  const cpu = new CPUSpatialPanelLagMaximumLikelihoodOperation(props);
  const gpu = new GPUSpatialPanelLagMaximumLikelihoodOperation(props);
  props.eigenvalues = undefined;
  const zero = await executeCPU(new Float32Array(FIXTURE.entityCount), cpu);
  expect(Array.from(zero['status'] as Uint32Array)).toEqual([0]);
  expect((zero['coefficients'] as Float32Array)[FIXTURE.predictorCount + 1]).toBeCloseTo(
    0.5383578412092436,
    6
  );
  props.effects = 'random';
  props.fitMode = 'reference';
  props.maximumSearchIterations = 1;
  props.output.coefficients = 'mutated';
  const eigenvalues = Float32Array.from({length: FIXTURE.entityCount}, (_, index) =>
    Math.cos((2 * Math.PI * index) / FIXTURE.entityCount)
  );
  const ring = await executeCPU(eigenvalues, cpu);
  expect((ring['coefficients'] as Float32Array)[FIXTURE.predictorCount + 1]).toBeCloseTo(
    0.3050408384,
    6
  );
  expect(cpu.inputs).toEqual(gpu.inputs);
  expect(cpu.inputs['eigenvalues']).toBe('spectrum');
  const contributor = constructRegressionOperationContributor(gpu, {
    offsets: {format: 'uint32', length: INPUT.weights.offsets.length},
    neighbors: {format: 'uint32', length: INPUT.weights.neighbors.length},
    weights: {format: 'float32', length: INPUT.weights.weights.length},
    predictors: {format: 'float32', length: INPUT.predictors.length},
    response: {format: 'float32', length: INPUT.response.length},
    eigenvalues: {format: 'float32', length: FIXTURE.entityCount}
  });
  expect((contributor as {fitMode?: string}).fitMode).toBe('tiled');
});

it('honors one explicit search step and withholds covariance in the pure and graph CPU fits (R2)', async () => {
  const eigenvalues = Float32Array.from({length: FIXTURE.entityCount}, (_, index) =>
    Math.cos((2 * Math.PI * index) / FIXTURE.entityCount)
  );
  const options = {...OPTIONS, maximumSearchIterations: 1};
  const pure = computeSpatialPanelLagMaximumLikelihoodOnCPU({...INPUT, eigenvalues}, options);
  const graph = await executeCPU(
    eigenvalues,
    new CPUSpatialPanelLagMaximumLikelihoodOperation({...PROPS, ...options})
  );
  for (const result of [pure, graph]) {
    expect(Array.from(result['status'] as Uint32Array)).toEqual([3]);
    expect((result['summary'] as Float32Array)[4]).toBeCloseTo((1.998 * (Math.sqrt(5) - 1)) / 2, 6);
    expect((result['covariance'] as Float32Array).every(Number.isNaN)).toBe(true);
    expect((result['fitted'] as Float32Array).every(Number.isFinite)).toBe(true);
  }
});

it('repeating a valid panel preserves its estimate, convergence and inverse-count covariance (R3)', async () => {
  const repeats = 1024;
  const eigenvalues = Float32Array.from({length: FIXTURE.entityCount}, (_, index) =>
    Math.cos((2 * Math.PI * index) / FIXTURE.entityCount)
  );
  const expected = computeSpatialPanelLagMaximumLikelihoodOnCPU({...INPUT, eigenvalues}, OPTIONS);
  const repeated = {
    ...INPUT,
    predictors: Float32Array.from(
      {length: INPUT.predictors.length * repeats},
      (_, index) => INPUT.predictors[index % INPUT.predictors.length]
    ),
    response: Float32Array.from(
      {length: INPUT.response.length * repeats},
      (_, index) => INPUT.response[index % INPUT.response.length]
    )
  };
  const periodCount = FIXTURE.periodCount * repeats;
  const pure = computeSpatialPanelLagMaximumLikelihoodOnCPU(
    {...repeated, eigenvalues},
    {...OPTIONS, periodCount}
  );
  const graph = await executeCPU(
    eigenvalues,
    new CPUSpatialPanelLagMaximumLikelihoodOperation({...PROPS, periodCount}),
    repeated
  );
  for (const actual of [pure, graph]) {
    expect(Array.from(actual['status'] as Uint32Array)).toEqual([0]);
    expectRegressionRowsClose(actual['coefficients'] as Float32Array, expected.coefficients, 1e-6);
    const covariance = actual['covariance'] as Float32Array;
    const scaled = Float32Array.from(covariance, value => value * repeats);
    expectPanelLagCovarianceClose(scaled, expected.covariance, FIXTURE.predictorCount + 2);
    const rhoIndex = FIXTURE.predictorCount + 1;
    expect((actual['coefficients'] as Float32Array)[rhoIndex]).toBeCloseTo(0.3050408384, 7);
    expect(Number.isFinite(covariance.at(-1)!)).toBe(true);
    expect(Math.abs((covariance.at(-1)! * repeats) / expected.covariance.at(-1)! - 1)).toBeLessThan(
      1e-5
    );
  }
});

it('CPU panel graph rejects a spectrum with the wrong entity count like the GPU constructor', async () => {
  const props = {...PROPS, id: 'review-panel'};
  const message = 'review-panel eigenvalues must hold one value per entity';
  const shapes = {
    offsets: {format: 'uint32' as const, length: INPUT.weights.offsets.length},
    neighbors: {format: 'uint32' as const, length: INPUT.weights.neighbors.length},
    weights: {format: 'float32' as const, length: INPUT.weights.weights.length},
    predictors: {format: 'float32' as const, length: INPUT.predictors.length},
    response: {format: 'float32' as const, length: INPUT.response.length},
    eigenvalues: {format: 'float32' as const, length: FIXTURE.entityCount - 1}
  };
  expect(() =>
    constructRegressionOperationContributor(
      new GPUSpatialPanelLagMaximumLikelihoodOperation(props),
      shapes
    )
  ).toThrowError(message);
  await expect(
    executeCPU(
      new Float32Array(FIXTURE.entityCount - 1),
      new CPUSpatialPanelLagMaximumLikelihoodOperation(props)
    )
  ).rejects.toEqual(new Error(message));
});

it.each([
  NaN,
  Infinity,
  -Infinity,
  1.01,
  -1.01
])('CPU panel graph rejects invalid eigenvalue %s with status 1 and NaN outputs', async value => {
  const eigenvalues = new Float32Array(FIXTURE.entityCount);
  eigenvalues[2] = value;
  const result = await executeCPU(eigenvalues);
  expect(Array.from(result['status'] as Uint32Array)).toEqual([1]);
  for (const port of ['coefficients', 'covariance', 'summary', 'fitted', 'residuals']) {
    expect((result[port] as Float32Array).every(Number.isNaN)).toBe(true);
  }
});

it('CPU panel graph uses the supplied spectrum in the objective, score and covariance', async () => {
  const eigenvalues = Float32Array.from({length: FIXTURE.entityCount}, (_, index) =>
    Math.cos((2 * Math.PI * index) / FIXTURE.entityCount)
  );
  // The reference function with no supplied spectrum evaluates the dense CSR determinant.
  const dense = computeSpatialPanelLagMaximumLikelihoodOnCPU(INPUT, OPTIONS);
  const actual = await executeCPU(eigenvalues);
  expect(Array.from(actual['status'] as Uint32Array)).toEqual([0]);
  for (const port of ['coefficients', 'covariance', 'summary', 'fitted', 'residuals'] as const) {
    expectRegressionRowsClose(
      (actual[port] as Float32Array).subarray(0, port === 'summary' ? 2 : undefined),
      dense[port].subarray(0, port === 'summary' ? 2 : undefined),
      1e-5
    );
  }
  // A zero spectrum removes the log determinant. Joint within-entity least squares on X and Wy
  // is an independent oracle for this objective, so merely validating the port cannot pass.
  const demean = (values: number[]) =>
    values.map(
      (value, row) =>
        value -
        Array.from(
          {length: FIXTURE.periodCount},
          (_, period) => values[period * FIXTURE.entityCount + (row % FIXTURE.entityCount)]
        ).reduce((sum, entry) => sum + entry, 0) /
          FIXTURE.periodCount
    );
  const target = demean(Array.from(INPUT.response));
  const columns = Array.from({length: FIXTURE.predictorCount}, (_, column) =>
    demean(
      Array.from(
        {length: target.length},
        (_, row) => INPUT.predictors[row * FIXTURE.predictorCount + column]
      )
    )
  );
  const lag = demean(
    Array.from(INPUT.response, (_, row) => {
      const entity = row % FIXTURE.entityCount;
      const start = row - entity;
      return (
        (INPUT.response[start + ((entity + 1) % FIXTURE.entityCount)] +
          INPUT.response[start + ((entity + FIXTURE.entityCount - 1) % FIXTURE.entityCount)]) /
        2
      );
    })
  );
  const oracle = fitLeastSquares(
    target.map((_, row) => [...columns.map(column => column[row]), lag[row]]),
    target
  );
  const changed = await executeCPU(new Float32Array(FIXTURE.entityCount));
  expect(Array.from(changed['status'] as Uint32Array)).toEqual([0]);
  const rhoIndex = FIXTURE.predictorCount + 1;
  expect((changed['coefficients'] as Float32Array)[rhoIndex]).toBeCloseTo(
    oracle.estimate.at(-1)!,
    6
  );
  expect(
    Math.abs((changed['coefficients'] as Float32Array)[rhoIndex] - dense.coefficients[rhoIndex])
  ).toBeGreaterThan(0.01);
  const variance = (changed['summary'] as Float32Array)[0];
  expect(variance).toBeCloseTo(oracle.squares / target.length, 6);
  const rows = INPUT.response.length;
  const expectedLikelihood =
    (-rows / 2) * (Math.log(variance) + 1 + Math.log(2 * Math.PI) + Math.log(rows));
  expect(Number.isFinite((changed['summary'] as Float32Array)[1])).toBe(true);
  expect((changed['summary'] as Float32Array)[1]).toBeCloseTo(expectedLikelihood, 4);
  const pure = computeSpatialPanelLagMaximumLikelihoodOnCPU(
    {...INPUT, eigenvalues: new Float32Array(FIXTURE.entityCount)},
    OPTIONS
  );
  expectRegressionRowsClose(changed['coefficients'] as Float32Array, pure.coefficients, 0);
});

it.each([
  {fitMode: 'reference' as const},
  {effects: 'random' as const},
  {eigenvalues: undefined, fitMode: 'tiled' as const},
  {fitMode: 'unknown' as never}
])('CPU panel graph rejects incompatible spectrum mode %j at construction', changes => {
  expect(() => new CPUSpatialPanelLagMaximumLikelihoodOperation({...PROPS, ...changes})).toThrow(
    /fitMode/
  );
});
