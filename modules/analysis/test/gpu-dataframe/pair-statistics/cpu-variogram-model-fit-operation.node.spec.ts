// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  CPUVariogramModelFitOperation,
  computeVariogramModelFitOnCPU
} from '../../../src/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation';
import {GPUVariogramModelFitOperation} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit-operation';
import {GPUVariogramModelFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {
  fitVariogramModel,
  type VariogramModelType
} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';

const LAGS = Array.from({length: 24}, (_, index) => (index + 1) * 0.25);
const PAIR_COUNTS = LAGS.map((_, index) => 50 + 3 * index);

it('final1b five-bin three-step variogram probe distinguishes budgets and rejects negative on both forms', async () => {
  const distances = Float32Array.of(1, 2, 3, 4, 5);
  const bins = {
    distances,
    semivariances: Float32Array.from(
      distances,
      distance => 0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 4.1323404))
    ),
    pairCounts: Uint32Array.of(10, 10, 10, 10, 10)
  };
  const rows = (values: Float32Array | Uint32Array) => ({
    format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const base = {
    id: 'three-step-fit',
    distances: 'distances',
    semivariances: 'semivariances',
    pairCounts: 'pairCounts',
    models: ['exponential'] as const,
    weighting: 'pairs' as const,
    rangeSteps: 3,
    output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
  };
  const results = [];
  for (const maximumIterations of [0, 32]) {
    const operation = new CPUVariogramModelFitOperation({...base, maximumIterations});
    const output = await operation.execute({
      inputs: {
        distances: rows(bins.distances),
        semivariances: rows(bins.semivariances),
        pairCounts: rows(bins.pairCounts)
      }
    });
    const parameters = output['parameters'] as Float32Array;
    const diagnostics = output['diagnostics'] as Float32Array;
    expect(parameters.every(Number.isFinite)).toBe(true);
    results.push({
      maximumIterations,
      range: parameters[2],
      bracket: diagnostics[3],
      status: (output['status'] as Uint32Array)[0]
    });
  }
  expect(results[0].status).toBe(3);
  expect(results[0].bracket).toBeGreaterThan(1);
  expect(results[1].bracket).toBeGreaterThan(0);
  expect(results[1].bracket).toBeLessThan(results[0].bracket * 1e-6);
  expect(results[0].range).not.toBe(results[1].range);
  expect(() => new CPUVariogramModelFitOperation({...base, maximumIterations: -1})).toThrow(
    'three-step-fit maximumIterations'
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    expect(
      () =>
        new GPUVariogramModelFit({
          id: 'three-step-fit',
          rangeSteps: 3,
          maximumIterations: -1,
          distances: createTransientView(graph, 'distances', 'float32', 5),
          semivariances: createTransientView(graph, 'semivariances', 'float32', 5),
          pairCounts: createTransientView(graph, 'pairCounts', 'uint32', 5),
          output: {
            parameters: createTransientView(graph, 'parameters', 'float32x4', 1),
            diagnostics: createTransientView(graph, 'diagnostics', 'float32x4', 1),
            status: createTransientView(graph, 'status', 'uint32', 1)
          }
        })
    ).toThrow('three-step-fit maximumIterations');
  } finally {
    device.destroy();
  }
  console.log(
    'final1b five-bin probe',
    results,
    'maximumIterations -1 rejected by CPU and GPU constructors'
  );
});

it.each([
  {rangeSteps: 2},
  {rangeSteps: 1025},
  {maximumIterations: -1},
  {maximumIterations: 65},
  {maximumIterations: 0.5}
])('final1 CPU variogram validates explicit search budget %j before executing', options => {
  const props = {
    id: 'budget-fit',
    distances: 'distances',
    semivariances: 'semivariances',
    pairCounts: 'pairCounts',
    output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'},
    ...options
  };
  expect(() => new CPUVariogramModelFitOperation(props)).toThrow(
    `budget-fit ${Object.keys(options)[0]}`
  );
  expect(() =>
    computeVariogramModelFitOnCPU({distances: [], semivariances: [], pairCounts: []}, options)
  ).toThrow(`variogram-model-fit ${Object.keys(options)[0]}`);
});

it.each([
  'pairs',
  'cressie'
] as const)('final1 CPU variogram honors explicit zero and one-step budgets in the function and operation (%s)', async weighting => {
  const distances = Float32Array.from({length: 20}, (_, index) => (index + 1) * 0.2);
  const bins = {
    distances,
    semivariances: Float32Array.from(
      distances,
      distance => 0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 2.2))
    ),
    pairCounts: Uint32Array.from({length: 20}, (_, index) => 10 + index)
  };
  const rows = (values: Float32Array | Uint32Array) => ({
    format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  for (const rangeSteps of [101, 200]) {
    for (const maximumIterations of [0, 1]) {
      const options = {
        models: ['exponential'] as const,
        weighting,
        rangeSteps,
        maximumIterations
      };
      const direct = computeVariogramModelFitOnCPU(bins, options);
      const model = fitVariogramModel(bins, {...options, model: 'exponential'});
      expect(Float32Array.of(model.nugget, model.sill, model.range, model.residual)).toEqual(
        direct.parameters
      );
      const props = {
        ...options,
        id: 'budget-fit',
        distances: 'distances',
        semivariances: 'semivariances',
        pairCounts: 'pairCounts',
        output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
      };
      const operation = new CPUVariogramModelFitOperation(props);
      props.maximumIterations = 32;
      props.rangeSteps = 3;
      const outputs = await operation.execute({
        inputs: {
          distances: rows(bins.distances),
          semivariances: rows(bins.semivariances),
          pairCounts: rows(bins.pairCounts)
        }
      });
      const expectedBracket =
        ((2 * Math.log((3 * distances[19]) / (distances[0] / 10))) / (rangeSteps - 1)) *
        ((Math.sqrt(5) - 1) / 2) ** maximumIterations;
      for (const result of [
        direct,
        {
          parameters: outputs['parameters'] as Float32Array,
          diagnostics: outputs['diagnostics'] as Float32Array,
          status: outputs['status'] as Uint32Array
        }
      ]) {
        expect(result.status[0]).toBe(3);
        expect(Number.isFinite(result.diagnostics[3])).toBe(true);
        expect(result.diagnostics[3]).toBeCloseTo(expectedBracket, 7);
        expect(result.parameters.every(Number.isFinite)).toBe(true);
      }
    }
  }
  const finished = computeVariogramModelFitOnCPU(bins, {
    models: ['exponential'],
    weighting,
    ...{maximumIterations: 32}
  });
  expect(finished.status[0]).toBe(0);
  expect(finished.diagnostics[3]).toBeLessThan(1e-5);
  expect(finished.diagnostics[3]).toBeGreaterThan(0);
  expect(finished.parameters[2]).toBeCloseTo(2.2, 5);
});

// The model families written out from their definitions (effective range: the spherical model
// reaches the sill there, the others reach 95% of it), independent of the source's shape function.
const SHAPES: Record<'spherical' | 'exponential' | 'gaussian', (scaled: number) => number> = {
  spherical: scaled => (scaled >= 1 ? 1 : 1.5 * scaled - 0.5 * scaled ** 3),
  exponential: scaled => 1 - Math.exp(-3 * scaled),
  gaussian: scaled => 1 - Math.exp(-3 * scaled * scaled)
};

function getBins(model: keyof typeof SHAPES, nugget: number, sill: number, range: number) {
  return {
    distances: LAGS,
    semivariances: LAGS.map(lag => nugget + sill * SHAPES[model](lag / range)),
    pairCounts: PAIR_COUNTS
  };
}

for (const mutation of ['names', 'models'] as const) {
  it(`final1b CPU variogram keeps original ${mutation}, output keys and row count`, async () => {
    const models: VariogramModelType[] = ['exponential', 'spherical'];
    const props = {
      distances: 'distances',
      semivariances: 'semivariances',
      pairCounts: 'pairCounts',
      models,
      weighting: 'pairs' as const,
      output: {parameters: 'p', diagnostics: 'd', status: 's'}
    };
    const operation = new CPUVariogramModelFitOperation(props);
    if (mutation === 'names') props.output.parameters = 'other';
    else models.splice(0, models.length, 'gaussian');
    const bins = getBins('exponential', 0.2, 1.5, 2.2);
    const rows = (values: Float32Array | Uint32Array) => ({
      format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
      length: values.length,
      chunks: [{values, length: values.length}],
      pack: () => values
    });
    const result = await operation.execute({
      inputs: {
        distances: rows(Float32Array.from(bins.distances)),
        semivariances: rows(Float32Array.from(bins.semivariances)),
        pairCounts: rows(Uint32Array.from(bins.pairCounts))
      }
    });
    expect(Object.keys(result).sort()).toEqual(['d', 'p', 's']);
    expect(result['p']).toHaveLength(8);
    expect(result['s']).toHaveLength(2);
    const parameters = result['p'] as Float32Array;
    expect(parameters[3]).toBeLessThan(1e-9);
    expect(parameters[7]).toBeGreaterThan(1e-3);
    expect(operation.outputs['p'].length).toBe(2);
  });
}

it('final1b GPU variogram operation retains model order, budget and view identities at lowering', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const models: VariogramModelType[] = ['exponential', 'spherical'];
  const props = {
    distances: 'distances',
    semivariances: 'semivariances',
    pairCounts: 'pairCounts',
    models,
    weighting: 'pairs' as const,
    rangeSteps: 3,
    maximumIterations: 0,
    output: {parameters: 'p', diagnostics: 'd', status: 's'}
  };
  const operation = new GPUVariogramModelFitOperation(props);
  models.splice(0, models.length, 'gaussian');
  props.maximumIterations = 1;
  props.rangeSteps = 7;
  props.output.parameters = 'other';
  try {
    const views = {
      distances: createTransientView(graph, 'distances', 'float32', 3),
      semivariances: createTransientView(graph, 'semivariances', 'float32', 3),
      pairCounts: createTransientView(graph, 'pairCounts', 'uint32', 3),
      parameters: createTransientView(graph, 'parameters', 'float32x4', 2),
      diagnostics: createTransientView(graph, 'diagnostics', 'float32x4', 2),
      status: createTransientView(graph, 'status', 'uint32', 2)
    };
    const contributor = operation.createContributor(
      name => views[name as keyof typeof views] as never
    ) as GPUVariogramModelFit;
    expect(contributor.models).toEqual(['exponential', 'spherical']);
    expect(contributor.maximumIterations).toBe(0);
    expect(contributor.rangeSteps).toBe(3);
    expect(contributor.props.output.parameters).toBe(views.parameters);
    expect(operation.outputs['parameters'].value).toBe('p');
    expect(operation.outputs['parameters'].length).toBe(2);
  } finally {
    operation.destroy();
    device.destroy();
  }
});

it('final1b direct GPU variogram snapshots mutable model arrays and getter-backed output maps', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const models: VariogramModelType[] = ['exponential', 'spherical'];
  const parameters = createTransientView(graph, 'parameters', 'float32x4', 2);
  let parameterView = parameters;
  const output = {
    get parameters() {
      return parameterView;
    },
    diagnostics: createTransientView(graph, 'diagnostics', 'float32x4', 2),
    status: createTransientView(graph, 'status', 'uint32', 2)
  };
  const props = {
    models,
    weighting: 'pairs' as const,
    distances: createTransientView(graph, 'distances', 'float32', 3),
    semivariances: createTransientView(graph, 'semivariances', 'float32', 3),
    pairCounts: createTransientView(graph, 'pairCounts', 'uint32', 3),
    output
  };
  const contributor = new GPUVariogramModelFit(props);
  models.reverse();
  parameterView = createTransientView(graph, 'replacement', 'float32x4', 1);
  try {
    expect(contributor.models).toEqual(['exponential', 'spherical']);
    expect(contributor.props.output.parameters === parameters).toBe(true);
    expect(
      getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).length
    ).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

for (const weighting of ['pairs', 'none', 'cressie'] as const) {
  it(`the CPU fit with ${weighting} weights recovers the parameters exact bins were drawn from`, () => {
    const truths = {
      spherical: {nugget: 0.3, sill: 2, range: 3.1},
      exponential: {nugget: 0.1, sill: 1.2, range: 2.4},
      gaussian: {nugget: 0.5, sill: 0.8, range: 1.7}
    };
    for (const [model, truth] of Object.entries(truths) as [
      keyof typeof SHAPES,
      (typeof truths)['spherical']
    ][]) {
      const result = computeVariogramModelFitOnCPU(
        getBins(model, truth.nugget, truth.sill, truth.range),
        {models: [model], weighting}
      );
      const [nugget, sill, range, residual] = result.parameters;
      // Exact bins: any weighting has its minimum, zero, at the truth.
      expect(nugget).toBeCloseTo(truth.nugget, 4);
      expect(sill).toBeCloseTo(truth.sill, 4);
      expect(range).toBeCloseTo(truth.range, 4);
      expect(residual).toBeLessThan(1e-9);
      expect(result.status[0]).toBe(0);
      const [binCount, unweightedResidual, informationCriterion, bracket] = result.diagnostics;
      expect(binCount).toBe(LAGS.length);
      expect(unweightedResidual).toBeLessThan(1e-9);
      // AIC of a three-parameter fit: n * ln(max(RSS / n, 1e-30)) + 6, rounded to float32.
      expect(informationCriterion).toBe(
        Math.fround(
          LAGS.length * Math.log(Math.max(result.diagnostics[1] / LAGS.length, 1e-30)) + 6
        )
      );
      expect(bracket).toBeGreaterThanOrEqual(0);
      expect(bracket).toBeLessThan(1e-9);
    }
  });
}

it('the CPU fit writes one row per model in the order of models', () => {
  const models: VariogramModelType[] = ['gaussian', 'exponential', 'spherical'];
  const result = computeVariogramModelFitOnCPU(getBins('exponential', 0.2, 1.5, 2.2), {
    models,
    weighting: 'pairs'
  });
  expect(result.parameters.length).toBe(12);
  expect(result.diagnostics.length).toBe(12);
  expect(result.status.length).toBe(3);
  // Only the family the bins came from reproduces them.
  const residuals = models.map((_, row) => result.parameters[row * 4 + 3]);
  expect(residuals[1]).toBeLessThan(1e-9);
  expect(residuals[0]).toBeGreaterThan(1e-3);
  expect(residuals[2]).toBeGreaterThan(1e-3);
});

it.each([
  'semivariances',
  'pairCounts'
] as const)('the CPU fit rejects unequal %s lengths instead of fitting the usable prefix', port => {
  const bins = getBins('exponential', 0.2, 1.5, 2.2);
  for (const length of [LAGS.length - 1, LAGS.length + 1]) {
    const values = Array.from({length}, (_, index) => bins[port][index] ?? 10);
    expect(() => computeVariogramModelFitOnCPU({...bins, [port]: values})).toThrow(
      'variogram-model-fit invalid model or bin topology'
    );
    const malformed = {...bins, [port]: values};
    const operation = new CPUVariogramModelFitOperation({
      distances: 'distances',
      semivariances: 'semivariances',
      pairCounts: 'pairCounts',
      output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
    });
    const makeRows = (values: Float32Array | Uint32Array) => ({
      format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
      length: values.length,
      chunks: [{values, length: values.length}],
      pack: () => values
    });
    expect(() =>
      operation.execute({
        inputs: {
          distances: makeRows(Float32Array.from(malformed.distances)),
          semivariances: makeRows(Float32Array.from(malformed.semivariances)),
          pairCounts: makeRows(Uint32Array.from(malformed.pairCounts))
        }
      })
    ).toThrow('variogram-model-fit invalid model or bin topology');
  }
});

it.each([
  'variogram-model-fit',
  'review-variogram-fit'
])('both variogram graph entry points identify unequal bins with operation ID %s', id => {
  const device = createNullWebGPUDevice();
  const props = {
    id,
    distances: 'distances',
    semivariances: 'semivariances',
    pairCounts: 'pairCounts',
    output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
  };
  const message = `${id} invalid model or bin topology`;
  const makeRows = (values: Float32Array | Uint32Array) => ({
    format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  try {
    for (const port of ['semivariances', 'pairCounts'] as const) {
      const graph = new GPUCommandGraph(device);
      const bins = getBins('exponential', 0.2, 1.5, 2.2);
      bins[port] = bins[port].slice(1);
      const operation = new GPUVariogramModelFitOperation(props);
      try {
        const shapes = {
          distances: {format: 'float32' as const, length: bins.distances.length},
          semivariances: {format: 'float32' as const, length: bins.semivariances.length},
          pairCounts: {format: 'uint32' as const, length: bins.pairCounts.length},
          ...operation.outputs
        };
        const views = Object.fromEntries(
          Object.entries(shapes).map(([name, shape]) => [
            name,
            createTransientView(graph, name, shape.format, shape.length)
          ])
        );
        expect(() => operation.createContributor(name => views[name] as never)).toThrow(message);
      } finally {
        operation.destroy();
      }
      const twin = new CPUVariogramModelFitOperation(props);
      expect(() =>
        twin.execute({
          inputs: {
            distances: makeRows(Float32Array.from(bins.distances)),
            semivariances: makeRows(Float32Array.from(bins.semivariances)),
            pairCounts: makeRows(Uint32Array.from(bins.pairCounts))
          }
        })
      ).toThrow(message);
    }
  } finally {
    device.destroy();
  }
});

it('the CPU fit reports status 1 and NaN rows without two usable bins at two lags', () => {
  const cases = [
    // No pairs anywhere.
    {distances: [1, 2, 3], semivariances: [1, 2, 3], pairCounts: [0, 0, 0]},
    // One usable bin: the others have a non-finite value, a non-positive lag or no pairs.
    {distances: [1, 2, 0, 4], semivariances: [1, Number.NaN, 3, 4], pairCounts: [5, 5, 5, 0]},
    // Two usable bins at the same lag.
    {distances: [2, 2], semivariances: [1, 1.5], pairCounts: [5, 5]}
  ];
  for (const bins of cases) {
    const result = computeVariogramModelFitOnCPU(bins, {models: ['spherical', 'gaussian']});
    expect(Array.from(result.status)).toEqual([1, 1]);
    expect(result.parameters.every(Number.isNaN)).toBe(true);
    expect(result.diagnostics.every(Number.isNaN)).toBe(true);
  }
});

it('the CPU fit reports status 3 when the range runs to the end of the search domain', () => {
  // Semivariance still rising linearly at the largest lag: no sill inside three times that lag.
  const bins = {
    distances: LAGS,
    semivariances: LAGS.map(lag => 0.5 * lag),
    pairCounts: PAIR_COUNTS
  };
  const result = computeVariogramModelFitOnCPU(bins, {models: ['spherical'], weighting: 'none'});
  expect(result.status[0]).toBe(3);
  expect(result.parameters[2]).toBeCloseTo(3 * LAGS[LAGS.length - 1], 3);
});

it('the CPU fit reports status 2 when a fitted value does not fit float32', () => {
  const result = computeVariogramModelFitOnCPU(
    {distances: LAGS, semivariances: LAGS.map(lag => 1e39 * lag), pairCounts: PAIR_COUNTS},
    {models: ['exponential'], weighting: 'none'}
  );
  expect(result.status[0]).toBe(2);
  expect(result.parameters.every(Number.isNaN)).toBe(true);
});

it('both operations reject unknown models and weightings at construction, naming the ID', () => {
  const props = {
    id: 'fit',
    distances: 'distances',
    semivariances: 'semivariances',
    pairCounts: 'pairCounts',
    output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
  };
  expect(() => new CPUVariogramModelFitOperation({...props, models: []})).toThrow(
    'fit models must name one to five model families'
  );
  expect(
    () => new CPUVariogramModelFitOperation({...props, weighting: 'inverse' as 'pairs'})
  ).toThrow("fit weighting must be 'cressie', 'pairs' or 'none'");
  expect(
    () => new CPUVariogramModelFitOperation({...props, models: ['cubic' as 'spherical']})
  ).toThrow();
  // The two classes take the same properties and declare the same graph values.
  const cpu = new CPUVariogramModelFitOperation(props);
  const gpu = new GPUVariogramModelFitOperation(props);
  expect(cpu.id).toBe(gpu.id);
  const getNames = (contract: typeof cpu.contract) =>
    contract.values.map(use => `${use.access}:${String(use.value)}`).sort();
  expect(getNames(cpu.contract)).toEqual(getNames(gpu.contract));
  expect(cpu.contract.outputs).toEqual(gpu.contract.outputs);
});

it.each([
  'distances',
  'semivariances',
  'pairCounts'
] as const)('both operations reject invalid graph names for input %s at construction', port => {
  for (const value of ['', undefined, null, 0, false, {id: 'bins'}, ['bins']]) {
    const props = {
      id: 'fit',
      distances: 'distances',
      semivariances: 'semivariances',
      pairCounts: 'pairCounts',
      [port]: value as string,
      output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
    };
    for (const operationClass of [GPUVariogramModelFitOperation, CPUVariogramModelFitOperation]) {
      expect(() => new operationClass(props)).toThrow(
        `fit inputs.${port} must be a non-empty graph value name`
      );
    }
  }
});

it.each([
  'parameters',
  'diagnostics',
  'status'
] as const)('both operations reject invalid graph names for output %s before names become object keys', port => {
  for (const value of ['', undefined, null, 0, false, {id: 'fitted'}, ['fitted']]) {
    const props = {
      id: 'fit',
      distances: 'distances',
      semivariances: 'semivariances',
      pairCounts: 'pairCounts',
      output: {
        parameters: 'parameters',
        diagnostics: 'diagnostics',
        status: 'status',
        [port]: value as string
      }
    };
    for (const operationClass of [GPUVariogramModelFitOperation, CPUVariogramModelFitOperation]) {
      expect(() => new operationClass(props)).toThrow(
        `fit outputs.${port}.value must be a non-empty graph value name`
      );
    }
  }
});

it.each([
  ['parameters', 'diagnostics'],
  ['parameters', 'status'],
  ['diagnostics', 'status']
] as const)('both operations reject output/output collisions between %s and %s before key collapse', (first, second) => {
  const props = {
    id: 'fit-collision',
    distances: 'distances',
    semivariances: 'semivariances',
    pairCounts: 'pairCounts',
    output: {
      parameters: 'parameters',
      diagnostics: 'diagnostics',
      status: 'status',
      [first]: 'x',
      [second]: 'x'
    }
  };
  const message = `fit-collision outputs.${second} must name a graph value ("x") that no other port uses`;
  for (const Operation of [GPUVariogramModelFitOperation, CPUVariogramModelFitOperation]) {
    expect(() => new Operation(props)).toThrowError(new Error(message));
  }
});

for (const output of ['parameters', 'diagnostics', 'status'] as const) {
  it.each([
    'distances',
    'semivariances',
    'pairCounts'
  ] as const)(`both operations reject input/output collisions for ${output} and %s`, input => {
    const props = {
      id: 'fit-collision',
      distances: 'distances',
      semivariances: 'semivariances',
      pairCounts: 'pairCounts',
      output: {
        parameters: 'parameters',
        diagnostics: 'diagnostics',
        status: 'status',
        [output]: input
      }
    };
    const message = `fit-collision outputs.${output} must name a graph value ("${input}") that no other port uses`;
    for (const Operation of [GPUVariogramModelFitOperation, CPUVariogramModelFitOperation]) {
      expect(() => new Operation(props)).toThrowError(new Error(message));
    }
  });
}

it('both operations permit shared read-only input names and retain all three distinct outputs', () => {
  const props = {
    distances: 'bins',
    semivariances: 'bins',
    pairCounts: 'pairCounts',
    output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
  };
  for (const Operation of [GPUVariogramModelFitOperation, CPUVariogramModelFitOperation]) {
    const operation = new Operation(props);
    expect(Object.keys(operation.contract.outputs!)).toEqual([
      'parameters',
      'diagnostics',
      'status'
    ]);
  }
});
