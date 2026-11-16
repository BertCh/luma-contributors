// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect, WgslExec, WgslParser} from 'wgsl_reflect';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import reference from '../../oracles/fixtures/generalized-gwr-aicc.json';

function setup(rows = 48) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `selection-${serial++}`, format, length);
  const props = {
    id: 'selection-contract',
    family: 'poisson' as const,
    positions: view('float32x2', rows),
    predictors: view('float32', rows),
    predictorCount: 1,
    response: view('float32', rows),
    offset: view('float32', rows),
    mask: view('uint32', rows),
    parameters: view('float32', 2),
    output: {
      coefficients: view('float32', rows * 2),
      fitted: view('float32', rows),
      status: {codes: view('uint32', rows), iterationCount: view('uint32', rows)}
    }
  };
  const selection = (maximumIterations = 2) => ({
    method: 'golden-section' as const,
    maximumIterations,
    parameters: view('float32', 3),
    output: {
      scores: view('float32', 2 * maximumIterations),
      bandwidth: view('float32', 1),
      status: view('uint32', 1),
      evaluatedBandwidths: view('float32', 2 * maximumIterations)
    },
    status: {
      converged: view('uint32', 1),
      iterationCount: view('uint32', 1),
      iterationLimitReached: view('uint32', 1)
    }
  });
  return {device, graph, view, props, selection};
}

it('AICc grid/golden kernels parse across families, adaptive modes and optional index/input/output bindings', () => {
  const context = setup();
  try {
    for (const family of ['poisson', 'logistic'] as const) {
      for (const bandwidthMode of ['fixed', 'adaptive'] as const) {
        for (const kernel of ['gaussian', 'bisquare'] as const) {
          for (const indexed of [false, true]) {
            const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
              ...context.props,
              id: `${family}-${bandwidthMode}-${kernel}-${indexed}`,
              family,
              bandwidthMode,
              kernel,
              indexGridSize: indexed ? [4, 4] : undefined,
              bandwidthSelection: context.selection(),
              inference: {
                standardErrors: context.view('float32', 96),
                influence: context.view('float32', 48)
              }
            });
            const nodes = contributor.getCommandNodes(context.graph);
            expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
            for (const source of getGPUCommandNodeKernelSources(nodes)) {
              expect(auditWGSLModule(source.source, source.constants), source.id).toEqual([]);
              expect(new WgslReflect(source.source).storage.length, source.id).toBeLessThanOrEqual(
                8
              );
            }
            const fitNodes = nodes.filter(
              node => node.id.includes('-candidate-') && /-fit$/.test(node.id)
            );
            expect(fitNodes).toHaveLength(4);
            for (const node of fitNodes) expect(node.condition?.source).toBe('gpu');
          }
        }
      }
    }
    const grid = context.selection().output;
    const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
      ...context.props,
      id: 'grid-aicc',
      bandwidthSelection: {criterion: 'aicc', bandwidths: [2, 3, 4, 5], output: grid}
    });
    expect(contributor.getCommandNodes(context.graph).length).toBeLessThanOrEqual(
      contributor.estimatedCommandNodeCount
    );
  } finally {
    context.device.destroy();
  }
});

it('golden search enforces named budgets, row limits, shapes, disjoint outputs and adaptive candidate capacity', () => {
  const context = setup();
  const {props, selection, view} = context;
  const construct = (bandwidthSelection: object) =>
    new GPUGeneralizedGeographicallyWeightedRegression({
      ...props,
      bandwidthSelection: bandwidthSelection as ReturnType<typeof selection>
    });
  try {
    for (const maximumIterations of [0, 65, 1.5]) {
      expect(() => construct({...selection(), maximumIterations})).toThrow(
        /selection-contract.*maximumIterations.*\[1, 64\]/
      );
    }
    expect(() => construct({...selection(), parameters: view('float32', 2)})).toThrow(
      /selection-contract.*parameters.*3 rows/
    );
    expect(() =>
      construct({...selection(), output: {...selection().output, scores: view('float32', 3)}})
    ).toThrow(/selection-contract.*scores.*4 rows/);
    expect(() => construct({...selection(), status: {converged: view('float32', 1)}})).toThrow(
      /selection-contract/
    );
    expect(() => construct({...selection(), criterion: 'loo-deviance'})).toThrow(/criterion.*aicc/);
    const selected = selection();
    expect(() =>
      construct({...selected, output: {...selected.output, bandwidth: selected.parameters}})
    ).toThrow();
    expect(
      () =>
        new GPUGeneralizedGeographicallyWeightedRegression({
          ...props,
          bandwidthMode: 'adaptive',
          maximumNeighborCount: 16,
          bandwidthSelection: {
            bandwidths: [16, 17],
            output: {
              scores: view('float32', 2),
              bandwidth: view('float32', 1),
              status: view('uint32', 1)
            }
          }
        })
    ).toThrow(/bandwidths\[1\].*\[1, 16\]/);
    const maximum = new GPUGeneralizedGeographicallyWeightedRegression({
      ...props,
      bandwidthSelection: selection(64)
    });
    expect(maximum.getCommandNodes(context.graph).length).toBeLessThanOrEqual(
      maximum.estimatedCommandNodeCount
    );
  } finally {
    context.device.destroy();
  }
  const oversized = setup(4097);
  try {
    expect(
      () =>
        new GPUGeneralizedGeographicallyWeightedRegression({
          ...oversized.props,
          bandwidthSelection: oversized.selection()
        })
    ).toThrow(/selection-contract.*AICc rows.*4096/);
  } finally {
    oversized.device.destroy();
  }
});

it('committed NumPy fixture uses the mgwr Poisson/Binomial AICc correction with masks and positive denominator', () => {
  const count = reference.mask.filter(Boolean).length;
  for (const model of reference.models) {
    const expected =
      model.deviance +
      2 * model.trace +
      (2 * model.trace * (model.trace + 1)) / (count - model.trace - 1);
    expect(model.score).toBeCloseTo(expected, 10);
    if (model.iterationCount !== null) {
      expect(model.iterationCount).toBeLessThanOrEqual(reference.provenance.maximumIterations);
      expect(model.history).toHaveLength(2 * model.iterationCount);
    }
    const evaluations = model.mode === 'adaptive' ? model.candidates : model.history;
    if (model.mode === 'adaptive') {
      expect(evaluations.map(candidate => candidate.bandwidth)).toEqual(
        Array.from(
          {length: model.bounds[1] - model.bounds[0] + 1},
          (_, index) => model.bounds[0] + index
        )
      );
    }
    expect(model.bandwidth).toBe(
      evaluations.reduce((best, candidate) => (candidate.score < best.score ? candidate : best))
        .bandwidth
    );
  }
});

it('the grid node allowance bounds published primitive nodes at the maximum indexed row count', () => {
  const context = setup(1048576);
  try {
    const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
      ...context.props,
      predictorCount: 24,
      predictors: context.view('float32', 1048576 * 24),
      indexGridSize: [1024, 1024],
      bandwidthMode: 'adaptive',
      output: {...context.props.output, coefficients: context.view('float32', 1048576 * 25)},
      inference: {
        standardErrors: context.view('float32', 1048576 * 25),
        influence: context.view('float32', 1048576)
      }
    });
    const nodes = contributor.getCommandNodes(context.graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  } finally {
    context.device.destroy();
  }
});

it('rejects overlapping final, inference and search outputs while allowing adjacent cross-stage slices', () => {
  const context = setup();
  const {graph, props, selection, view} = context;
  try {
    const inference = {standardErrors: view('float32', 96), influence: view('float32', 48)};
    for (const target of [
      props.output.coefficients,
      props.output.fitted,
      props.output.status.codes,
      props.output.status.iterationCount,
      inference.standardErrors,
      inference.influence
    ]) {
      const converged = graph.createDataView(target.buffer, {
        format: 'uint32',
        byteOffset: target.byteOffset,
        length: 1
      });
      const selected = selection();
      expect(
        () =>
          new GPUGeneralizedGeographicallyWeightedRegression({
            ...props,
            inference,
            bandwidthSelection: {...selected, status: {...selected.status, converged}}
          })
      ).toThrow(/selection-contract.*output.*overlap/);
    }
    const selected = selection();
    expect(
      () =>
        new GPUGeneralizedGeographicallyWeightedRegression({
          ...props,
          bandwidthSelection: {
            ...selected,
            status: {...selected.status, converged: selected.output.status}
          }
        })
    ).toThrow(/selection-contract.*overlap/);
    expect(
      () =>
        new GPUGeneralizedGeographicallyWeightedRegression({
          ...props,
          output: {...props.output, fitted: props.predictors}
        })
    ).toThrow(/selection-contract.*inputs/);
    const storage = view('uint32', 49);
    const codes = graph.createDataView(storage.buffer, {format: 'uint32', length: 48});
    const converged = graph.createDataView(storage.buffer, {
      format: 'uint32',
      byteOffset: 48 * 4,
      length: 1
    });
    expect(
      () =>
        new GPUGeneralizedGeographicallyWeightedRegression({
          ...props,
          output: {...props.output, status: {...props.output.status, codes}},
          bandwidthSelection: {...selected, status: {...selected.status, converged}}
        })
    ).not.toThrow();
  } finally {
    context.device.destroy();
  }
});

/** Execute emitted control kernels with independent scores, without duplicating their search logic. */
function runSearchControl(
  bandwidthMode: 'fixed' | 'adaptive',
  bounds: number[],
  score: (bandwidth: number) => number,
  maximumIterations = 8
) {
  const context = setup();
  try {
    const selected = context.selection(maximumIterations);
    const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
      ...context.props,
      family: 'logistic',
      bandwidthMode,
      kernel: 'bisquare',
      bandwidthSelection: selected
    });
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(context.graph));
    const buffers: Record<string, Float32Array | Uint32Array> = {
      scores: new Float32Array(2 * maximumIterations),
      history: new Float32Array(2 * maximumIterations),
      bounds: Float32Array.from(bounds),
      state: new Float32Array(32),
      dispatch: new Uint32Array(12),
      focalDispatch: new Uint32Array(3),
      source: Float32Array.from([1, 1e-6]),
      parameters: new Float32Array(3),
      bandwidth: new Float32Array(1),
      status: new Uint32Array(1),
      converged: new Uint32Array(1),
      iterationCount: new Uint32Array(1),
      iterationLimitReached: new Uint32Array(1)
    };
    const execute = (suffix: string) => {
      const source = sources.find(candidate => candidate.id.endsWith(suffix))!;
      const reflection = new WgslReflect(source.source);
      const bindings = Object.fromEntries(
        reflection.storage.map(binding => [binding.binding, buffers[binding.name]])
      );
      // The interpreter does not implement WGSL phony assignments. These generated no-ops
      // only preserve binding layout; the actual control statements remain unchanged.
      const interpretedSource = source.source
        .replace(/^\s*_\s*=.*;$/gm, '')
        .replace(/@workgroup_size\((\d+)\)/, '@workgroup_size($1, 1, 1)');
      const interpreter = new WgslExec(new WgslParser().parse(interpretedSource));
      interpreter.dispatchWorkgroups('main', 1, {0: bindings}, {constants: source.constants});
    };
    execute('-selection-init');
    for (let iteration = 0; iteration < maximumIterations; iteration++) {
      if (buffers['dispatch'][0] === 0) break;
      for (let interior = 0; interior < 2; interior++) {
        const candidate = 2 * iteration + interior;
        execute(`-candidate-${candidate}-parameters`);
        buffers['scores'][candidate] = score(buffers['history'][candidate]);
        // The score stage advances the shared cursor; independent scores replace that stage here.
        buffers['parameters'][2] += 1;
      }
      execute(`-search-${iteration}-iterate`);
    }
    execute('-select-bandwidth');
    return buffers;
  } finally {
    context.device.destroy();
  }
}

it('emitted adaptive control resolves the remaining integers instead of converging on a rounded duplicate', () => {
  // Every integer was independently fitted by NumPy; no search policy supplies the expected winner.
  const model = reference.models.find(
    candidate =>
      candidate.family === 'logistic' &&
      candidate.kernel === 'bisquare' &&
      candidate.mode === 'adaptive'
  )!;
  const result = runSearchControl('adaptive', [40, 44, 0.0001], bandwidth => {
    const evaluation = model.candidates.find(candidate => candidate.bandwidth === bandwidth);
    expect(evaluation).toBeDefined();
    return evaluation!.score;
  });
  expect(result['bandwidth'][0]).toBe(44);
  expect(result['converged'][0]).toBe(1);
  for (const bandwidth of [40, 41, 42, 43, 44])
    expect(Array.from(result['history'])).toContain(bandwidth);
  const limited = runSearchControl('adaptive', [40, 44, 0.0001], bandwidth => 100 - bandwidth, 1);
  expect(limited['converged'][0]).toBe(0);
  expect(limited['iterationLimitReached'][0]).toBe(1);
  expect(limited['iterationCount'][0]).toBe(1);
});

it('emitted fixed control explores the original bracket after two invalid interiors', () => {
  const result = runSearchControl('fixed', [1, 150, 0.0001], bandwidth => {
    if (bandwidth <= 100) return NaN;
    // Closed-form local logistic AICc for the eight-row, two-site balanced-predictor scene.
    const weight = (1 - (100 / bandwidth) ** 2) ** 2;
    const trace = 4 / (1 + weight);
    return 16 * Math.log1p(weight) + 2 * trace + (2 * trace * (trace + 1)) / (7 - trace);
  });
  expect(result['status'][0]).toBe(0);
  expect(result['bandwidth'][0]).toBeGreaterThan(100);
  expect(Array.from(result['history'])).toContain(150);
  expect(Array.from(result['history'])).toContain(75.5);
  expect(Array.from(result['history'])).toContain(112.75);
  expect(result['converged'][0]).toBe(0);
  expect(result['iterationLimitReached'][0]).toBe(1);
});

it('adaptive control preserves the best independently scored evaluation for every committed family/kernel model', () => {
  for (const model of reference.models.filter(candidate => candidate.mode === 'adaptive')) {
    const result = runSearchControl('adaptive', model.bounds, bandwidth => {
      const evaluation = model.candidates.find(candidate => candidate.bandwidth === bandwidth);
      expect(evaluation).toBeDefined();
      return evaluation!.score;
    });
    // These integer score curves need not be unimodal: the documented local search can
    // discard a better global candidate, but must preserve the best candidate it did evaluate.
    const evaluated = model.candidates.filter(candidate =>
      Array.from(result['history']).includes(candidate.bandwidth)
    );
    const best = evaluated.reduce((previous, candidate) =>
      candidate.score < previous.score ? candidate : previous
    );
    const chosen = model.candidates.find(
      candidate => candidate.bandwidth === result['bandwidth'][0]
    )!;
    expect(chosen.score - best.score, `${model.family}/${model.kernel}`).toBeLessThan(2e-5);
    expect(result['converged'][0]).toBe(1);
    expect(result['iterationLimitReached'][0]).toBe(0);
  }
});
