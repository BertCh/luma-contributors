// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUProbitRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression';
import {GPUProbitRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression-operation';
import {
  CPUProbitRegressionOperation,
  computeProbitRegressionOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-probit-regression-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUProbitConvergenceThreshold,
  MAXIMUM_PROBIT_CONVERGENCE_TOLERANCE,
  MAXIMUM_PROBIT_TOLERANCE,
  PROBIT_STEP_VERDICT_WGSL,
  resolveProbitConvergenceTolerance
} from '../../../src/gpu-spatial-analysis/spatial-regression/probit-convergence';
import {createProbitConvergenceFixture} from './probit-convergence-fixture';

function createProps(tolerance: number) {
  return {
    id: 'certificate-test',
    rowCount: 40,
    predictorCount: 1,
    tolerance,
    maximumIterations: 8,
    predictors: 'predictors',
    response: 'response',
    output: {
      coefficients: 'coefficients',
      standardErrors: 'errors',
      fitted: 'fitted',
      marginalEffects: 'effects',
      summary: 'summary',
      status: 'status'
    }
  };
}

it('SSR-109 accepts the tolerance ceiling and small positive requests before float32 rounding', () => {
  for (const tolerance of [1e-2, 1e-46, Number.MIN_VALUE]) {
    for (const Operation of [CPUProbitRegressionOperation, GPUProbitRegressionOperation]) {
      expect(() => new Operation(createProps(tolerance))).not.toThrow();
    }
    expect(resolveProbitConvergenceTolerance('test', tolerance)).toBe(Math.min(tolerance, 1e-5));
    expect(getGPUProbitConvergenceThreshold('test', tolerance)).toBeGreaterThan(0);
  }
  expect(MAXIMUM_PROBIT_TOLERANCE).toBe(1e-2);
  expect(MAXIMUM_PROBIT_CONVERGENCE_TOLERANCE).toBe(1e-5);
});

it('SSR-109 rejects above-ceiling tolerance in every form and names the limit and request', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  try {
    const direct = {
      id: 'certificate-test',
      predictorCount: 1,
      predictors: view('predictors', 'float32', 40),
      response: view('response', 'float32', 40),
      output: {
        coefficients: view('coefficients', 'float32', 2),
        standardErrors: view('errors', 'float32', 2),
        fitted: view('fitted', 'float32', 40),
        marginalEffects: view('effects', 'float32', 1),
        summary: view('summary', 'float32', 4),
        status: view('status', 'uint32', 1)
      }
    };
    expect(() => new GPUProbitRegression({...direct, tolerance: 1e-2})).not.toThrow();
    for (const tolerance of [0.010000000000000002, 1e39, Number.MAX_VALUE]) {
      const message = `certificate-test tolerance must be a positive finite number at most 0.01; received ${tolerance}`;
      expect(() => new GPUProbitRegression({...direct, tolerance})).toThrow(message);
      for (const Operation of [CPUProbitRegressionOperation, GPUProbitRegressionOperation]) {
        expect(() => new Operation(createProps(tolerance))).toThrow(message);
      }
      expect(() =>
        computeProbitRegressionOnCPU(createProbitConvergenceFixture(), {
          predictorCount: 1,
          tolerance
        })
      ).toThrow(message.replace('certificate-test', 'computeProbitRegressionOnCPU'));
    }
  } finally {
    device.destroy();
  }
});

it('SSR-109 both operation forms reject invalid tolerance at construction', () => {
  for (const tolerance of [0, -0, -1, -1e-46, NaN, Infinity, -Infinity]) {
    for (const Operation of [CPUProbitRegressionOperation, GPUProbitRegressionOperation]) {
      expect(() => new Operation(createProps(tolerance))).toThrow(/certificate-test.*tolerance/);
    }
    expect(() =>
      computeProbitRegressionOnCPU(createProbitConvergenceFixture(), {predictorCount: 1, tolerance})
    ).toThrow(/tolerance/);
  }
});

it('SSR-109 GPU operations emit the shared capped full-direction certificate', () => {
  for (const [requested, expected] of [
    [0.01, 1e-5],
    [1e-3, 1e-5],
    [1e-7, 1e-7],
    // Below the smallest normal float32 the kernel threshold is floored at 2^-126, never 0.0.
    [1e-46, 2 ** -126]
  ]) {
    const device = createNullWebGPUDevice();
    const operation = new GPUProbitRegressionOperation(createProps(requested));
    try {
      const graph = new GPUCommandGraph(device);
      const views = {
        predictors: createTransientView(graph, 'predictors', 'float32', 40),
        response: createTransientView(graph, 'response', 'float32', 40),
        ...Object.fromEntries(
          Object.entries(operation.outputs).map(([name, shape]) => [
            name,
            createTransientView(graph, name, shape.format, shape.length)
          ])
        )
      };
      const contributor = operation.createContributor(
        name => views[name as keyof typeof views] as never
      );
      expect(contributor).toBeInstanceOf(GPUProbitRegression);
      const nodes = contributor.getCommandNodes(graph);
      const sources = getGPUCommandNodeKernelSources(nodes);
      const decisions = sources.filter(kernel => kernel.id.endsWith('-decide'));
      expect(decisions).toHaveLength(8);
      for (const kernel of decisions) {
        const threshold = kernel.source.match(
          /getProbitStepVerdict\(scale, direction, candidate, ([\deE.+-]+)\)/
        );
        expect(threshold).not.toBeNull();
        expect(Number(threshold![1])).toBe(Math.fround(expected));
        expect(kernel.source).toContain(PROBIT_STEP_VERDICT_WGSL);
        // Per coefficient: the verdict is the minimum over the columns, and verdict 1 (accepted
        // step small, full direction large) stops unconditionally, whatever shortened the step.
        expect(kernel.source).toContain('verdict = min(verdict, getProbitStepVerdict(');
        expect(kernel.source).toMatch(
          /if \(verdict != 0u\) \{\n \/\/[^\n]*\n if \(verdict == 1u\) \{ state\[stateOffset \+ S_DONE\] = 1\.0;/
        );
        expect(kernel.source).not.toContain('limitedByBound');
        expect(kernel.source).not.toContain('largestDirection');
      }
    } finally {
      operation.destroy();
      device.destroy();
    }
  }
});
