// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {ScalarData, WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {GPUColocationQuotient} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-colocation-quotient';
import {GPULocalCategoryDependence} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-local-category-dependence';
import {executeSpatialAuditKernel} from '../spatial-validation/spatial-design-audit.test-utils';
import {FIXED_POINT_WORD_COUNT} from '../../../src/analysis-core/kernels/fixed-point-accumulator';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  CATEGORICAL_AUDIT_CATEGORIES,
  CATEGORICAL_AUDIT_OFFSETS,
  CATEGORICAL_AUDIT_NEIGHBORS,
  CATEGORICAL_AUDIT_WEIGHTS,
  CATEGORICAL_AUDIT_ROW_COUNTS,
  CATEGORICAL_AUDIT_LOCAL
} from './categorical-association-audit.test-utils';

it('R7 local binomial tail stays below 0.05 for the independent 2190-neighbor oracle', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `binomial-audit-${nextView++}`, format, length);
  try {
    const contributor = new GPULocalCategoryDependence({
      weights: {
        offsets: view('uint32', 2271),
        neighbors: view('uint32', 2190),
        weights: view('float32', 2190)
      },
      categories: view('uint32', 2270),
      categoryCount: 2,
      output: {statistics: view('float32', 2270 * 6)}
    });
    const [kernel] = getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(graph).filter(node => node.id.endsWith('-statistics'))
    );
    const statistics = new Float32Array(2270 * 6);
    executeSpatialAuditKernel(
      kernel,
      {
        offsets: Uint32Array.from({length: 2271}, (_, index) => (index === 0 ? 0 : 2190)),
        neighbors: Uint32Array.from({length: 2190}, (_, index) => index + 1),
        weights: new Float32Array(2190).fill(1),
        categories: Uint32Array.from({length: 2270}, (_, row) => Number(row >= 1135)),
        rowCounts: Uint32Array.of(1135, 1135),
        statistics
      },
      {invocationCount: 1}
    );
    // scipy.stats.binom.sf(1133,2190,.5), generated in categorical-association-audit.oracle.py.
    expect(Number.isFinite(statistics[4])).toBe(true);
    expect(statistics[4]).toBeLessThan(0.05);
    expect(Math.abs(statistics[4] - 0.049933762130512475)).toBeLessThan(2e-7);
  } finally {
    device.destroy();
  }
});

it('review tiny positive colocation weights retain a unit neighbor share under reciprocal division', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `tiny-${serial++}`, format, length);
  try {
    const nodes = new GPUColocationQuotient({
      weights: {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 1),
        weights: view('float32', 1)
      },
      categories: view('uint32', 2),
      categoryCount: 2,
      output: {globalQuotients: view('float32', 4), localQuotients: view('float32', 4)}
    }).getCommandNodes(graph);
    const [kernel] = getGPUCommandNodeKernelSources(
      nodes.filter(node => node.id === 'colocation-quotient-shares')
    );
    for (const weight of [2 ** -149, 2 ** -130, 2 ** -126, 2 ** -100]) {
      const localValues = new Float32Array(4);
      executeSpatialAuditKernel(
        kernel,
        {
          offsets: Uint32Array.of(0, 1, 1),
          neighbors: Uint32Array.of(1),
          linkWeights: Float32Array.of(weight),
          categories: Uint32Array.of(0, 1),
          rowCounts: new Uint32Array(2),
          linkedCounts: new Uint32Array(2),
          shareWords: new Uint32Array(4 * FIXED_POINT_WORD_COUNT),
          localValues
        },
        {invocationCount: 1, reciprocalDivision: true, flushSubnormalComparisons: true}
      );
      // One eligible link: its neighbor share is w/w=1 for every positive w.
      expect(Number.isFinite(localValues[1]), `weight ${weight}`).toBe(true);
      expect(Array.from(localValues.slice(0, 2))).toEqual([0, 1]);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('audit-validation co-location preserves huge-weight shares with flushed reciprocals=%s', flushReciprocals => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `colocation-audit-${nextView++}`, format, length);
  const contributor = new GPUColocationQuotient({
    weights: {
      offsets: view('uint32', 9),
      neighbors: view('uint32', 14),
      weights: view('float32', 14)
    },
    categories: view('uint32', 8),
    categoryCount: 4,
    output: {
      globalQuotients: view('float32', 16),
      localQuotients: view('float32', 32),
      categoryRowCounts: view('uint32', 4)
    }
  });
  const kernels = getGPUCommandNodeKernelSources(
    contributor
      .getCommandNodes(graph)
      .filter(
        node => node.id === 'colocation-quotient-shares' || node.id === 'colocation-quotient-local'
      )
  );
  expect(kernels).toHaveLength(2);
  const errors = vi.spyOn(console, 'error').mockImplementation(message => {
    if (!String(message).startsWith('Variable _ not found.')) throw new Error(String(message));
  });
  try {
    for (const scale of [1, 2 ** 64, 2 ** 126, 1.25 * 2 ** 126]) {
      const values: Record<string, Float32Array | Uint32Array> = {
        offsets: Uint32Array.from(CATEGORICAL_AUDIT_OFFSETS),
        neighbors: Uint32Array.from(CATEGORICAL_AUDIT_NEIGHBORS),
        linkWeights: Float32Array.from(CATEGORICAL_AUDIT_WEIGHTS, weight => weight * scale),
        categories: Uint32Array.from(CATEGORICAL_AUDIT_CATEGORIES),
        rowCounts: new Uint32Array(4),
        linkedCounts: new Uint32Array(4),
        shareWords: new Uint32Array(16 * FIXED_POINT_WORD_COUNT),
        localValues: new Float32Array(32)
      };
      for (const [index, kernel] of kernels.entries()) {
        // Isolate row zero. wgsl_reflect stops the rest of a workgroup at another
        // invocation's early return, so the full eight-row graph is checked on-device.
        if (index === 1) values['rowCounts'].set(CATEGORICAL_AUDIT_ROW_COUNTS);
        const bindings = Object.fromEntries(
          new WgslReflect(kernel.source).storage.map(binding => [
            binding.binding,
            values[binding.name]
          ])
        );
        const execution = new WgslExec(new WgslParser().parse(kernel.source));
        const evaluateBinary = execution._evalBinaryOp.bind(execution);
        execution._evalBinaryOp = (expression, context) => {
          if (flushReciprocals && expression.operator === '/') {
            const left = execution.evalExpression(expression.left, context);
            const right = execution.evalExpression(expression.right, context);
            if (
              left instanceof ScalarData &&
              right instanceof ScalarData &&
              left.typeInfo.name === 'f32' &&
              right.typeInfo.name === 'f32'
            ) {
              // Metal can lower division to a reciprocal that flushes below the normal range.
              const reciprocal = Math.fround(1 / right.value);
              if (Math.abs(reciprocal) < 2 ** -126)
                return new ScalarData(Math.fround(left.value * 0), left.typeInfo);
            }
          }
          return evaluateBinary(expression, context);
        };
        execution.dispatchWorkgroups(
          kernel.entryPoint,
          [1, 1, 1],
          {0: bindings},
          {constants: {...kernel.constants, INVOCATION_COUNT: index === 0 ? 1 : 4}}
        );
      }
      expect(Array.from(values['rowCounts'])).toEqual(CATEGORICAL_AUDIT_ROW_COUNTS);
      expect(Array.from(values['localValues'].slice(0, 4))).toEqual(
        CATEGORICAL_AUDIT_LOCAL.slice(0, 4)
      );
    }
  } finally {
    errors.mockRestore();
    device.destroy();
  }
});
