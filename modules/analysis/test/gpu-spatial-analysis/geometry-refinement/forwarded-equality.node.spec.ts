// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {VectorData, WgslExec} from 'wgsl_reflect';
import {getCanonicalEqualityMetricWGSL} from '../../../src/gpu-spatial-analysis/geometry-refinement/canonical-equality-metric';
import {createWaveHarness} from '../geometry-validity/w5-test-harness';
import type {EqualityView} from './canonical-equality-fixture';
import {createSubnormalEqualityFixture} from './forwarded-equality-fixture';

it('forward metric N1 reaches the exact Euclidean certificate before float subtraction', () => {
  // N=2^-126, left=1.5N, right=.75N: exact stored distance .75N <= N.
  // Flushing right before a float subtraction instead yields 1.5N > N. The
  // canonical wrapper must let the shared bit-based certificate make this decision.
  const source = getCanonicalEqualityMetricWGSL(2 ** -126, 'euclidean');
  const wrapper = source.slice(source.indexOf('fn withinTolerance('));
  expect(wrapper).toContain('return geometryWithinTolerance(left,right,tolerance);');
  expect(wrapper).not.toContain('left-right');
  expect(wrapper).not.toContain('return false');
  const perCoordinate = getCanonicalEqualityMetricWGSL(2 ** -126, 'per-coordinate');
  expect(perCoordinate.slice(perCoordinate.indexOf('fn withinTolerance('))).toContain('left-right');
});

it.each([
  'tree',
  'polygon'
] as const)('forward metric N1 canonical %s retains a within-tolerance subnormal operand under FTZ', async kind => {
  const harness = createWaveHarness();
  const view: EqualityView = (format, values) =>
    harness.allocate(
      'subnormal',
      format,
      values.length / (format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1),
      values
    );
  const fixture = createSubnormalEqualityFixture(kind, view);
  const original = WgslExec.prototype._evalBinaryOp;
  const minimumNormal = 2 ** -126;
  const flush = (value: number) => (Math.abs(value) < minimumNormal ? 0 : value);
  // Emulate a device that flushes float vector subtraction operands and results.
  // Bitcasts and integer certificate arithmetic retain the original stored bits.
  const arithmetic = vi.spyOn(WgslExec.prototype, '_evalBinaryOp').mockImplementation(function (
    this: WgslExec,
    expression,
    context
  ) {
    const result = original.call(this, expression, context);
    if (
      expression.operator === '-' &&
      result instanceof VectorData &&
      result.data instanceof Float32Array
    ) {
      const left = this.evalExpression(expression.left, context);
      const right = this.evalExpression(expression.right, context);
      if (left instanceof VectorData && right instanceof VectorData) {
        for (let component = 0; component < result.data.length; component++) {
          result.data[component] = flush(
            Math.fround(flush(left.data[component]) - flush(right.data[component]))
          );
        }
      }
    }
    return result;
  });
  try {
    await harness.run(fixture.contributor.getCommandNodes(harness.graph));
    // Exact binary closed form: |1.5N - .75N| = .75N <= N.
    expect(await harness.read(fixture.output)).toEqual([1]);
  } finally {
    arithmetic.mockRestore();
    harness.destroy();
  }
});
