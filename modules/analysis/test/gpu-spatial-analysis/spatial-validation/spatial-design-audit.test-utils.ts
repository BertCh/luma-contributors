// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, vi} from 'vitest';
import {ScalarData, VectorData, WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';

/** Host interpretation of the generated kernel; device regressions remain separate. */
export function executeSpatialAuditKernel(
  kernel: ReturnType<typeof getGPUCommandNodeKernelSources>[number],
  values: Record<string, Float32Array | Uint32Array>,
  options: {
    invocationCount?: number;
    expandArrayIncrement?: boolean;
    flushSubnormalComparisons?: boolean;
    reciprocalDivision?: boolean;
    flushSubnormalProducts?: boolean;
    forbidCalls?: readonly string[];
  } = {}
) {
  // wgsl_reflect does not implement ++ on an array element. Only expand this exact
  // expression for the Lloyd probe; the device case compiles the unchanged shader.
  const source = options.expandArrayIncrement
    ? kernel.source.replaceAll('counts[selected]++;', 'counts[selected] += 1u;')
    : kernel.source;
  const errors = vi.spyOn(console, 'error').mockImplementation(message => {
    if (!String(message).startsWith('Variable _ not found.')) throw new Error(String(message));
  });
  try {
    const bindings = Object.fromEntries(
      new WgslReflect(source).storage.map(binding => {
        expect(values[binding.name], binding.name).toBeDefined();
        return [binding.binding, values[binding.name]];
      })
    );
    const execution = new WgslExec(new WgslParser().parse(source));
    const evaluateCall = execution._evalCall.bind(execution);
    execution._evalCall = (expression, context) => {
      if (options.forbidCalls?.includes(expression.name))
        throw new Error(`forbidden call: ${expression.name}`);
      if (options.flushSubnormalProducts && expression.name === 'dot') {
        const first = execution.evalExpression(expression.args![0], context),
          second = execution.evalExpression(expression.args![1], context);
        if (first instanceof VectorData && second instanceof VectorData) {
          let total = 0;
          for (let component = 0; component < first.data.length; component++) {
            const product = Math.fround(first.data[component] * second.data[component]);
            total = Math.fround(total + (Math.abs(product) < 2 ** -126 ? 0 : product));
          }
          return new ScalarData(total, execution.getTypeInfo('f32')!);
        }
      }
      return evaluateCall(expression, context);
    };
    const evaluateBinary = execution._evalBinaryOp.bind(execution);
    execution._evalBinaryOp = (expression, context) => {
      // Model a permitted float32 reciprocal implementation of division. This
      // exposes centroid rounding that the interpreter's exact host division hides.
      if (options.reciprocalDivision && expression.operator === '/') {
        const left = execution.evalExpression(expression.left, context),
          right = execution.evalExpression(expression.right, context);
        if (
          left instanceof ScalarData &&
          right instanceof ScalarData &&
          left.typeInfo.name === 'f32' &&
          right.typeInfo.name === 'f32'
        ) {
          return new ScalarData(
            Math.fround(left.value * Math.fround(1 / right.value)),
            left.typeInfo
          );
        }
      }
      if (
        options.flushSubnormalComparisons &&
        ['>', '<', '>=', '<='].includes(expression.operator)
      ) {
        const left = execution.evalExpression(expression.left, context),
          right = execution.evalExpression(expression.right, context);
        if (
          left instanceof ScalarData &&
          right instanceof ScalarData &&
          left.typeInfo.name === 'f32' &&
          right.typeInfo.name === 'f32'
        ) {
          const first = Math.abs(left.value) < 2 ** -126 ? 0 : left.value,
            second = Math.abs(right.value) < 2 ** -126 ? 0 : right.value;
          const result =
            expression.operator === '>'
              ? first > second
              : expression.operator === '<'
                ? first < second
                : expression.operator === '>='
                  ? first >= second
                  : first <= second;
          return new ScalarData(Number(result), execution.getTypeInfo('bool')!);
        }
      }
      return evaluateBinary(expression, context);
    };
    execution.dispatchWorkgroups(
      kernel.entryPoint,
      [1, 1, 1],
      {0: bindings},
      {
        constants: {
          ...kernel.constants,
          ...(options.invocationCount === undefined
            ? {}
            : {INVOCATION_COUNT: options.invocationCount})
        }
      }
    );
  } finally {
    errors.mockRestore();
  }
}

// Exact Python Philox4x32-10 plus SciPy cdist oracle: spatial-design-audit.oracle.py.
export const BLOCK_AUDIT_POSITIONS = [0, 0, 1, 0, 1, 0, 2, 0, 4, 0, 6, 0, 10, 0, 12, 0, 99, 0];
export const BLOCK_AUDIT_BLOCKS = [30, 10, 30, 50, 70, 10, 90, 110, 999];
export const BLOCK_AUDIT_MASK = [1, 1, 1, 1, 1, 1, 1, 1, 0];
export const BLOCK_AUDIT_EXPECTED = {
  folds: [3, 1, 3, 0, 1, 1, 2, 0, 0xffffffff],
  training: [
    1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 0
  ],
  testing: [
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0,
    0, 0, 0, 0
  ],
  counts: [4, 2, 2, 2, 3, 3, 7, 1, 0, 4, 2, 2],
  status: [0, 0, 0, 0]
};
