// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import ts from 'typescript';
import {expect, it} from 'vitest';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

// Evaluate only the emitted arithmetic expression, rounding every operation to binary32.
// This executes no GPU code; the matching headless case checks actual shader execution.
function evaluateFloat32(
  expression: ts.Expression,
  values: Record<string, number | number[]>
): number {
  if (ts.isParenthesizedExpression(expression))
    return evaluateFloat32(expression.expression, values);
  if (ts.isIdentifier(expression)) return Math.fround(values[expression.text] as number);
  if (ts.isNumericLiteral(expression)) return Number(expression.text);
  if (ts.isElementAccessExpression(expression) && ts.isIdentifier(expression.expression))
    return Math.fround(
      (values[expression.expression.text] as number[])[
        evaluateFloat32(expression.argumentExpression, values)
      ]
    );
  if (ts.isBinaryExpression(expression)) {
    const first = evaluateFloat32(expression.left, values);
    const second = evaluateFloat32(expression.right, values);
    if (expression.operatorToken.kind === ts.SyntaxKind.SlashToken)
      return Math.fround(first / second);
    if (expression.operatorToken.kind === ts.SyntaxKind.AsteriskToken)
      return Math.fround(first * second);
    if (expression.operatorToken.kind === ts.SyntaxKind.PlusToken)
      return Math.fround(first + second);
    if (expression.operatorToken.kind === ts.SyntaxKind.MinusToken)
      return Math.fround(first - second);
  }
  throw new Error(`Unsupported arithmetic: ${expression.getText()}`);
}

it('the emitted error-model pseudo-R squared stays 1/2 when products overflow or underflow', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `value-${serial++}`, format, length);
    const contributor = new GPUSpatialErrorMaximumLikelihood({
      id: 'error-scale',
      weights: {
        offsets: view('uint32', 9),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      },
      predictors: view('float32', 8),
      response: view('float32', 8),
      predictorCount: 1,
      maximumPower: 2,
      output: {
        table: view('float32', 12),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      }
    });
    const finish = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(
      kernel => kernel.id === 'error-scale-finish'
    )!;
    const expression = finish.source.match(/summaryOut\[summaryOutOffset \+ 6u\] = ([^;]+);/)![1];
    const parsed = ts.createSourceFile('expression.ts', expression, ts.ScriptTarget.Latest, true);
    const statement = parsed.statements[0];
    if (!ts.isExpressionStatement(statement))
      throw new Error('Expected the pseudo-R squared expression');
    for (const scale of [2 ** -40, 1, 2 ** 34]) {
      // Eight orthogonal +/-1 vectors v,w: y=s*v, prediction=s*(v+w)/2.
      // y'y=8s^2, prediction'prediction=y'prediction=4s^2; corr(y,prediction)^2=1/2.
      const actual = evaluateFloat32(statement.expression, {
        responseSquares: 8 * scale ** 2,
        predictionCovariance: 4 * scale ** 2,
        predictionVariance: 4 * scale ** 2
      });
      expect(Number.isFinite(actual), `scale=${scale}: ${actual}`).toBe(true);
      expect(actual).toBe(0.5);
    }
    // Shifted one-hot x: rounded xbar=1, fitted deviations are [0,0,1,0,0,0,0,0].
    // y=[0,0,1,-1,1,-1,1,-1]. Exact centered corr^2=1/(6*(1-1/8))=4/21.
    const moments: Record<string, number | number[]> = {
      residualSums: [6, 1, 1, 6, 1, 0],
      residualSumsOffset: 0,
      n: 8
    };
    for (const name of [
      'predictionTotal',
      'responseTotal',
      'predictionVariance',
      'predictionCovariance',
      'responseSquares'
    ]) {
      const initializer = finish.source.match(new RegExp(`let ${name} = ([^;]+);`))?.[1];
      if (!initializer) continue;
      const parsedMoment = ts.createSourceFile(
        'moment.ts',
        initializer.replace(/(\d+)u\b/g, '$1'),
        ts.ScriptTarget.Latest,
        true
      );
      const momentStatement = parsedMoment.statements[0];
      if (!ts.isExpressionStatement(momentStatement)) throw new Error('Expected moment expression');
      moments[name] = evaluateFloat32(momentStatement.expression, moments);
    }
    const centered = evaluateFloat32(statement.expression, moments);
    expect(Number.isFinite(centered)).toBe(true);
    expect(Math.abs(centered - 4 / 21), 'review 2 R3: recenter rounded deviations').toBeLessThan(
      2e-8
    );
  } finally {
    device.destroy();
  }
});
