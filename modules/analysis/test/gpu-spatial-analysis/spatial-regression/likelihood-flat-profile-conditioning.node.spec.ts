// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import ts from 'typescript';
import {expect, it} from 'vitest';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {getChiSquareSurvival} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import fixtures from './likelihood-flat-profile-fixtures.json';

type NumericValue = number | readonly number[];

// Host evaluation of emitted scalar arithmetic, with a binary32 rounding after each operation.
// It does not emulate Metal fusion; the unchanged headless oracle is the actual GPU regression.
function evaluate(expression: ts.Expression, values: Record<string, NumericValue>): number {
  if (ts.isParenthesizedExpression(expression)) return evaluate(expression.expression, values);
  if (ts.isNumericLiteral(expression)) return Math.fround(Number(expression.text));
  if (ts.isIdentifier(expression)) return values[expression.text] as number;
  if (ts.isElementAccessExpression(expression) && ts.isIdentifier(expression.expression)) {
    return (values[expression.expression.text] as readonly number[])[
      evaluate(expression.argumentExpression, values)
    ];
  }
  if (ts.isBinaryExpression(expression)) {
    const first = evaluate(expression.left, values),
      second = evaluate(expression.right, values);
    switch (expression.operatorToken.kind) {
      case ts.SyntaxKind.PlusToken:
        return Math.fround(first + second);
      case ts.SyntaxKind.MinusToken:
        return Math.fround(first - second);
      case ts.SyntaxKind.AsteriskToken:
        return Math.fround(first * second);
      case ts.SyntaxKind.SlashToken:
        return Math.fround(first / second);
      case ts.SyntaxKind.ExclamationEqualsToken:
        return Number(first !== second);
    }
  }
  if (ts.isCallExpression(expression) && expression.expression.getText() === 'select') {
    return evaluate(
      expression.arguments[Number(Boolean(evaluate(expression.arguments[2], values)))],
      values
    );
  }
  throw new Error(`Unsupported expression: ${expression.getText()}`);
}

function evaluateText(source: string, values: Record<string, NumericValue>): number {
  const parsed = ts.createSourceFile(
    'arithmetic.ts',
    source.replace(/(\d+)u\b/g, '$1'),
    ts.ScriptTarget.Latest,
    true
  );
  const statement = parsed.statements[0];
  if (!ts.isExpressionStatement(statement)) throw new Error('Expected scalar expression');
  return evaluate(statement.expression, values);
}

it('the emitted scalar-profile solve preserves the unfiltered coefficient SE and p-value for negative weights', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `value-${serial++}`, format, length);
    const contributor = new GPUSpatialErrorMaximumLikelihood({
      id: 'scalar-conditioning',
      predictorCount: 1,
      maximumPower: 3,
      logDeterminant: 'exact',
      weights: {
        offsets: view('uint32', 9),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      },
      eigenvalues: view('float32', 8),
      predictors: view('float32', 8),
      response: view('float32', 8),
      output: {
        table: view('float32', 12),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      }
    });
    const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(
      kernel => kernel.id === 'scalar-conditioning-filtered-solve'
    )!.source;
    const float = Math.fround;
    // x=0..7, y=x^2: centered [1,x,e] has Gram diagonal [8,42,168], all cross terms zero.
    // For W=-3I: G1=-6G0, G2=9G0. The unfiltered intercept variance is 21*(1/8+3.5^2/42)=8.75.
    const gram = [8, 0, 42, 0, 0, 168, -48, 0, -252, 0, 0, -1008, 72, 0, 378, 0, 0, 1512];
    const parameterSymbol = [
      ...source.matchAll(/let diagonal = getGramValue\(column, column, (\w+)\);/g)
    ]
      .map(match => match[1])
      .find(value => value !== '0')!;
    const lambda = float(-0.33);
    const parameter =
      parameterSymbol === 'lambda'
        ? lambda
        : evaluateText(source.match(/let solveParameter = ([^;]+);/)![1], {
            lambda,
            scalarWeight: -3
          });
    const gramExpression = source.match(/fn getGramValue[\s\S]*?return ([\s\S]*?);/)![1];
    const diagonal = [0, 2, 5].map(entry =>
      evaluateText(gramExpression, {gram, gramOffset: 0, entry, PACK: 6, lambda: parameter})
    );
    const regressionSigma2 = float(diagonal[2] / 8);
    const filteringScale = float(1 - float(lambda * -3));
    const sigma2 =
      parameterSymbol === 'lambda'
        ? regressionSigma2
        : float(float(regressionSigma2 * filteringScale) * filteringScale);
    const covarianceExpression = source.match(/covariance\[row \* P \+ column\] = ([^;]+);/)![1];
    const covariance = [0, 1].map(column =>
      evaluateText(covarianceExpression, {
        sigma2,
        regressionSigma2,
        row: column,
        column,
        unitVector: [Number(column === 0), Number(column === 1)],
        scale: diagonal.slice(0, 2).map(value => float(Math.sqrt(value)))
      })
    );
    const standardError = float(Math.sqrt(float(covariance[0] + float(12.25 * covariance[1]))));
    const statistic = float(-7 / standardError);
    const probability = getChiSquareSurvival(statistic ** 2, 1);
    const expected = fixtures.find(value => value.maximumPower === 3 && value.scalarWeight === -3)!
      .expected.table;
    expect(Number.isFinite(standardError) && Number.isFinite(probability)).toBe(true);
    expect(Math.abs(standardError / expected[1]! - 1)).toBeLessThan(3e-6);
    expect(Math.abs(probability / expected[3]! - 1)).toBeLessThan(1e-5);
  } finally {
    device.destroy();
  }
});
