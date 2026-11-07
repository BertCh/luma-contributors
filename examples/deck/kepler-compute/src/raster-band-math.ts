// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createRasterBand,
  type NumericRasterBand,
  type NumericRasterTile
} from './raster-tile-ingest';

type BandMathNode =
  | {kind: 'number'; value: number}
  | {kind: 'band'; bandIndex: number}
  | {kind: 'unary'; operand: BandMathNode}
  | {kind: 'binary'; operator: '+' | '-' | '*' | '/'; left: BandMathNode; right: BandMathNode}
  | {kind: 'call'; name: BandMathFunctionName; args: BandMathNode[]};

type BandMathFunctionName = 'abs' | 'sqrt' | 'min' | 'max' | 'log';

const FUNCTION_ARITY: Record<BandMathFunctionName, number> = {
  abs: 1,
  sqrt: 1,
  min: 2,
  max: 2,
  log: 1
};

export type CompiledBandMath = Readonly<{
  expression: string;
  /** One-based band indexes the expression reads. */
  bandIndexes: readonly number[];
  evaluate: (bandValues: readonly number[]) => number;
}>;

/** Kepler `BandCombination` presets expressed as band math over b1, b2, b3 (red, green/second, blue/third). */
export const KEPLER_BAND_COMBINATION_EXPRESSIONS: Readonly<Record<string, string>> = {
  normalizedDifference: '(b1 - b2) / (b1 + b2)',
  enhancedVegetationIndex: '2.5 * (b1 - b2) / (b1 + 6 * b2 - 7.5 * b3 + 1)',
  soilAdjustedVegetationIndex: '(b1 - b2) / ((b1 + b2 + 0.5) * 1.5)',
  modifiedSoilAdjustedVegetationIndex:
    '(2 * b1 + 1 - sqrt((2 * b1 + 1) * (2 * b1 + 1) - 8 * (b1 - b2))) / 2'
};

function tokenize(expression: string): string[] {
  const pattern = /\s*(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+|[A-Za-z_]\w*|[-+*/(),])/y;
  const tokens: string[] = [];
  let position = 0;
  while (position < expression.length) {
    pattern.lastIndex = position;
    const match = pattern.exec(expression);
    if (!match) {
      // Only trailing whitespace may remain unmatched.
      if (expression.slice(position).trim() === '') break;
      throw new Error('Band math expression contains unsupported characters');
    }
    tokens.push(match[1]);
    position = pattern.lastIndex;
  }
  return tokens;
}

export function parseBandMath(expression: string): BandMathNode {
  const tokens = tokenize(expression);
  let position = 0;
  const peek = () => tokens[position];
  const take = () => tokens[position++];

  function parseSum(): BandMathNode {
    let left = parseProduct();
    while (peek() === '+' || peek() === '-')
      left = {kind: 'binary', operator: take() as '+' | '-', left, right: parseProduct()};
    return left;
  }
  function parseProduct(): BandMathNode {
    let left = parseUnary();
    while (peek() === '*' || peek() === '/')
      left = {kind: 'binary', operator: take() as '*' | '/', left, right: parseUnary()};
    return left;
  }
  function parseUnary(): BandMathNode {
    if (peek() === '-') {
      take();
      return {kind: 'unary', operand: parseUnary()};
    }
    return parseAtom();
  }
  function parseAtom(): BandMathNode {
    const token = take();
    // Grammar violations: unexpected end or token.
    if (token === undefined) throw new Error('Band math expression ended unexpectedly');
    if (token === '(') {
      const inner = parseSum();
      if (take() !== ')') throw new Error('Band math expression has an unclosed parenthesis');
      return inner;
    }
    if (/^[\d.]/.test(token)) return {kind: 'number', value: Number(token)};
    const bandMatch = /^b(\d+)$/.exec(token);
    if (bandMatch && peek() !== '(') {
      const bandIndex = Number(bandMatch[1]);
      if (bandIndex < 1) throw new Error('Band indexes are one-based');
      return {kind: 'band', bandIndex};
    }
    if (token in FUNCTION_ARITY && take() === '(') {
      const name = token as BandMathFunctionName;
      const args: BandMathNode[] = [parseSum()];
      while (peek() === ',') {
        take();
        args.push(parseSum());
      }
      if (take() !== ')' || args.length !== FUNCTION_ARITY[name])
        throw new Error(`Band math function ${name} has the wrong arguments`);
      return {kind: 'call', name, args};
    }
    throw new Error(`Band math token ${token} is not recognized`);
  }

  const root = parseSum();
  if (position !== tokens.length) throw new Error('Band math expression has trailing tokens');
  return root;
}

function collectBandIndexes(node: BandMathNode, indexes: Set<number>): void {
  if (node.kind === 'band') indexes.add(node.bandIndex);
  else if (node.kind === 'unary') collectBandIndexes(node.operand, indexes);
  else if (node.kind === 'binary') {
    collectBandIndexes(node.left, indexes);
    collectBandIndexes(node.right, indexes);
  } else if (node.kind === 'call')
    for (const argument of node.args) collectBandIndexes(argument, indexes);
}

function evaluateNode(node: BandMathNode, bandValues: ReadonlyMap<number, number>): number {
  switch (node.kind) {
    case 'number':
      return node.value;
    case 'band':
      return bandValues.get(node.bandIndex) as number;
    case 'unary':
      return -evaluateNode(node.operand, bandValues);
    case 'binary': {
      const left = evaluateNode(node.left, bandValues);
      const right = evaluateNode(node.right, bandValues);
      return node.operator === '+'
        ? left + right
        : node.operator === '-'
          ? left - right
          : node.operator === '*'
            ? left * right
            : left / right;
    }
    case 'call': {
      const args = node.args.map(argument => evaluateNode(argument, bandValues));
      if (node.name === 'abs') return Math.abs(args[0]);
      if (node.name === 'sqrt') return Math.sqrt(args[0]);
      if (node.name === 'log') return Math.log(args[0]);
      return node.name === 'min' ? Math.min(args[0], args[1]) : Math.max(args[0], args[1]);
    }
  }
}

/** Emits a WGSL f32 expression over `b1..bN` locals. Compile-only design; not executed on a GPU yet. */
export function emitBandMathWgsl(node: BandMathNode): string {
  switch (node.kind) {
    case 'number':
      return Number.isInteger(node.value) ? `${node.value}.0` : String(node.value);
    case 'band':
      return `b${node.bandIndex}`;
    case 'unary':
      return `(-${emitBandMathWgsl(node.operand)})`;
    case 'binary':
      return `(${emitBandMathWgsl(node.left)} ${node.operator} ${emitBandMathWgsl(node.right)})`;
    case 'call':
      return `${node.name === 'log' ? 'log' : node.name}(${node.args.map(emitBandMathWgsl).join(', ')})`;
  }
}

export function compileBandMath(expression: string): CompiledBandMath {
  const root = parseBandMath(expression);
  const indexSet = new Set<number>();
  collectBandIndexes(root, indexSet);
  const bandIndexes = [...indexSet].sort((first, second) => first - second);
  const bandValues = new Map<number, number>();
  return {
    expression,
    bandIndexes,
    evaluate: values => {
      for (const [position, bandIndex] of bandIndexes.entries())
        bandValues.set(bandIndex, values[position]);
      return evaluateNode(root, bandValues);
    }
  };
}

/**
 * Evaluates band math over one tile. Nodata propagates: a sample is nodata when any referenced
 * band is nodata there, or the result is not finite (division by zero, sqrt of a negative).
 * The output is a Float32 band whose nodata is NaN.
 */
export function evaluateBandMath(
  tile: NumericRasterTile,
  compiled: CompiledBandMath,
  name = compiled.expression
): NumericRasterBand {
  const sampleCount = tile.width * tile.height;
  const sourceBands = compiled.bandIndexes.map(bandIndex => {
    const band = tile.bands.find(candidate => candidate.index === bandIndex);
    // A missing band is a binding error, not a data condition.
    if (!band) throw new Error(`Band math references missing band b${bandIndex}`);
    return band;
  });
  const output = new Float32Array(sampleCount);
  const inputs = new Array<number>(sourceBands.length);
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
    let isValid = true;
    for (const [position, band] of sourceBands.entries()) {
      if (!band.validity[sampleIndex]) {
        isValid = false;
        break;
      }
      inputs[position] = band.values[sampleIndex];
    }
    const result = isValid ? compiled.evaluate(inputs) : NaN;
    output[sampleIndex] = Number.isFinite(result) ? result : NaN;
  }
  return createRasterBand({name, values: output, noDataValue: null, index: 0}, 0, sampleCount);
}
