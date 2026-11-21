// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {ScalarData, WgslExec, WgslParser} from 'wgsl_reflect';

it('keeps bucket three when Metal divides using a flushed reciprocal', () => {
  const source = readFileSync(
    new URL(
      '../../../src/dataframe-analysis/temporal-reduction/gpu-temporal-reduction.ts',
      import.meta.url
    ),
    'utf8'
  );
  const body = source.match(/const FLOAT_BUCKET_BODY = \/\* wgsl \*\/ `([\s\S]*?)`;/)?.[1];
  if (!body) throw new Error('FLOAT_BUCKET_BODY not found');
  const shader = `@group(0) @binding(0) var<storage, read> timestamps: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;
const timestampsOffset: u32 = 0u; const paramsOffset: u32 = 0u; const BUCKET_COUNT: u32 = 4u;
fn isNanBits(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
fn getOrderedKey(value: f32) -> u32 { return bitcast<u32>(value); }
@compute @workgroup_size(1) fn main() { let row = 0u; var bucket = 99u; var timeKey = 0u;
${body}
result[0] = bucket; }`;
  const execution = new WgslExec(new WgslParser().parse(shader));
  const evaluate = execution._evalBinaryOp.bind(execution);
  execution._evalBinaryOp = (expression, context) => {
    if (expression.operator === '/') {
      const left = execution.evalExpression(expression.left, context);
      const right = execution.evalExpression(expression.right, context);
      if (
        left instanceof ScalarData &&
        right instanceof ScalarData &&
        left.typeInfo.name === 'f32'
      ) {
        const reciprocal = Math.fround(1 / right.value);
        const flushed = Math.abs(reciprocal) < 2 ** -126 ? 0 : reciprocal;
        return new ScalarData(Math.fround(left.value * flushed), left.typeInfo);
      }
    }
    return evaluate(expression, context);
  };
  const result = new Uint32Array([99]);
  execution.dispatchWorkgroups('main', 1, {
    0: {0: new Float32Array([3e38]), 1: new Float32Array([0, 1e38]), 2: result}
  });
  // Binary32 inputs: 3.0000000054977558e38 / 9.999999680285692e37 = 3.0000001014... .
  expect(Array.from(result)).toEqual([3]);
});
