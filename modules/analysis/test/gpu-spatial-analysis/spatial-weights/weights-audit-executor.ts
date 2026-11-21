// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ScalarData, TypedData, WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import type {RecordedGPUKernelSource} from '../../utils/gpu-contributor-test-utils';

/** Execute a noncollective emitted kernel on the CPU; this does not certify GPU compilation. */
export function executeWeightsAuditKernel(
  kernel: RecordedGPUKernelSource,
  buffers: Record<string, Float32Array | Uint32Array>,
  dispatchCount: number[] = [1, 1, 1]
): void {
  let source = ('struct WeightsAuditFrexp { fract: f32, exp: i32, }\n' + kernel.source)
    .replace(/^\s*_\s*=[^;]+;/gm, '')
    .replace(/@workgroup_size\((\d+)\)/g, '@workgroup_size($1, 1, 1)');
  // WgslExec cannot select a component directly on a returned vector. Name
  // that temporary without changing its arithmetic or the production shader.
  source = source.replace(
    'rounded = shiftLeftUint64(magnitude, u32(-shift)).y;',
    'let auditShifted = shiftLeftUint64(magnitude, u32(-shift)); rounded = auditShifted.y;'
  );
  const parameters = buffers['graphChunkGroupParameters'];
  if (parameters) {
    // The interpreter cannot index a uniform vec4 twice. Substitute only the
    // immutable uniform lookup; row arithmetic and control flow stay unchanged.
    source = source.replace(
      'return graphChunkGroupParameters.words[word / 4u][word % 4u];',
      `switch word { ${Array.from(parameters)
        .map((value, index) => `case ${index}u: { return ${value}u; }`)
        .join(' ')} default: { return 0u; } }`
    );
  }
  const bindings: Record<number, Float32Array | Uint32Array> = {};
  const reflection = new WgslReflect(source);
  for (const resource of [...reflection.storage, ...reflection.uniforms]) {
    const buffer = buffers[resource.name];
    if (!buffer) throw new Error(`Missing audit fixture binding ${resource.name}`);
    bindings[resource.binding] = buffer;
  }
  const executor = new WgslExec(new WgslParser().parse(source));
  // WgslExec leaves these WGSL builtins as TODOs. Supply only their scalar
  // definitions, not model arithmetic; independent fixture literals stay unchanged.
  executor.builtins.Frexp = (node, context) => {
    const value = executor.evalExpression(node.args![0], context) as ScalarData;
    const exponent =
      value.value === 0 || !Number.isFinite(value.value)
        ? 0
        : Math.floor(Math.log2(Math.abs(value.value))) + 1;
    const words = new ArrayBuffer(8);
    new Float32Array(words)[0] = value.value * 2 ** -exponent;
    new Int32Array(words)[1] = exponent;
    return new TypedData(words, executor.getTypeInfo('WeightsAuditFrexp')!);
  };
  executor.builtins.Ldexp = (node, context) => {
    const value = executor.evalExpression(node.args![0], context) as ScalarData;
    const exponent = executor.evalExpression(node.args![1], context) as ScalarData;
    return new ScalarData(
      value.value === 0 ? value.value : value.value * 2 ** exponent.value,
      executor.getTypeInfo('f32')!
    );
  };
  const evaluateBinary = executor._evalBinaryOp.bind(executor);
  executor._evalBinaryOp = (node, context) => {
    // WGSL short-circuits these operators. WgslExec eagerly evaluates the
    // right operand, including guarded position-1 or end-of-array accesses.
    if (node.operator === '&&' || node.operator === '||') {
      const left = executor.evalExpression(node.left, context);
      if (
        left instanceof ScalarData &&
        ((node.operator === '&&' && !left.value) || (node.operator === '||' && left.value))
      )
        return new ScalarData(left.value, left.typeInfo);
    }
    const result = evaluateBinary(node, context);
    // The interpreter uses JavaScript's signed >> for WGSL u32 operands.
    // Restore WGSL's logical shift (0xffffffffu >> 1u must be 0x7fffffff).
    if (node.operator === '>>' && result instanceof ScalarData) {
      const left = executor.evalExpression(node.left, context);
      const right = executor.evalExpression(node.right, context);
      if (left instanceof ScalarData && left.typeInfo.name === 'u32' && right instanceof ScalarData)
        result.value = left.value >>> right.value;
    }
    return result;
  };
  executor.dispatchWorkgroups('main', dispatchCount, {0: bindings}, {constants: kernel.constants});
}
