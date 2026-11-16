// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WgslExec, ScalarData, TypedData, StructInfo, MemberInfo} from 'wgsl_reflect';
import {
  executeCommandNodesOnCPU,
  type createCPUGraphMemory
} from '../segment-intersection/wgsl-graph-executor';

/**
 * wgsl_reflect computes arrayLength from TypedArray.buffer.byteLength, ignoring the view length.
 * The shared executor intentionally pads each fake backing by four words. Give each binding an
 * exact-size backing during interpretation, then copy writes back. This preserves the real bound
 * range and keeps runtime-length shaders testable without changing production WGSL for the tool.
 */
export function executeGeometryNodesOnCPU(
  nodes: readonly unknown[],
  memory: ReturnType<typeof createCPUGraphMemory>
): void {
  const descriptor = Object.getOwnPropertyDescriptor(WgslExec.prototype, 'dispatchWorkgroups')!;
  const original = WgslExec.prototype.dispatchWorkgroups;
  Object.defineProperty(WgslExec.prototype, 'dispatchWorkgroups', {
    ...descriptor,
    value: function (this: WgslExec, ...args: Parameters<typeof original>) {
      type Words = Float32Array | Uint32Array | Int32Array;
      const groups = args[2] as unknown as Record<number, Record<number, Words>>;
      const copies: [Words, Words][] = [];
      const exactGroups = Object.fromEntries(
        Object.entries(groups).map(([group, fields]) => [
          group,
          Object.fromEntries(
            Object.entries(fields).map(([binding, words]) => {
              const exact = words.slice();
              copies.push([words, exact]);
              return [binding, exact];
            })
          )
        ])
      );
      // The shared integer addition-error shader uses scalar ldexp, unimplemented by 1.2.3.
      const originalLdexp = this.builtins.Ldexp;
      const originalFrexp = this.builtins.Frexp;
      // frexp is likewise missing. Describe its scalar result through the interpreter's public
      // data API so the production shader still executes unchanged, including exponent arithmetic.
      const fraction = new MemberInfo('fract', this.getTypeInfo('f32')!, null);
      const exponent = new MemberInfo('exp', this.getTypeInfo('i32')!, null);
      fraction.offset = 0;
      fraction.size = 4;
      exponent.offset = 4;
      exponent.size = 4;
      const resultType = new StructInfo('GeometryFrexpResult', null);
      resultType.members = [fraction, exponent];
      resultType.align = 4;
      resultType.size = 8;
      this.builtins.Frexp = (node, context) => {
        const value = this.evalExpression(node.args![0], context) as ScalarData;
        const power = value.value === 0 ? 0 : Math.floor(Math.log2(Math.abs(value.value))) + 1;
        const result = new ArrayBuffer(8);
        new Float32Array(result)[0] = Math.fround(value.value / 2 ** power);
        new Int32Array(result)[1] = power;
        return new TypedData(result, resultType).getSubData(
          this,
          'postfix' in node ? node.postfix : null,
          context
        );
      };
      this.builtins.Ldexp = (node, context) => {
        const value = this.evalExpression(node.args![0], context) as ScalarData;
        const exponent = this.evalExpression(node.args![1], context) as ScalarData;
        return new ScalarData(Math.fround(value.value * 2 ** exponent.value), value.typeInfo);
      };
      const exactArgs = [...args] as Parameters<typeof original>;
      exactArgs[2] = exactGroups;
      try {
        return original.apply(this, exactArgs);
      } finally {
        this.builtins.Ldexp = originalLdexp;
        this.builtins.Frexp = originalFrexp;
        for (const [words, exact] of copies) words.set(exact);
      }
    }
  });
  try {
    executeCommandNodesOnCPU(nodes, memory);
  } finally {
    Object.defineProperty(WgslExec.prototype, 'dispatchWorkgroups', descriptor);
  }
}
