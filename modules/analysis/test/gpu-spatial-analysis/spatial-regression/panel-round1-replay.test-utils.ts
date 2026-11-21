// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ScalarData, VectorData, WgslExec, TemplateInfo} from 'wgsl_reflect';
import {executeCommandNodesOnCPU} from '../segment-intersection/wgsl-graph-executor';

/**
 * wgsl_reflect 1.2.3 stores vec2<u32> constructor results in Float32Array when reflection
 * represents the type as TemplateInfo('vec2', u32). 0xc0000032 becomes 0xc0000000. Preserve
 * the constructor's declared integer element type. No shader arithmetic or source is replaced.
 */
export function executePanelRound1NodesOnCPU(
  ...arguments_: Parameters<typeof executeCommandNodesOnCPU>
): void {
  const original = WgslExec.prototype._callConstructorVec;
  WgslExec.prototype._callConstructorVec = function (node, context) {
    if (!node.type) return original.call(this, node, context);
    const type = this.getTypeInfo(node.type);
    if (!type) return original.call(this, node, context);
    const format = type instanceof TemplateInfo ? type.format?.name : undefined;
    if (
      'args' in node &&
      node.args &&
      (node.type.getTypeName().endsWith('u') || format === 'u32')
    ) {
      const words = node.args.flatMap(argument => {
        const value = this.evalExpression(argument, context);
        return value instanceof ScalarData
          ? [value.value]
          : value instanceof VectorData
            ? Array.from(value.data)
            : [];
      });
      const length = Number(/vec(\d)/.exec(node.type.getTypeName())![1]);
      if (words.length === 0) while (words.length < length) words.push(0);
      if (words.length === 1) while (words.length < length) words.push(words[0]);
      return new VectorData(Uint32Array.from(words), type).getSubData(this, node.postfix, context);
    }
    return original.call(this, node, context);
  };
  try {
    executeCommandNodesOnCPU(...arguments_);
  } finally {
    WgslExec.prototype._callConstructorVec = original;
  }
}
