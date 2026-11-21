// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CreateExpr, ScalarData, VectorData, WgslExec} from 'wgsl_reflect';
import {afterEach, beforeEach, vi} from 'vitest';

/** wgsl_reflect 1.2.3 backs generic integer vectors with float32, losing low product bits. */
export function usePrecisionExecutorHooks(): void {
  let restore: (() => void) | undefined;
  beforeEach(() => {
    const original = WgslExec.prototype._callConstructorVec;
    const spy = vi.spyOn(WgslExec.prototype, '_callConstructorVec').mockImplementation(function (
      this: WgslExec,
      node,
      context
    ) {
      const type = node.type ? this.getTypeInfo(node.type) : null;
      const format = type && 'format' in type ? type.format : undefined;
      if (
        node instanceof CreateExpr &&
        type &&
        format &&
        typeof format === 'object' &&
        'name' in format &&
        (format.name === 'u32' || format.name === 'i32')
      ) {
        const values: number[] = [];
        for (const argument of node.args ?? []) {
          const value = this.evalExpression(argument, context);
          if (value instanceof ScalarData) values.push(value.value);
          else if (value instanceof VectorData) values.push(...value.data);
        }
        const count = Number(type.name.match(/\d/)?.[0]);
        if (values.length === 1) while (values.length < count) values.push(values[0]);
        if (!values.length) values.push(...Array(count).fill(0));
        const data = format.name === 'u32' ? Uint32Array.from(values) : Int32Array.from(values);
        return new VectorData(data, type).getSubData(this, node.postfix, context);
      }
      return original.call(this, node, context);
    });
    restore = () => spy.mockRestore();
  });
  afterEach(() => {
    restore?.();
  });
}
