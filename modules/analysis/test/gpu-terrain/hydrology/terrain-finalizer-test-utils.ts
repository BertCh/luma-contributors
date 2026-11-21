// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ShaderFactory} from '@luma.gl/core';
import type {GPUCommandNode} from '@luma.gl/graph/gpu-core';

/** Inject exactly one permitted weak-exchange failure per invocation before the real atomic. */
export function injectSpuriousFailure<Parameters>(
  node: GPUCommandNode<Parameters>,
  onInjectedModule?: (source: string) => void
): GPUCommandNode<Parameters> {
  if (node.type !== 'compute') return node;
  return {
    ...node,
    compile(context) {
      const factory = ShaderFactory.getDefaultShaderFactory(context.device);
      const createShader = factory.createShader;
      factory.createShader = props => {
        const pattern =
          /atomicCompareExchangeWeak\(&(acc|order)\[\1Offset \+ cell\], SENTINEL, (bitcast<u32>\(sum\)|result)\)/;
        const match = props.source.match(pattern);
        if (!match) return createShader.call(factory, props);
        const storage = match[1];
        const source =
          props.source.replace(pattern, 'testExchange(cell, ' + match[2] + ')') +
          `
struct TestExchange { old_value: u32, exchanged: bool }
var<private> testFirstExchange: bool = true;
fn testExchange(cell: u32, value: u32) -> TestExchange {
  if (testFirstExchange) {
    testFirstExchange = false;
    return TestExchange(SENTINEL, false);
  }
  let actual = atomicCompareExchangeWeak(&${storage}[${storage}Offset + cell], SENTINEL, value);
  return TestExchange(actual.old_value, actual.exchanged);
}`;
        // Transform the source before ShaderFactory hashes it, so both cache orders are safe.
        const shader = createShader.call(factory, {...props, source});
        if (shader.source !== source)
          throw new Error('Spurious-exchange probe used an uninjected shader');
        onInjectedModule?.(shader.source);
        return shader;
      };
      try {
        return node.compile(context);
      } finally {
        factory.createShader = createShader;
      }
    }
  };
}
