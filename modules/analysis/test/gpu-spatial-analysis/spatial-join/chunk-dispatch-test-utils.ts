// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device, ComputePipeline, type ComputePass} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import type {GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';

/** Records the actual runtime dispatches without executing GPU work. */
export function getChunkDispatches(device: Device, nodes: readonly GPUCommandNode[]): number[][] {
  const pipelines = vi.spyOn(device, 'createComputePipeline').mockImplementation(
    props =>
      Object.assign(Object.create(ComputePipeline.prototype), {
        id: props.id,
        hash: '',
        destroy: () => {}
      }) as ComputePipeline
  );
  const dispatch = vi.spyOn(Kernel.prototype, 'dispatch').mockImplementation(() => {});
  const buffer = device.createBuffer({byteLength: 1_048_576, usage: Buffer.STORAGE});
  const commandEncoder = device.createCommandEncoder();
  try {
    for (const node of nodes) {
      if (node.type !== 'compute') throw new Error('Expected compute chunk nodes');
      const compiled = node.compile({device});
      try {
        compiled.encode({
          commandEncoder,
          computePass: {} as ComputePass,
          parameters: undefined,
          getBuffer: () => buffer,
          getTexture: () => {
            throw new Error('Unexpected texture');
          },
          getTextureView: () => {
            throw new Error('Unexpected texture view');
          },
          getExternalTexture: () => {
            throw new Error('Unexpected external texture');
          }
        });
      } finally {
        compiled.destroy?.();
      }
    }
    return dispatch.mock.calls.map(([, options]) => [
      options.x ?? 1,
      options.y ?? 1,
      options.z ?? 1
    ]);
  } finally {
    dispatch.mockRestore();
    pipelines.mockRestore();
    commandEncoder.destroy();
    buffer.destroy();
  }
}

/**
 * Names runtime binding-offset expressions for address assertions, using the reflected binding
 * locations. A read of the wrong uniform slot therefore names the wrong buffer and still fails.
 * This is only for source comparisons: parsing, audits and execution use the original WGSL.
 */
export function getNamedRuntimeBindingSource(source: string): string {
  const uniform = source.includes('var<uniform> chunkSettings:')
    ? 'chunkSettings'
    : source.includes('var<uniform> selectionSettings:')
      ? 'selectionSettings'
      : undefined;
  if (!uniform) return source;
  for (const binding of new WgslReflect(source).storage) {
    const word = binding.binding + (uniform === 'chunkSettings' ? 2 : 1);
    const expression =
      uniform === 'chunkSettings'
        ? `chunkSettings[${Math.floor(word / 4)}][${word % 4}]`
        : `selectionSettings[${Math.floor(word / 4)}].${'xyzw'[word % 4]}`;
    source = source.replaceAll(expression, `${binding.name}Offset`);
  }
  return uniform === 'chunkSettings'
    ? source
        .replaceAll('chunkSettings[0][0]', 'CHUNK_FIRST_ROW')
        .replaceAll('chunkSettings[2][2]', 'SOURCE_ROW_COUNT')
    : source;
}
