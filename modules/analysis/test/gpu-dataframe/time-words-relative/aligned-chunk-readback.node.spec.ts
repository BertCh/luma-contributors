// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {GPUTimeWordsRelative} from '../../../src/dataframe-analysis/time-words-relative/gpu-time-words-relative';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

for (const validity of [0, 7]) {
  it(`writes the live relative-time row rather than an aligned binding tail (mask=${validity})`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const view = <Format extends 'uint32x2' | 'uint32' | 'float32'>(
        name: string,
        format: Format
      ) => createTransientView(graph, name, format, 1);
      const [kernel] = getGPUCommandNodeKernelSources(
        new GPUTimeWordsRelative({
          timestamps: view('time', 'uint32x2'),
          mask: view('mask', 'uint32'),
          output: view('relative', 'float32'),
          origin: 0n,
          temporalContext: {
            unit: 'millisecond',
            epoch: 'unix',
            encoding: 'int64-words',
            axis: {kind: 'irregular'}
          }
        }).getCommandNodes(graph)
      );
      // Execute the actual body with padded bindings and a nonzero live row offset.
      // Only uniform transport is replaced: count=1 and origin=(0,0).
      const source = `@group(0) @binding(0) var<storage, read> timestamps: array<u32>;
@group(0) @binding(1) var<storage, read> timeValidity: array<u32>;
@group(0) @binding(2) var<storage, read_write> relative: array<f32>;
const timestampsOffset: u32 = 2u; const timeValidityOffset: u32 = 1u; const relativeOffset: u32 = 1u;
fn getChunkParameter(word: u32) -> u32 { return select(0u, 1u, word == 1u); }
${kernel.source.slice(kernel.source.indexOf('fn timeWordsSubtract'), kernel.source.indexOf('@compute'))}
@compute @workgroup_size(1) fn main() { let index = 0u; ${kernel.source.slice(kernel.source.lastIndexOf('if (index >= getChunkParameter(1u))'), kernel.source.lastIndexOf('}'))} }`;
      const result = new Float32Array([99, 99, 99, 99]);
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {
        0: {
          0: new Uint32Array([0, 0, 1, 0, 0, 0]),
          1: new Uint32Array([1, validity, 1, 1]),
          2: result
        }
      });
      expect(Array.from(result)).toEqual([99, validity === 0 ? NaN : 1, 99, 99]);
    } finally {
      device.destroy();
    }
  });
}
