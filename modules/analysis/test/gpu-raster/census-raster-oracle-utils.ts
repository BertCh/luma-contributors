// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {makeGPURasterMetadata} from '../../src/raster-analysis/raster-grid/index';
import {readFloat32, readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';

type Format = 'float32' | 'float32x2' | 'float32x4' | 'uint32';

/** Shared placement for the point-zone-only GPU case and its host-side geometry check. */
export function createCensusRasterJoinFixture() {
  return {
    grid: makeGPURasterMetadata({width: 2, height: 1, origin: [0, 1]}),
    points: [1.5, 0.5, 0.5, 0.5, 1.5, 0.5, 2.5, 0.5],
    zones: [1, 0]
  };
}

/** Small, poisoned caller-owned outputs: a missing zero write must fail the oracle. */
export function createCensusRasterHarness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const outputs = new Map<string, {buffer: Buffer; words: number; integer: boolean}>();
  let serial = 0;
  function input<SelectedFormat extends Format>(
    format: SelectedFormat,
    values: number[]
  ): GraphDataView<SelectedFormat> {
    const components = format === 'float32x2' ? 2 : format === 'float32x4' ? 4 : 1;
    const buffer = device.createBuffer({
      data: format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `census-${serial++}`,
      buffer,
      format,
      values.length / components
    );
  }
  return {
    graph,
    input,
    output<SelectedFormat extends Format>(name: string, format: SelectedFormat, length: number) {
      const components = format === 'float32x2' ? 2 : format === 'float32x4' ? 4 : 1;
      const words = length * components;
      const view = input(format, new Array(words).fill(12345));
      outputs.set(name, {buffer: buffers[buffers.length - 1], words, integer: format === 'uint32'});
      return view;
    },
    async run(...contributors: GPUCommandNodeProducer[]) {
      let compiled: ReturnType<typeof graph.compile> | undefined;
      try {
        contributors.forEach(contributor => graph.add(contributor));
        compiled = graph.compile();
        submitGraph(device, compiled, undefined);
        const result: Record<string, number[]> = {};
        for (const [name, output] of outputs) {
          result[name] = await (output.integer ? readUint32 : readFloat32)(
            output.buffer,
            output.words
          );
        }
        return result;
      } finally {
        compiled?.destroy();
        buffers.forEach(buffer => buffer.destroy());
      }
    }
  };
}

/** NaN-safe comparison; exact dyadic coordinates can use tolerance zero. */
export function expectCensusValues(actual: ArrayLike<number>, expected: number[], tolerance = 0) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) => {
    if (!Number.isFinite(value)) {
      expect(actual[index]).toBe(value);
    } else {
      expect(Number.isFinite(actual[index]), `element ${index}`).toBe(true);
      expect(Math.abs(actual[index] - value), `element ${index}`).toBeLessThanOrEqual(tolerance);
    }
  });
}
