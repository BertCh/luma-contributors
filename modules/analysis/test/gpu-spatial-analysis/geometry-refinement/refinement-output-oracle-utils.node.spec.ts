// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {expect, it, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createRefinementOracleHarness} from './refinement-output-oracle-utils';
import {executeGPUOracleOperation} from './refinement-output-oracle-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import * as graphTestUtils from '../../utils/gpu-contributor-test-utils';

it('oracle readback buffers declare COPY_SRC and explicitly upload poison or input literals', () => {
  const device = createNullWebGPUDevice();
  const createBuffer = device.createBuffer.bind(device);
  const uploads: number[][] = [];
  vi.spyOn(device, 'createBuffer').mockImplementation(props => {
    const buffer = createBuffer(props);
    vi.spyOn(buffer, 'write').mockImplementation(values => {
      const bytes = ArrayBuffer.isView(values)
        ? new Uint8Array(values.buffer, values.byteOffset, values.byteLength)
        : new Uint8Array(values);
      uploads.push(Array.from(new Uint32Array(bytes.slice().buffer)));
    });
    return buffer;
  });
  const harness = createRefinementOracleHarness(device);
  try {
    const integerOutput = harness.view('uint32x2', 2);
    const floatOutput = harness.view('float32x2', 2);
    const input = harness.view('uint32', 2, [71, 4093]);
    for (const view of [integerOutput, floatOutput, input]) {
      // readAsync stages a GPU copy. STORAGE/COPY_DST alone produces an invalid copy on
      // real WebGPU; it cannot certify either the independent oracle or the initial poison.
      expect(view.buffer.usage & Buffer.COPY_SRC, 'readback needs COPY_SRC').toBe(Buffer.COPY_SRC);
      expect(view.buffer.usage & Buffer.STORAGE).toBe(Buffer.STORAGE);
      expect(view.buffer.usage & Buffer.COPY_DST).toBe(Buffer.COPY_DST);
    }
    expect(uploads).toEqual([
      [913, 913, 913, 913],
      // IEEE-754 encoding of the exact f32 integer 913 is 0x44644000.
      [0x44644000, 0x44644000, 0x44644000, 0x44644000],
      [71, 4093]
    ]);
  } finally {
    harness.destroy();
    vi.restoreAllMocks();
    device.destroy();
  }
});

it('R5 oracle helper rejects a contributor that skips the second submission', async () => {
  const device = createNullWebGPUDevice();
  const createBuffer = device.createBuffer.bind(device);
  const buffers: Buffer[] = [];
  vi.spyOn(device, 'createBuffer').mockImplementation(props => {
    const buffer = createBuffer(props);
    const memory = new Uint8Array(buffer.byteLength);
    buffers.push(buffer);
    vi.spyOn(buffer, 'write').mockImplementation(values => {
      memory.set(
        ArrayBuffer.isView(values)
          ? new Uint8Array(values.buffer, values.byteOffset, values.byteLength)
          : new Uint8Array(values)
      );
    });
    vi.spyOn(buffer, 'readAsync').mockImplementation(async () => memory.slice());
    return buffer;
  });
  vi.spyOn(GPUCommandGraph.prototype, 'compile').mockReturnValue({
    id: 'once',
    destroy: () => {}
  } as never);
  let submissions = 0;
  let assertions = 0;
  vi.spyOn(graphTestUtils, 'submitGraph').mockImplementation(() => {
    if (submissions++ === 0) buffers[0].write(Uint32Array.of(7));
    return {} as never;
  });
  const operation = {
    outputs: {answer: {format: 'uint32' as const, length: 1}},
    createContributor: () => ({id: 'once', getCommandNodes: () => []})
  };
  try {
    await expect(
      executeGPUOracleOperation(device, operation, {}, rows => {
        assertions++;
        expect(rows['answer']).toEqual([7]);
      })
    ).rejects.toThrow();
    expect(assertions).toBe(2);
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});
