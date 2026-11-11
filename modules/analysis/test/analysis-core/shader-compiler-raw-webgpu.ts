// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../utils/require-test-device';

// Plain WebGPU for the shader-compiler reproductions: one compute shader, one dispatch, one
// read-back. Nothing of this library or of @luma.gl/graph is between the WGSL and the browser, so
// a spec built on it can be pasted into a browser bug report as it stands.

/** One compute shader run once (one workgroup), and the buffers it sees. */
export type RawComputeShader = {
  /** WGSL with an entry point `main` and `@group(0) @binding(0) var<storage, read_write> output: array<u32>`. */
  code: string;
  /** Number of `u32` words in `output`. */
  outputLength: number;
  /** Words of the uniform buffer at `@binding(1)`, when the shader declares one. */
  uniformWords?: readonly number[];
  /** Values of the read-only `array<f32>` storage buffer at `@binding(2)`, when the shader declares one. */
  inputValues?: Float32Array;
};

/** The GPUDevice of the shared WebGPU test device. */
export async function getRawTestDevice(): Promise<GPUDevice> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device.handle;
}

/** `vendor architecture` of the adapter behind a device, for messages (`apple metal-3` on the test GPU). */
export function getAdapterLabel(device: GPUDevice): string {
  const information = device.adapterInfo;
  return `${information?.vendor ?? 'unknown'} ${information?.architecture ?? ''}`.trim();
}

/** Compiles and runs one compute shader and returns its `output` words. Throws on a WGSL or validation error. */
export async function runRawComputeShader(
  device: GPUDevice,
  shader: RawComputeShader
): Promise<Uint32Array> {
  device.pushErrorScope('validation');
  const module = device.createShaderModule({code: shader.code});
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: {module, entryPoint: 'main'}
  });
  const byteLength = shader.outputLength * Uint32Array.BYTES_PER_ELEMENT;
  const output = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });
  const buffers = [output, staging];
  const entries: GPUBindGroupEntry[] = [{binding: 0, resource: {buffer: output}}];
  if (shader.uniformWords) {
    // A uniform buffer is at least 16 bytes here so that any struct of up to four words fits.
    const words = new Uint32Array(Math.max(4, Math.ceil(shader.uniformWords.length / 4) * 4));
    words.set(shader.uniformWords);
    const uniform = device.createBuffer({
      size: words.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(uniform, 0, words);
    entries.push({binding: 1, resource: {buffer: uniform}});
    buffers.push(uniform);
  }
  if (shader.inputValues) {
    const input = device.createBuffer({
      size: shader.inputValues.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(input, 0, shader.inputValues);
    entries.push({binding: 2, resource: {buffer: input}});
    buffers.push(input);
  }
  const bindGroup = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries});
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(output, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  const validationError = await device.popErrorScope();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  if (validationError) {
    throw new Error(`WebGPU validation error: ${validationError.message}`);
  }
  return words;
}
