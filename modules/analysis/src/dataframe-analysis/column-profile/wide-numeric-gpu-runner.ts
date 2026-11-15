// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, addGPUCommandNodes} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUWideNumericScan, type GPUWideNumericScanProps} from './gpu-wide-numeric-scan';

export type WideNumericKernelProps = Omit<GPUWideNumericScanProps, 'output'> & {
  device: Device;
  initialWords?: Uint32Array;
};

/** Runs the phases over all chunks of a resident Int64/Uint64/Float64 word vector and reads back only `resultWordCount` words. */
export async function runWideNumericKernel(props: WideNumericKernelProps): Promise<Uint32Array> {
  const {device} = props;
  validateCapacity(props.id, 'resultWordCount', props.resultWordCount);
  const byteLength = Math.max(16, props.resultWordCount * 4);
  validateGPUStorageBindingByteLength(
    device,
    props.id,
    'result buffer',
    byteLength,
    'Reduce resultWordCount'
  );
  const output = device.createBuffer({
    id: `${props.id}-result`,
    data: props.initialWords
      ? new Uint32Array(byteLength / 4).map((_, index) => props.initialWords![index] ?? 0)
      : new Uint32Array(byteLength / 4),
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const graph = new GPUCommandGraph(device, {id: props.id});
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  try {
    const result = graph.createDataView(
      graph.importBuffer({id: 'result', byteLength, usage: output.usage}, output),
      {format: 'uint32', length: byteLength / 4}
    );
    const scan = new GPUWideNumericScan({...props, output: result});
    addGPUCommandNodes(graph, scan.getCommandNodes(graph));
    compiled = graph.compile();
    const encoder = device.createCommandEncoder();
    compiled.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    const bytes = await output.readAsync(0, byteLength);
    return new Uint32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + props.resultWordCount * 4)
    );
  } finally {
    compiled?.destroy();
    output.destroy();
  }
}
