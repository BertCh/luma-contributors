// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {addResidentJoinPass} from './resident-join-gpu-kernels';

/**
 * GPU-side refusal of non-finite float32 results for GPU-resident (over-cap) datasets. One pass per
 * column ORs a flag when a valid row holds a NaN or infinity (all exponent bits set); only the
 * four byte flag is read back, so the result columns themselves are never read.
 */
export async function hasResidentNonFiniteValues(props: {
  device: Device;
  id: string;
  rowCount: number;
  columns: readonly {values: Buffer; validity: Buffer}[];
  assertCurrent(): void;
}): Promise<boolean> {
  const {device, rowCount} = props;
  if (!props.columns.length || !rowCount) return false;
  const flag = device.createBuffer({
    id: `${props.id}-nonfinite-flag`,
    byteLength: 4,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST,
    data: new Uint32Array(1)
  });
  const graph = new GPUCommandGraph(device, {id: `${props.id}-nonfinite`});
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  let fence: ReturnType<Device['createFence']> | undefined;
  try {
    const view = (name: string, buffer: Buffer, length: number): GraphDataView<'uint32'> =>
      graph.createDataView(
        graph.importBuffer({id: name, usage: buffer.usage, byteLength: buffer.byteLength}, buffer),
        {format: 'uint32', length}
      );
    // The graph resolves one physical buffer per import, so every pass shares one flag view.
    const flagView = view('nonfinite-flag', flag, 1);
    for (const [index, column] of props.columns.entries())
      addResidentJoinPass(graph, {
        id: `${props.id}-nonfinite-${index}`,
        count: rowCount,
        bindings: {
          values: {
            view: view(`nonfinite-values-${index}`, column.values, rowCount),
            access: 'read'
          },
          validity: {
            view: view(`nonfinite-validity-${index}`, column.validity, rowCount),
            access: 'read'
          },
          flag: {view: flagView, access: 'atomic'}
        },
        body: `  if (validity[index] != 0u && (values[index] & 0x7f800000u) == 0x7f800000u) { atomicOr(&flag[0], 1u); }`
      });
    compiled = graph.compile();
    props.assertCurrent();
    const encoder = device.createCommandEncoder();
    compiled.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    fence = device.createFence();
    await Promise.race([fence.signaled, device.lost]);
    props.assertCurrent();
    const bytes = await flag.readAsync(0, 4);
    return new Uint32Array(bytes.slice().buffer)[0] !== 0;
  } finally {
    fence?.destroy();
    compiled?.destroy();
    flag.destroy();
  }
}
