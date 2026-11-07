// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUVariableStandardization} from '@luma-contributors/graph/gpu-dataframe';

/** Builds a GPU sample z-score recipe; null source values have already been encoded as NaN. */
export async function runExample(
  device: Device,
  {graph, values}: {graph: GPUCommandGraph<undefined>; values: Float32Array}
) {
  const buffer = device.createBuffer({
    id: 'values',
    data: values,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const handle = graph.importBuffer(
    {id: 'values', byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  // #region example
  const input = graph.createDataView(handle, {format: 'float32', length: values.length});
  const output = createTransientView(
    graph,
    'output',
    'float32',
    values.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  graph.add(
    new GPUVariableStandardization({values: input, method: 'z-score', variance: 'sample', output})
  );
  // #endregion example
  return {
    values: input,
    output,
    destroy() {
      buffer.destroy();
    }
  };
}
