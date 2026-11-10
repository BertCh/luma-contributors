// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';

// example:hide-start
declare const device: Device;
/** Observations as x, y pairs. */
declare const observations: Float32Array;
/** Ring-to-vertex offsets of a polygon layer, with one terminal entry. */
declare const ringOffsets: Uint32Array;
// example:hide-end
const graph = new GPUCommandGraph<void>(device, {id: 'data-views'});

// Your buffer: the graph imports it and never destroys it.
const pointBuffer = device.createBuffer({usage: Buffer.STORAGE, data: observations});
const pointHandle = graph.importBuffer(
  {id: 'observations', byteLength: pointBuffer.byteLength, usage: pointBuffer.usage},
  pointBuffer
);

// A view lays a format over the bytes: two float32 per row, one row per point.
const positions = graph.createDataView(pointHandle, {
  format: 'float32x2',
  length: observations.length / 2
});

// Offsets are an ordinary view too: one uint32 per entry.
const offsetBuffer = device.createBuffer({usage: Buffer.STORAGE, data: ringOffsets});
const ringOffsetView = graph.createDataView(
  graph.importBuffer(
    {id: 'ring-offsets', byteLength: offsetBuffer.byteLength, usage: offsetBuffer.usage},
    offsetBuffer
  ),
  {format: 'uint32', length: ringOffsets.length}
);
// example:hide-start
export {positions, ringOffsetView};
// example:hide-end
