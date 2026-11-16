// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createTrackTableOffsetsNode} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/movement-common';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('publishes 65537 track boundaries with gaps and refreshes a smaller active count', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const trackCount = 65537;
  const groupCount = 21846;
  const tracks = createInputBuffer(
    device,
    Uint32Array.from({length: groupCount * 2}, (_, row) => Math.floor(row / 2) * 3)
  );
  const count = createInputBuffer(device, new Uint32Array([groupCount * 2]));
  const offsets = createOutputBuffer(device, trackCount + 1);
  const node = createTrackTableOffsetsNode(graph, {
    id: 'offsets',
    operation: 'TestTrackOffsets',
    tracks: importGraphBuffer(graph, 'tracks', tracks, 'uint32'),
    count: importGraphBuffer(graph, 'count', count, 'uint32'),
    offsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32'),
    capacity: groupCount * 2,
    trackCount
  });
  graph.add({id: 'track-offsets', getCommandNodes: () => [node]});
  const compiled = graph.compile();
  try {
    for (const total of [groupCount * 2, 2, 0]) {
      count.write(new Uint32Array([total]));
      submitGraph(device, compiled, undefined);
      expect(Array.from(await readUint32(offsets, trackCount + 1))).toEqual(
        Array.from({length: trackCount + 1}, (_, boundary) =>
          Math.min(Math.ceil(boundary / 3) * 2, total)
        )
      );
    }
  } finally {
    compiled.destroy();
    tracks.destroy();
    count.destroy();
    offsets.destroy();
  }
}, 120000);
