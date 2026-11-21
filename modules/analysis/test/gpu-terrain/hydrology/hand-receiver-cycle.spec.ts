// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_TERRAIN_DRAINAGE_UNRESOLVED,
  GPUTerrainHeightAboveDrainage
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-height-above-drainage';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('GPUTerrainHeightAboveDrainage reports a two-cell receiver cycle as unresolved (RT-123)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 2 x 1 ESRI directions [1 (east), 16 (west)] form a rootless cycle; no stream cells. Neither
  // path reaches a stream or a cell without a receiver, so both drainage cells are unknown,
  // HAND is NaN and the run must not report convergence. In-place doubling collapses the
  // pointers to [0, 0] or [1, 1], which an unchecked fixpoint reads as "no drainage".
  const grid = makeGPURasterMetadata({width: 2, height: 1});
  const inputs = [
    createInputBuffer(device, new Float32Array([1, 2])),
    createInputBuffer(device, new Uint32Array([1, 16])),
    createInputBuffer(device, new Uint32Array([0, 0]))
  ];
  const handBuffer = createOutputBuffer(device, 2);
  const drainageBuffer = createOutputBuffer(device, 2);
  const convergedBuffer = createOutputBuffer(device, 1);
  const graph = new GPUCommandGraph(device, {id: 'hand-receiver-cycle'});
  const elevation = importGraphBuffer(graph, 'elevation', inputs[0], 'float32', 2);
  graph.add(
    new GPUTerrainHeightAboveDrainage({
      grid,
      elevation: makeGPURasterBand(elevation, grid, {id: 'elevation'}),
      flowDirections: importGraphBuffer(graph, 'directions', inputs[1], 'uint32', 2),
      streams: importGraphBuffer(graph, 'streams', inputs[2], 'uint32', 2),
      output: {
        heightAboveDrainage: importGraphBuffer(graph, 'hand', handBuffer, 'float32', 2),
        drainageCells: importGraphBuffer(graph, 'drainage', drainageBuffer, 'uint32', 2),
        status: {converged: importGraphBuffer(graph, 'converged', convergedBuffer, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const hand = await readFloat32(handBuffer, 2);
  const drainage = await readUint32(drainageBuffer, 2);
  const converged = await readUint32(convergedBuffer, 1);
  expect(drainage).toEqual([GPU_TERRAIN_DRAINAGE_UNRESOLVED, GPU_TERRAIN_DRAINAGE_UNRESOLVED]);
  expect(hand.every(value => Number.isNaN(value))).toBe(true);
  expect(converged).toEqual([0]);
  compiled.destroy();
  for (const buffer of [...inputs, handBuffer, drainageBuffer, convergedBuffer]) buffer.destroy();
});
