// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUTerrainPeakProminence} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-peak-prominence';
import {CPUTerrainPeakProminenceOperation} from '../../../src/gpu-terrain/terrain-features/cpu-terrain-peak-prominence-operation';
import {GPUTerrainPeakProminenceOperation} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-peak-prominence-operation';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/raster-analysis';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('TWN-08 integer-origin upload workaround preserves calibrated heights and elder IDs in every form', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 2, height: 1});
  const raw = Int32Array.of(16777216, 16777217);
  // Subtract in JavaScript's exact integer range before converting to f32. Raw nodata, if any,
  // must be captured in a separate mask before this rebase; this fixture has no missing samples.
  const rebased = Float32Array.from(raw, value => value - 16777216);
  expect(Array.from(rebased)).toEqual([0, 1]);
  const elevation = createInputBuffer(device, rebased);
  const prominence = createOutputBuffer(device, 2);
  const parent = createOutputBuffer(device, 2);
  const graph = new GPUCommandGraph(device);
  graph.add(
    new GPUTerrainPeakProminence({
      grid,
      elevation: makeGPURasterBand(
        importGraphBuffer(graph, 'elevation', elevation, 'float32', 2),
        grid
      ),
      output: {
        prominence: importGraphBuffer(graph, 'prominence', prominence, 'float32', 2),
        parentPeak: importGraphBuffer(graph, 'parent', parent, 'uint32', 2)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Heights 0 and 1 meet at saddle 0. Cell 1 survives, cell 0 has zero prominence.
    expect(await readUint32(parent, 2)).toEqual([1, 0xffffffff]);
    expect(await readFloat32(prominence, 2)).toEqual([0, Infinity]);
    for (const Operation of [
      GPUTerrainPeakProminenceOperation,
      CPUTerrainPeakProminenceOperation
    ]) {
      const rows = await runOperationGraph(device, {
        inputs: {elevation: {format: 'float32', values: rebased}},
        operations: [
          new Operation({
            grid,
            elevation: {
              id: 'elevation',
              format: 'float32',
              storage: {kind: 'buffer', values: 'elevation'}
            },
            output: {prominence: 'prominence', parentPeak: 'parent'}
          })
        ],
        outputs: ['prominence', 'parent']
      });
      expect(Array.from(rows['parent'])).toEqual([1, 0xffffffff]);
      expect(Array.from(rows['prominence'])).toEqual([0, Infinity]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of [elevation, prominence, parent]) buffer.destroy();
  }
});
