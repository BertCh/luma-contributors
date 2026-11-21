// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {createLeadingSlotRasterFixture} from './leading-slot-raster-fixture';

it.each([
  false,
  true
])('rejects unowned leading slots in raster splats (sourcePaths=%s)', sourcePaths => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let count = 0;
  const fixture = createLeadingSlotRasterFixture(
    {
      input<Format extends GPUVectorFormat>(format: Format, values: Float32Array | Uint32Array) {
        const buffer = graph.createTransientBuffer({
          id: `input-${count++}`,
          byteLength: values.byteLength,
          usage: Buffer.STORAGE
        });
        const view = graph.createDataView(buffer, {
          format,
          length: values.byteLength / getGPUVectorFormatInfo(format).byteLength
        });
        memory.write(view, values);
        return view;
      },
      output<Format extends GPUVectorFormat>(format: Format, length: number) {
        const buffer = graph.createTransientBuffer({
          id: `output-${count++}`,
          byteLength: length * getGPUVectorFormatInfo(format).byteLength,
          usage: Buffer.STORAGE
        });
        return graph.createDataView(buffer, {format, length});
      }
    },
    sourcePaths
  );
  try {
    const nodes = fixture.contributor.getCommandNodes(graph);
    executeCommandNodesOnCPU(nodes, memory);
    if (sourcePaths) {
      // A distinct zero-length source-path edge at (.5,.5), cost 7, paints exactly three
      // centers at radius 1. Removing the entire source-path splat must fail this control.
      expect(Array.from(memory.readFloat32(fixture.values))).toEqual([7, 7, 7, Math.fround(1e30)]);
      memory.write(fixture.sourcePathOffsets!, Uint32Array.of(1, 1, 1));
      executeCommandNodesOnCPU(nodes, memory);
    }
    // Main row intervals own only an unreachable edge; the replay path slot has no owner.
    expect(Array.from(memory.readFloat32(fixture.values))).toEqual(
      new Array(4).fill(Math.fround(1e30))
    );
  } finally {
    device.destroy();
  }
});
