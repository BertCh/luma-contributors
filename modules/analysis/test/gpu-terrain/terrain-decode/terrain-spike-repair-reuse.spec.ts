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
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUTerrainSpikeRepair} from '../../../src/gpu-terrain/terrain-decode/gpu-terrain-spike-repair';
import {getSpikeRepairReuseFrames} from './terrain-spike-repair-reuse-fixture';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('spike repair rebuilds connectivity and resets component and seam state on a smaller second frame (RT-139)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 5, height: 5});
  const {first, second, secondValidity} = getSpikeRepairReuseFrames();
  const elevationBuffer = createInputBuffer(device, first);
  const validityBuffer = createInputBuffer(device, new Uint32Array(25).fill(1));
  const buffers = {
    values: createOutputBuffer(device, 25),
    validity: createOutputBuffer(device, 25),
    labels: createOutputBuffer(device, 25),
    statistics: createOutputBuffer(device, 5),
    converged: createOutputBuffer(device, 1),
    limit: createOutputBuffer(device, 1)
  };
  const graph = new GPUCommandGraph(device);
  const elevation = makeGPURasterBand(
    importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', 25),
    grid
  );
  elevation.validity = importGraphBuffer(graph, 'input-validity', validityBuffer, 'uint32', 25);
  const contributor = new GPUTerrainSpikeRepair({
    grid,
    elevation,
    output: {
      values: importGraphBuffer(graph, 'values', buffers.values, 'float32', 25),
      outputValidity: importGraphBuffer(graph, 'validity', buffers.validity, 'uint32', 25),
      labels: importGraphBuffer(graph, 'labels', buffers.labels, 'uint32', 25),
      statistics: importGraphBuffer(graph, 'statistics', buffers.statistics, 'uint32', 5),
      status: {
        converged: importGraphBuffer(graph, 'converged', buffers.converged, 'uint32', 1),
        iterationLimitReached: importGraphBuffer(graph, 'limit', buffers.limit, 'uint32', 1)
      }
    }
  });
  graph.add(contributor);
  const compiled = graph.compile();
  try {
    // A 2x2 interior plateau has eight unordered border edges. Its four pixels share minimum
    // label 6; the other 21 pixels share label 0. All seams vote +1, so exactly one component
    // shifts down 256. Statistics: jumps, repaired pixels, shifted components, remaining, converged.
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(buffers.values, 25)).toEqual(new Array(25).fill(100));
    expect(await readUint32(buffers.validity, 25)).toEqual(new Array(25).fill(1));
    const firstLabels = new Array(25).fill(0);
    for (const cell of [6, 7, 11, 12]) firstLabels[cell] = 6;
    expect(await readUint32(buffers.labels, 25)).toEqual(firstLabels);
    expect(await readUint32(buffers.statistics, 5)).toEqual([8, 4, 1, 0, 1]);
    expect(await readUint32(buffers.converged, 1)).toEqual([1]);
    expect(await readUint32(buffers.limit, 1)).toEqual([0]);

    // Reverse the shift and shrink to cell 6; cell 11 is now nodata. Label 6 stays active,
    // so retaining its previous eight seams prevents the required repair (3/11 < 0.8).
    // Cell 6 has three valid neighbours (1, 5, 7); nodata is never a seam. CSR connectivity,
    // component sizes and vote sign change. Each valid output must be 100.
    elevationBuffer.write(second);
    validityBuffer.write(secondValidity);
    submitGraph(device, compiled, undefined);
    const repaired = await readFloat32(buffers.values, 25);
    for (let cell = 0; cell < 25; cell++) {
      if (cell === 11) expect(Number.isNaN(repaired[cell])).toBe(true);
      else {
        expect(Number.isFinite(repaired[cell])).toBe(true);
        expect(repaired[cell]).toBe(100);
      }
    }
    expect(await readUint32(buffers.validity, 25)).toEqual(Array.from(secondValidity));
    const secondLabels = new Array(25).fill(0);
    secondLabels[6] = 6;
    secondLabels[11] = 11;
    expect(await readUint32(buffers.labels, 25)).toEqual(secondLabels);
    expect(await readUint32(buffers.statistics, 5)).toEqual([3, 1, 1, 0, 1]);
    expect(await readUint32(buffers.converged, 1)).toEqual([1]);
    expect(await readUint32(buffers.limit, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    contributor.destroy();
    for (const buffer of [elevationBuffer, validityBuffer, ...Object.values(buffers)])
      buffer.destroy();
  }
});
