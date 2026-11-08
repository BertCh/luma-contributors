// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPURasterTileReprojection,
  type GPURasterReprojectionSourceTile
} from '../../src/raster-analysis/gpu-raster-tile-reprojection';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import reference from '../oracles/raster-tile-reprojection.json';

const sourceGrid = {
  width: 4,
  height: 2,
  affine: [1, 0, 0, 0, -1, 2] as const,
  pixelInterpretation: 'area' as const
};

for (const scenario of ['seam', 'missing', 'overlap'] as const) {
  for (const noDataPolicy of ['strict', 'renormalize'] as const) {
    it(`GPU source tile routing matches independent ${scenario}/${noDataPolicy} reference`, async () => {
      const testDevice = await getWebGPUTestDevice('core');
      requireTestDevice(testDevice);
      const device = testDevice;
      const graph = new GPUCommandGraph(device);
      const buffers: ReturnType<typeof createOutputBuffer>[] = [];
      function tile(
        values: number[],
        pixelBounds: GPURasterReprojectionSourceTile['pixelBounds']
      ): GPURasterReprojectionSourceTile {
        const buffer = createInputBuffer(device, Float32Array.from(values));
        buffers.push(buffer);
        return {
          pixelBounds,
          input: {
            id: `tile-${buffers.length}`,
            format: 'float32',
            noDataValue: -999,
            storage: {
              kind: 'buffer',
              values: importGraphBuffer(
                graph,
                `tile-${buffers.length}`,
                buffer,
                'float32',
                values.length
              )
            }
          }
        };
      }
      const sources = [tile([10, 20, 50, 60], [0, 0, 2, 2])];
      if (scenario !== 'missing') sources.push(tile([30, 40, 70, 80], [2, 0, 4, 2]));
      if (scenario === 'overlap') sources.push(tile([200, -999, 600, 700], [1, 0, 3, 2]));
      const values = createOutputBuffer(device, 8);
      const validity = createOutputBuffer(device, 8);
      buffers.push(values, validity);
      const contributor = new GPURasterTileReprojection({
        sourceGrid,
        sources,
        noDataPolicy,
        method: 'bilinear',
        destinationGrid: {...sourceGrid, affine: [1, 0, 0.5, 0, -1, 2]},
        output: {
          values: importGraphBuffer(graph, 'values', values, 'float32', 8),
          validity: importGraphBuffer(graph, 'validity', validity, 'uint32', 8)
        }
      });
      graph.add(contributor);
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const expected =
          scenario === 'seam'
            ? noDataPolicy === 'strict'
              ? reference.seamStrict
              : reference.seamRenormalize
            : scenario === 'missing'
              ? noDataPolicy === 'strict'
                ? reference.missingStrict
                : reference.missingRenormalize
              : noDataPolicy === 'strict'
                ? reference.overlapStrict
                : reference.overlapRenormalize;
        const result = await readFloat32(values, 8);
        expect(await readUint32(validity, 8)).toEqual(
          expected.map(value => Number(value !== null))
        );
        expected.forEach((value, index) =>
          value === null
            ? expect(result[index]).toBeNaN()
            : expect(result[index]).toBeCloseTo(value, 5)
        );
        // Rerun the same compiled graph after changing borrowed GPU-resident source storage.
        buffers[0].write(Float32Array.from([110, 120, 150, 160]));
        submitGraph(device, compiled, undefined);
        const updated = await readFloat32(values, 8);
        expect(updated[0]).toBeCloseTo(scenario === 'overlap' ? 155 : 115, 5);
      } finally {
        compiled.destroy();
        contributor.destroy();
        buffers.forEach(buffer => buffer.destroy());
      }
    });
  }
}

it('an empty resident tile set produces explicit uncovered nodata', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const values = createOutputBuffer(device, 8);
  const validity = createOutputBuffer(device, 8);
  const contributor = new GPURasterTileReprojection({
    sourceGrid,
    destinationGrid: sourceGrid,
    sources: [],
    method: 'nearest',
    output: {
      values: importGraphBuffer(graph, 'values', values, 'float32', 8),
      validity: importGraphBuffer(graph, 'validity', validity, 'uint32', 8)
    }
  });
  graph.add(contributor);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(validity, 8)).toEqual(Array(8).fill(0));
    expect((await readFloat32(values, 8)).every(Number.isNaN)).toBe(true);
  } finally {
    compiled.destroy();
    contributor.destroy();
    values.destroy();
    validity.destroy();
  }
});

it('four resident tiles supply all bilinear taps across a corner seam', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  const sources = [
    {values: [10, 20], pixelBounds: [0, 0, 2, 1] as const},
    {values: [30, 40], pixelBounds: [2, 0, 4, 1] as const},
    {values: [50, 60], pixelBounds: [0, 1, 2, 2] as const},
    {values: [70, 80], pixelBounds: [2, 1, 4, 2] as const}
  ].map((tile, index) => {
    const buffer = createInputBuffer(device, Float32Array.from(tile.values));
    buffers.push(buffer);
    return {
      pixelBounds: tile.pixelBounds,
      input: {
        id: `tile-${index}`,
        format: 'float32' as const,
        storage: {
          kind: 'buffer' as const,
          values: importGraphBuffer(graph, `tile-${index}`, buffer, 'float32', 2)
        }
      }
    };
  });
  const values = createOutputBuffer(device, 3);
  const validity = createOutputBuffer(device, 3);
  buffers.push(values, validity);
  const contributor = new GPURasterTileReprojection({
    sourceGrid,
    sources,
    method: 'bilinear',
    destinationGrid: {...sourceGrid, width: 3, height: 1, affine: [1, 0, 0.5, 0, -1, 1.5]},
    output: {
      values: importGraphBuffer(graph, 'values', values, 'float32', 3),
      validity: importGraphBuffer(graph, 'validity', validity, 'uint32', 3)
    }
  });
  graph.add(contributor);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(values, 3)).toEqual(reference.diagonalRasterio);
    expect(await readUint32(validity, 3)).toEqual([1, 1, 1]);
  } finally {
    compiled.destroy();
    contributor.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});

it('32 calibrated source tiles execute on a core device without tile binding arrays', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  const sources = Array.from({length: 32}, (_, index) => {
    const buffer = createInputBuffer(device, Uint32Array.of(index));
    buffers.push(buffer);
    return {
      pixelBounds: [index, 0, index + 1, 1] as const,
      input: {
        id: `tile-${index}`,
        format: 'uint32' as const,
        scale: 2,
        offset: 5,
        storage: {
          kind: 'buffer' as const,
          values: importGraphBuffer(graph, `tile-${index}`, buffer, 'uint32', 1)
        }
      }
    };
  });
  const grid = {...sourceGrid, width: 32, height: 1};
  const values = createOutputBuffer(device, 32);
  const validity = createOutputBuffer(device, 32);
  buffers.push(values, validity);
  const contributor = new GPURasterTileReprojection({
    sourceGrid: grid,
    destinationGrid: grid,
    sources,
    method: 'nearest',
    output: {
      values: importGraphBuffer(graph, 'values', values, 'float32', 32),
      validity: importGraphBuffer(graph, 'validity', validity, 'uint32', 32)
    }
  });
  graph.add(contributor);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(values, 32)).toEqual(
      Array.from({length: 32}, (_, index) => index * 2 + 5)
    );
    expect(await readUint32(validity, 32)).toEqual(Array(32).fill(1));
  } finally {
    compiled.destroy();
    contributor.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});
