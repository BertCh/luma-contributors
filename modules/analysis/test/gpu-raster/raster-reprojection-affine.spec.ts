// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata, GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {
  getGPURasterReprojectionParameterValues,
  GPURasterReprojection
} from '../../src/raster-analysis/gpu-raster-reprojection';
import {
  GPURasterTileReprojection,
  type GPURasterReprojectionSourceTile
} from '../../src/raster-analysis/gpu-raster-tile-reprojection';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';

it('B1 native signed warps reject INT32_MIN nodata and retain the next exact label', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const tiled of [false, true]) {
    const graph = new GPUCommandGraph(device);
    const sourceGrid = {
      width: 2,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'point' as const
    };
    const buffers = [
      createInputBuffer(device, Int32Array.of(-2147483648, -2147483647)),
      createOutputBuffer(device, 2),
      createOutputBuffer(device, 2),
      createOutputBuffer(device, 2)
    ];
    const input: GPURasterBand = {
      id: 'input',
      format: 'sint32',
      noDataValue: -2147483648,
      storage: {kind: 'buffer', values: importGraphBuffer(graph, 'input', buffers[0], 'sint32', 2)}
    };
    const props = {
      sourceGrid,
      destinationGrid: sourceGrid,
      method: 'nearest' as const,
      output: {
        values: importGraphBuffer(graph, 'values', buffers[1], 'float32', 2),
        validity: importGraphBuffer(graph, 'validity', buffers[2], 'uint32', 2),
        labels: importGraphBuffer(graph, 'labels', buffers[3], 'sint32', 2)
      }
    };
    const contributor = tiled
      ? new GPURasterTileReprojection({...props, sources: [{pixelBounds: [0, 0, 2, 1], input}]})
      : new GPURasterReprojection({...props, input});
    graph.add(contributor);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(buffers[2], 2)).toEqual([0, 1]);
      const bytes = await buffers[3].readAsync(0, 8);
      expect(new Int32Array(bytes.buffer, bytes.byteOffset, 2)[1]).toBe(-2147483647);
    } finally {
      compiled.destroy();
      contributor.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});

it('S1 bilinear policies distinguish a missing positive-weight tap on both warp routes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const tiled of [false, true])
    for (const noDataPolicy of ['strict', 'propagate', 'renormalize'] as const) {
      const graph = new GPUCommandGraph(device);
      const sourceGrid = {
        width: 2,
        height: 1,
        affine: [1, 0, 0, 0, 1, 0] as const,
        pixelInterpretation: 'point' as const
      };
      const buffers = [
        createInputBuffer(device, Float32Array.of(20, NaN)),
        createOutputBuffer(device, 1),
        createOutputBuffer(device, 1)
      ];
      const input: GPURasterBand = {
        id: 'input',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'input', buffers[0], 'float32', 2)
        }
      };
      const props = {
        sourceGrid,
        destinationGrid: {...sourceGrid, width: 1, affine: [1, 0, 0.25, 0, 1, 0] as const},
        method: 'bilinear' as const,
        noDataPolicy,
        output: {
          values: importGraphBuffer(graph, 'values', buffers[1], 'float32', 1),
          validity: importGraphBuffer(graph, 'validity', buffers[2], 'uint32', 1)
        }
      };
      const contributor = tiled
        ? new GPURasterTileReprojection({...props, sources: [{pixelBounds: [0, 0, 2, 1], input}]})
        : new GPURasterReprojection({...props, input});
      graph.add(contributor);
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(buffers[2], 1)).toEqual([noDataPolicy === 'renormalize' ? 1 : 0]);
        const [value] = await readFloat32(buffers[1], 1);
        // Positive weights are .75 and .25; dropping the missing second tap leaves 20.
        if (noDataPolicy === 'renormalize') expect(value).toBe(20);
        else expect(value).toBeNaN();
      } finally {
        compiled.destroy();
        contributor.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    }
});

it('NAM-05 samples 0 at nearest and 25 at bilinear for source column .25', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const tiled of [false, true])
    for (const method of ['nearest', 'bilinear'] as const)
      for (const noDataPolicy of ['strict', 'propagate', 'renormalize'] as const) {
        const graph = new GPUCommandGraph(device);
        const sourceGrid = {
          width: 2,
          height: 1,
          affine: [1, 0, 0, 0, 1, 0] as const,
          pixelInterpretation: 'point' as const
        };
        const buffers = [
          createInputBuffer(device, Float32Array.of(0, 100)),
          createOutputBuffer(device, 1),
          createOutputBuffer(device, 1)
        ];
        const input = {
          id: 'input',
          format: 'float32' as const,
          storage: {
            kind: 'buffer' as const,
            values: importGraphBuffer(graph, 'input', buffers[0], 'float32', 2)
          }
        };
        const props = {
          sourceGrid,
          destinationGrid: {...sourceGrid, width: 1, affine: [1, 0, 0.25, 0, 1, 0] as const},
          method,
          noDataPolicy,
          output: {
            values: importGraphBuffer(graph, 'values', buffers[1], 'float32', 1),
            validity: importGraphBuffer(graph, 'validity', buffers[2], 'uint32', 1)
          }
        };
        const contributor = tiled
          ? new GPURasterTileReprojection({...props, sources: [{pixelBounds: [0, 0, 2, 1], input}]})
          : new GPURasterReprojection({...props, input});
        graph.add(contributor);
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          expect(await readUint32(buffers[2], 1)).toEqual([1]);
          const [value] = await readFloat32(buffers[1], 1);
          expect(Number.isFinite(value)).toBe(true);
          expect(value).toBe(method === 'nearest' ? 0 : 25);
        } finally {
          compiled.destroy();
          contributor.destroy();
          for (const buffer of buffers) buffer.destroy();
        }
      }
});

it('URT-04 preserves adjacent native labels and validity through identity and half-cell warps', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const tiled of [false, true])
    for (const format of ['uint32', 'sint32'] as const)
      for (const shift of [0, 0.5])
        for (const noDataValue of [undefined, 16777217]) {
          const graph = new GPUCommandGraph(device);
          const sourceGrid = {
            width: 2,
            height: 1,
            affine: [1, 0, 0, 0, 1, 0] as const,
            pixelInterpretation: 'point' as const
          };
          const buffers = [
            createInputBuffer(
              device,
              format === 'uint32'
                ? Uint32Array.of(16777216, 16777217)
                : Int32Array.of(16777216, 16777217)
            ),
            createOutputBuffer(device, 2),
            createOutputBuffer(device, 2),
            createOutputBuffer(device, 2)
          ];
          const input = {
            id: 'input',
            format,
            noDataValue,
            storage: {
              kind: 'buffer' as const,
              values: importGraphBuffer(graph, 'input', buffers[0], format, 2)
            }
          } as GPURasterBand;
          const props = {
            sourceGrid,
            destinationGrid: {...sourceGrid, affine: [1, 0, shift, 0, 1, 0] as const},
            method: 'nearest' as const,
            output: {
              values: importGraphBuffer(graph, 'values', buffers[1], 'float32', 2),
              validity: importGraphBuffer(graph, 'validity', buffers[2], 'uint32', 2),
              labels: importGraphBuffer(graph, 'labels', buffers[3], format, 2)
            }
          };
          const contributor = tiled
            ? new GPURasterTileReprojection({
                ...props,
                sources: [{pixelBounds: [0, 0, 2, 1], input}]
              })
            : new GPURasterReprojection({...props, input});
          graph.add(contributor);
          const compiled = graph.compile();
          try {
            submitGraph(device, compiled, undefined);
            // Nearest ties select the higher column: shift .5 samples [1, outside].
            expect(await readUint32(buffers[2], 2)).toEqual(
              shift === 0
                ? [1, noDataValue === undefined ? 1 : 0]
                : [noDataValue === undefined ? 1 : 0, 0]
            );
            const labels = await readUint32(buffers[3], 2);
            // Only validity=1 promises a readable label; invalid label rows are unspecified.
            if (shift === 0) expect(labels[0]).toBe(16777216);
            if (noDataValue === undefined) expect(labels[shift === 0 ? 1 : 0]).toBe(16777217);
          } finally {
            compiled.destroy();
            contributor.destroy();
            for (const buffer of buffers) buffer.destroy();
          }
        }
});

it('F7 bilinear renormalization returns 7 from one normal float32 tiny-weight neighbour', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const tiled of [false, true]) {
    const graph = new GPUCommandGraph(device);
    const sourceGrid = {
      width: 2,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'point' as const
    };
    const destinationGrid = {...sourceGrid, width: 1, affine: [1, 0, 1e-32, 0, 1, 0] as const};
    const buffers = [
      createInputBuffer(device, Float32Array.of(0, 7)),
      createInputBuffer(device, Uint32Array.of(0, 1)),
      createOutputBuffer(device, 1),
      createOutputBuffer(device, 1)
    ];
    const input = {
      id: 'source',
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: importGraphBuffer(graph, 'source', buffers[0], 'float32', 2)
      },
      validity: importGraphBuffer(graph, 'source-validity', buffers[1], 'uint32', 2)
    };
    const props = {
      sourceGrid,
      destinationGrid,
      method: 'bilinear' as const,
      noDataPolicy: 'renormalize' as const,
      output: {
        values: importGraphBuffer(graph, 'output', buffers[2], 'float32', 1),
        validity: importGraphBuffer(graph, 'validity', buffers[3], 'uint32', 1)
      }
    };
    const contributor = tiled
      ? new GPURasterTileReprojection({...props, sources: [{pixelBounds: [0, 0, 2, 1], input}]})
      : new GPURasterReprojection({...props, input});
    graph.add(contributor);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(buffers[3], 1)).toEqual([1]);
      const [value] = await readFloat32(buffers[2], 1);
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeCloseTo(7, 5);
    } finally {
      compiled.destroy();
      contributor.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});

for (const tiled of [false, true]) {
  for (const pixelInterpretation of ['point', 'area'] as const) {
    for (const method of ['nearest', 'bilinear'] as const) {
      it(`preserves an identical sheared grid and updates packed parameters: tiled=${tiled}, ${pixelInterpretation}, ${method} (RT-04)`, async () => {
        const device = await getWebGPUTestDevice('core');
        requireTestDevice(device);
        const graph = new GPUCommandGraph(device);
        const buffers: Buffer[] = [];
        const track = (buffer: Buffer) => {
          buffers.push(buffer);
          return buffer;
        };
        const sourceGrid = {
          width: 8,
          height: 1,
          affine: [1, 1, 0, 1, 1 + 0.55 * 2 ** -23, 0] as const,
          pixelInterpretation
        };
        // Columns 4 and 7 must remain distinguishable: the original packer sampled 80 for 50.
        const sourceValues = Float32Array.of(10, 20, 30, 40, 50, 60, 70, 80);
        const input = {
          id: 'source',
          format: 'float32' as const,
          storage: {
            kind: 'buffer' as const,
            values: importGraphBuffer(
              graph,
              'source',
              track(createInputBuffer(device, sourceValues)),
              'float32',
              8
            )
          }
        };
        const valuesBuffer = track(createOutputBuffer(device, 8));
        const validityBuffer = track(createOutputBuffer(device, 8));
        const parameterValues = getGPURasterReprojectionParameterValues({
          sourceGrid,
          destinationGrid: sourceGrid
        });
        const parameters = new GPUParameterBuffer(device, {
          id: 'affine',
          format: 'float32',
          length: 20,
          values: parameterValues
        });
        const props = {
          sourceGrid,
          destinationGrid: sourceGrid,
          method,
          parameters: parameters.importToGraph(graph),
          output: {
            values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', 8),
            validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', 8)
          }
        };
        const contributor = tiled
          ? new GPURasterTileReprojection({...props, sources: [{pixelBounds: [0, 0, 8, 1], input}]})
          : new GPURasterReprojection({...props, input});
        graph.add(contributor);
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          expect(await readFloat32(valuesBuffer, 8)).toEqual(Array.from(sourceValues));
          expect(await readUint32(validityBuffer, 8)).toEqual(Array(8).fill(1));

          // Move destination origin one source column, preserving the strongly sheared axes.
          parameters.write(
            getGPURasterReprojectionParameterValues({
              sourceGrid,
              destinationGrid: {...sourceGrid, affine: [1, 1, 1, 1, sourceGrid.affine[4], 1]}
            })
          );
          submitGraph(device, compiled, undefined);
          const shifted = await readFloat32(valuesBuffer, 8);
          expect(shifted.slice(0, 7)).toEqual([20, 30, 40, 50, 60, 70, 80]);
          expect(shifted[7]).toBeNaN();
          expect(await readUint32(validityBuffer, 8)).toEqual([1, 1, 1, 1, 1, 1, 1, 0]);

          // A legacy manual upload must produce invalid samples rather than misread new slots.
          for (const legacyVersion of [0, 2]) {
            parameterValues[17] = legacyVersion;
            parameters.write(parameterValues);
            submitGraph(device, compiled, undefined);
            expect(await readUint32(validityBuffer, 8)).toEqual(Array(8).fill(0));
            for (const value of await readFloat32(valuesBuffer, 8)) expect(value).toBeNaN();
          }
        } finally {
          compiled.destroy();
          contributor.destroy();
          parameters.destroy();
          for (const buffer of buffers) buffer.destroy();
        }
      });
    }
  }
}

// Reuse the same graph while changing only destination georeference words. Each returned
// snapshot is a real submission, followed by both value and validity readback.
async function sampleNearestCases(
  sourceGrid: GPURasterMetadata,
  destinationGrids: GPURasterMetadata[],
  sourceValues: Float32Array,
  tiled: boolean,
  mutateSourcePlan: boolean = false
): Promise<{values: number[]; validity: number[]}[]> {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const sourceBuffer = createInputBuffer(device, sourceValues);
  const count = destinationGrids[0].width * destinationGrids[0].height;
  const valuesBuffer = createOutputBuffer(device, count);
  const validityBuffer = createOutputBuffer(device, count);
  const input = {
    id: 'source',
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: importGraphBuffer(graph, 'source', sourceBuffer, 'float32', sourceValues.length)
    }
  };
  const parameters = new GPUParameterBuffer(device, {
    id: 'affine',
    format: 'float32',
    length: 20,
    values: getGPURasterReprojectionParameterValues({
      sourceGrid,
      destinationGrid: destinationGrids[0]
    })
  });
  const props = {
    sourceGrid,
    destinationGrid: destinationGrids[0],
    method: 'nearest' as const,
    parameters: parameters.importToGraph(graph),
    output: {
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', count),
      validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', count)
    }
  };
  const sourceTile: GPURasterReprojectionSourceTile = {
    pixelBounds: [0, 0, sourceGrid.width, sourceGrid.height],
    input
  };
  const contributor = tiled
    ? new GPURasterTileReprojection({
        ...props,
        sources: [sourceTile]
      })
    : new GPURasterReprojection({...props, input});
  if (mutateSourcePlan && contributor instanceof GPURasterTileReprojection) {
    contributor.props.sources = [sourceTile, sourceTile];
    sourceTile.pixelBounds = [1, 0, 2, 1];
    sourceTile.input.scale = 99;
  }
  graph.add(contributor);
  const compiled = graph.compile();
  const snapshots: {values: number[]; validity: number[]}[] = [];
  try {
    for (const destinationGrid of destinationGrids) {
      parameters.write(getGPURasterReprojectionParameterValues({sourceGrid, destinationGrid}));
      submitGraph(device, compiled, undefined);
      snapshots.push({
        values: await readFloat32(valuesBuffer, count),
        validity: await readUint32(validityBuffer, count)
      });
    }
    return snapshots;
  } finally {
    compiled.destroy();
    contributor.destroy();
    parameters.destroy();
    sourceBuffer.destroy();
    valuesBuffer.destroy();
    validityBuffer.destroy();
  }
}

for (const tiled of [false, true]) {
  for (const axis of ['column', 'row'] as const) {
    it(`keeps an exact source center after scale/translation cancellation: ${axis}, tiled=${tiled} (R1)`, async () => {
      const sourceGrid: GPURasterMetadata = {
        width: axis === 'column' ? 8 : 1,
        height: axis === 'row' ? 8 : 1,
        affine: axis === 'column' ? [5, 0, 0, 0, 1, 0] : [1, 0, 0, 0, 5, 0],
        pixelInterpretation: 'point'
      };
      const destinationGrid: GPURasterMetadata = {
        ...sourceGrid,
        width: axis === 'column' ? 4 : 1,
        height: axis === 'row' ? 4 : 1,
        affine:
          axis === 'column' ? [16777232, 0, -50331676, 0, 1, 0] : [1, 0, 0, 0, 16777232, -50331676]
      };
      const [snapshot] = await sampleNearestCases(
        sourceGrid,
        [destinationGrid],
        Float32Array.of(0, 1, 2, 3, 4, 5, 6, 7),
        tiled
      );
      expect(snapshot.validity).toEqual([0, 0, 0, 1]);
      expect(snapshot.values[3]).toBe(4);
      for (const value of snapshot.values.slice(0, 3)) expect(value).toBeNaN();
    });

    it(`distinguishes the float32 neighbours of a nearest half-cell tie: ${axis}, tiled=${tiled} (R4)`, async () => {
      const sourceGrid: GPURasterMetadata = {
        width: axis === 'column' ? 2 : 1,
        height: axis === 'row' ? 2 : 1,
        affine: [1, 0, 0, 0, 1, 0],
        pixelInterpretation: 'point'
      };
      const destinationGrids: GPURasterMetadata[] = [0.5 - 2 ** -25, 0.5, 0.5 + 2 ** -24].map(
        coordinate => ({
          ...sourceGrid,
          width: 1,
          height: 1,
          affine: axis === 'column' ? [1, 0, coordinate, 0, 1, 0] : [1, 0, 0, 0, 1, coordinate]
        })
      );
      const snapshots = await sampleNearestCases(
        sourceGrid,
        destinationGrids,
        Float32Array.of(10, 20),
        tiled
      );
      expect(snapshots.map(snapshot => snapshot.values[0])).toEqual([10, 20, 20]);
      expect(snapshots.map(snapshot => snapshot.validity[0])).toEqual([1, 1, 1]);
    });
  }
}

it('preserves column 417792 on a portable-dispatch-sized sheared strip (RT-04 review)', async () => {
  const grid: GPURasterMetadata = {
    width: 417794,
    height: 1,
    affine: [1, 1, 0, 1, 1.047, 0],
    pixelInterpretation: 'point'
  };
  const [snapshot] = await sampleNearestCases(
    grid,
    [grid],
    Float32Array.from({length: 417794}, (_, column) => column),
    false
  );
  expect(snapshot.values[417792]).toBe(417792);
  expect(snapshot.validity[417792]).toBe(1);
});

it('samples the privately validated tile plan after public and borrowed metadata mutation (R3)', async () => {
  const grid: GPURasterMetadata = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0],
    pixelInterpretation: 'point'
  };
  const [snapshot] = await sampleNearestCases(grid, [grid], Float32Array.of(10), true, true);
  expect(snapshot.values).toEqual([10]);
  expect(snapshot.validity).toEqual([1]);
});
