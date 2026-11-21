// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {GPURasterRankFilter} from '../../src/raster-analysis/gpu-raster-rank-filter';
import {GPURasterFillNoData} from '../../src/raster-analysis/gpu-raster-fill-nodata';
import {GPURasterReprojection} from '../../src/raster-analysis/gpu-raster-reprojection';
import {GPURasterTileReprojection} from '../../src/raster-analysis/gpu-raster-tile-reprojection';
import {addSurfacePreparationRecipe} from '../../src/gpu-spatial-analysis/recipes/surface-preparation-recipe';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';

for (const operator of ['rank', 'fill', 'reprojection', 'tiles'] as const) {
  it.each([
    'own',
    'inherited'
  ] as const)(`${operator} retains %s channel and calibration captured before graph wiring (R1; raster-terrain:F3)`, async representation => {
    const device = await getWebGPUTestDevice('core');
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const grid = {
      width: 1,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const texture = device.createTexture({
      id: 'measurement',
      format: 'rgba32float',
      width: 1,
      height: 1,
      usage: Texture.SAMPLE | Texture.COPY_DST
    });
    texture.writeData(Float32Array.of(7, 0, 1, 0));
    const maskBuffer = createInputBuffer(device, Uint32Array.of(1));
    const rejectedMaskBuffer = createInputBuffer(device, Uint32Array.of(0));
    const valuesBuffer = createOutputBuffer(device, 1);
    const validityBuffer = createOutputBuffer(device, 1);
    const filledBuffer = createOutputBuffer(device, 1);
    const input: GPURasterBand = {
      id: 'measurement',
      format: 'float32',
      scale: 2,
      offset: 10,
      validity: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', 1),
      storage: {
        kind: 'texture',
        channel: 2,
        view: graph.createTextureView(
          graph.importTexture(
            {
              id: 'measurement',
              format: 'rgba32float',
              width: 1,
              height: 1,
              usage: texture.props.usage
            },
            texture
          )
        )
      }
    };
    const inheritedMetadata = {scale: 2, offset: 10, channel: 2};
    if (representation === 'inherited') {
      Object.setPrototypeOf(input, {
        get scale() {
          return inheritedMetadata.scale;
        },
        get offset() {
          return inheritedMetadata.offset;
        }
      });
      delete input.scale;
      delete input.offset;
      Object.setPrototypeOf(input.storage, {
        get channel() {
          return inheritedMetadata.channel;
        }
      });
      delete input.storage.channel;
    }
    const output = {
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', 1),
      validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', 1)
    };
    const contributor =
      operator === 'rank'
        ? new GPURasterRankFilter({grid, input, radius: 0, output})
        : operator === 'fill'
          ? new GPURasterFillNoData({
              grid,
              input,
              maximumDistance: 1,
              output: {
                ...output,
                filled: importGraphBuffer(graph, 'filled', filledBuffer, 'uint32', 1),
                status: {}
              }
            })
          : operator === 'reprojection'
            ? new GPURasterReprojection({sourceGrid: grid, destinationGrid: grid, input, output})
            : new GPURasterTileReprojection({
                sourceGrid: grid,
                destinationGrid: grid,
                sources: [{pixelBounds: [0, 0, 1, 1], input}],
                output
              });
    graph.add(contributor);
    // The published texture converter compiles lazily: metadata and mask bindings must stay captured.
    if (representation === 'inherited') {
      inheritedMetadata.channel = 0;
      inheritedMetadata.scale = 3;
    } else {
      input.storage.channel = 0;
      input.scale = 3;
    }
    input.validity = importGraphBuffer(graph, 'rejected-mask', rejectedMaskBuffer, 'uint32', 1);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const [value] = await readFloat32(valuesBuffer, 1);
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBe(12);
      expect(await readUint32(validityBuffer, 1)).toEqual([1]);
    } finally {
      compiled.destroy();
      if ('destroy' in contributor) contributor.destroy();
      texture.destroy();
      for (const buffer of [
        maskBuffer,
        rejectedMaskBuffer,
        valuesBuffer,
        validityBuffer,
        filledBuffer
      ])
        buffer.destroy();
    }
  });
}

for (const operator of ['rank', 'fill', 'reprojection', 'tiles', 'surface-recipe'] as const) {
  it(`${operator} uses the selected RGBA channel, raw nodata, calibration and mask on resubmission (RT-09)`, async () => {
    const device = await getWebGPUTestDevice('core');
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const track = (buffer: Buffer) => {
      buffers.push(buffer);
      return buffer;
    };
    const grid = {
      width: 5,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const textureWidth = operator === 'tiles' ? 3 : 5;
    const texture = device.createTexture({
      id: 'rgba',
      format: 'rgba32float',
      width: textureWidth,
      height: 1,
      usage: Texture.SAMPLE | Texture.COPY_DST
    });
    // Other channels deliberately disagree with blue, including at masked and raw-nodata cells.
    const packChannels = (values: readonly number[]) =>
      Float32Array.from(values.flatMap(value => [101, 202, value, 404]));
    texture.writeData(packChannels([1, -9, 100, 4, 5].slice(0, textureWidth)));
    const maskBuffer = track(
      createInputBuffer(device, Uint32Array.from([1, 1, 0, 1, 1].slice(0, textureWidth)))
    );
    const input: GPURasterBand = {
      id: 'blue',
      format: 'float32',
      noDataValue: -9,
      scale: 2,
      offset: 10,
      validity: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', textureWidth),
      storage: {
        kind: 'texture',
        channel: 2,
        view: graph.createTextureView(
          graph.importTexture(
            {
              id: 'rgba',
              format: 'rgba32float',
              width: textureWidth,
              height: 1,
              usage: texture.props.usage
            },
            texture
          )
        )
      }
    };
    const valuesBuffer = track(createOutputBuffer(device, 5));
    const validityBuffer = track(createOutputBuffer(device, 5));
    const filledBuffer = track(createOutputBuffer(device, 5));
    const invalidCountBuffer = track(createOutputBuffer(device, 1));
    const output = {
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', 5),
      validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', 5)
    };
    const contributor =
      operator === 'surface-recipe'
        ? undefined
        : operator === 'rank'
          ? new GPURasterRankFilter({grid, input, radius: 1, noDataPolicy: 'ignore', output})
          : operator === 'fill'
            ? new GPURasterFillNoData({
                grid,
                input,
                maximumDistance: 1,
                method: 'nearest',
                output: {
                  ...output,
                  filled: importGraphBuffer(graph, 'filled', filledBuffer, 'uint32', 5),
                  status: {
                    invalidCount: importGraphBuffer(
                      graph,
                      'invalid-count',
                      invalidCountBuffer,
                      'uint32',
                      1
                    )
                  }
                }
              })
            : operator === 'reprojection'
              ? new GPURasterReprojection({sourceGrid: grid, destinationGrid: grid, input, output})
              : new GPURasterTileReprojection({
                  sourceGrid: grid,
                  destinationGrid: grid,
                  sources: [
                    {pixelBounds: [0, 0, 3, 1], input},
                    {
                      pixelBounds: [3, 0, 5, 1],
                      input: {
                        id: 'integer-tile',
                        format: 'sint32',
                        scale: 2,
                        offset: 10,
                        storage: {
                          kind: 'buffer',
                          values: importGraphBuffer(
                            graph,
                            'integer-tile',
                            track(createInputBuffer(device, Int32Array.of(4, 5))),
                            'sint32',
                            2
                          )
                        }
                      }
                    }
                  ],
                  output
                });
    if (contributor) {
      graph.add(contributor);
    } else {
      // A rejected alias must not reserve default output IDs before the corrected same-ID recipe.
      const aliasedBuffer = track(createOutputBuffer(device, 5));
      expect(() =>
        addSurfacePreparationRecipe(graph, {
          grid,
          input,
          maximumDistance: 1,
          output: {
            originalValidity: importGraphBuffer(
              graph,
              'aliased-original',
              aliasedBuffer,
              'uint32',
              5
            ),
            unresolvedCount: importGraphBuffer(graph, 'aliased-count', aliasedBuffer, 'uint32', 1)
          }
        })
      ).toThrow(/overlap|share a buffer/);
      const originalValidityBuffer = track(createOutputBuffer(device, 5));
      const originalValidity = importGraphBuffer(
        graph,
        'corrected-original',
        originalValidityBuffer,
        'uint32',
        5
      );
      expect(() =>
        addSurfacePreparationRecipe(graph, {
          grid,
          input,
          maximumDistance: 1,
          output: {
            originalValidity: graph.createDataView(originalValidity.buffer, {
              format: 'uint32',
              length: 1
            })
          }
        })
      ).toThrow(/provided output must have 5 rows/);
      addSurfacePreparationRecipe(graph, {
        grid,
        input,
        maximumDistance: 1,
        method: 'nearest',
        filter: {radius: 0},
        output: {
          ...output,
          originalValidity,
          filled: importGraphBuffer(graph, 'filled', filledBuffer, 'uint32', 5),
          unresolvedCount: importGraphBuffer(
            graph,
            'invalid-count',
            invalidCountBuffer,
            'uint32',
            1
          )
        }
      });
    }
    const compiled = graph.compile();
    try {
      for (const frame of [0, 1]) {
        if (frame === 1) {
          texture.writeData(packChannels([1, 2, 3, 4, 5].slice(0, textureWidth)));
          maskBuffer.write(Uint32Array.from([0, 1, 1, 1, 1].slice(0, textureWidth)));
        }
        submitGraph(device, compiled, undefined);
        const expected =
          operator === 'rank'
            ? frame === 0
              ? [12, NaN, NaN, 20, 20]
              : [NaN, 16, 16, 18, 20]
            : operator === 'fill' || operator === 'surface-recipe'
              ? frame === 0
                ? [12, 12, 18, 18, 20]
                : [14, 14, 16, 18, 20]
              : frame === 0
                ? [12, NaN, NaN, 18, 20]
                : [NaN, 14, 16, 18, 20];
        const actual = await readFloat32(valuesBuffer, 5);
        for (const [index, value] of expected.entries()) {
          if (Number.isNaN(value)) expect(actual[index]).toBeNaN();
          else {
            expect(Number.isFinite(actual[index])).toBe(true);
            expect(actual[index]).toBe(value);
          }
        }
        expect(await readUint32(validityBuffer, 5)).toEqual(
          expected.map(value => (Number.isFinite(value) ? 1 : 0))
        );
        if (operator === 'fill' || operator === 'surface-recipe') {
          expect(await readUint32(filledBuffer, 5)).toEqual(
            frame === 0 ? [0, 1, 1, 0, 0] : [1, 0, 0, 0, 0]
          );
          expect(await readUint32(invalidCountBuffer, 1)).toEqual([0]);
        }
      }
    } finally {
      compiled.destroy();
      if (contributor && 'destroy' in contributor) contributor.destroy();
      texture.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
