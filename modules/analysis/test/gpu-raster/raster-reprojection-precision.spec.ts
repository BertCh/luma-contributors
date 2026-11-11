// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {compileProjectionProgram, type CompiledProjection} from '@luma.gl/graph/gpu-project';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPURasterReprojection} from '../../src/raster-analysis/gpu-raster-reprojection';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

// raster-fields-04. Float32 holds a UTM northing of 4.6e6 m to 0.5 m. The kernel used to build
// absolute destination centres in float32 and subtract the absolute source origin, so the source
// pixel was off by up to a quarter of a 1 m cell and most of a 0.25 m cell.
//
// The source raster is the plane `value = ROW_WEIGHT * row + column`. Bilinear interpolation
// reproduces a plane exactly, so the output at a destination pixel is the plane at its fractional
// source pixel coordinate, which the test computes from the two affines in float64.
const ROW_WEIGHT = 64;
const EASTING = 500000;
const NORTHING = 4600000;

type Case = {
  name: string;
  sourceGrid: GPURasterMetadata;
  destinationGrid: GPURasterMetadata;
  inverseProjection?: CompiledProjection;
  /** Float64 source pixel coordinate `[column, row]` of destination pixel `(column, row)`. */
  getSourcePixel: (column: number, row: number) => [number, number];
};

function makeGrid(
  width: number,
  height: number,
  cellSize: number,
  originX: number,
  originY: number
): GPURasterMetadata {
  // North-up: row 0 is the northernmost row.
  return {
    width,
    height,
    affine: [cellSize, 0, originX, 0, -cellSize, originY],
    pixelInterpretation: 'area'
  };
}

const CASES: Case[] = [
  {
    // 1 m cells, grids offset by a quarter of a cell.
    name: '1 m cells at a northing of 4.6e6, same CRS',
    sourceGrid: makeGrid(32, 32, 1, EASTING, NORTHING + 100),
    destinationGrid: makeGrid(16, 16, 1, EASTING + 3.25, NORTHING + 90.25),
    getSourcePixel: (column, row) => [3.25 + column, 9.75 + row]
  },
  {
    // 0.25 m cells, grids offset by 0.1 m (0.4 cell).
    name: '0.25 m cells at a northing of 4.6e6, same CRS',
    sourceGrid: makeGrid(64, 64, 0.25, EASTING, NORTHING + 100),
    destinationGrid: makeGrid(16, 16, 0.25, EASTING + 2.1, NORTHING + 97.9),
    getSourcePixel: (column, row) => [8.4 + column, 8.4 + row]
  },
  {
    // The destination grid is local (metres from the projection origin); the inverse projection
    // moves it to the absolute frame, and its result is relative to `destinationOrigin`. The
    // source origin is 100.75 m north of that origin and 0.3 m east. Both grids are south-up here:
    // the centre of destination row 0 is local y 122.3 + 0.5 = 122.8, which is 22.05 m past the
    // source origin, the centre of source row 21.55. (Float32 would round the absolute northing
    // 4,600,122.8 to 4,600,123.0 and land on row 21.75.)
    name: '1 m cells through an inverse projection with a destination origin at 4.6e6',
    sourceGrid: {
      width: 48,
      height: 48,
      affine: [1, 0, EASTING + 0.3, 0, 1, NORTHING + 100.75],
      pixelInterpretation: 'area'
    },
    destinationGrid: {
      width: 16,
      height: 16,
      affine: [1, 0, 4, 0, 1, 122.3],
      pixelInterpretation: 'area'
    },
    inverseProjection: compileProjectionProgram({
      precision: 'local-f32',
      destinationOrigin: [EASTING, NORTHING],
      operations: [{type: 'affine', scale: [1, 1], offset: [EASTING, NORTHING]}]
    }),
    getSourcePixel: (column, row) => [4 + column - 0.3, 122.3 + row - 100.75]
  }
];

for (const entry of CASES) {
  it(`GPURasterReprojection resolves the source pixel to well under a cell: ${entry.name} (raster-fields-04)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {sourceGrid, destinationGrid} = entry;
    const graph = new GPUCommandGraph(device, {id: 'reprojection-precision'});
    const buffers: Buffer[] = [];
    const track = (buffer: Buffer) => {
      buffers.push(buffer);
      return buffer;
    };
    const sourceValues = new Float32Array(sourceGrid.width * sourceGrid.height);
    for (let row = 0; row < sourceGrid.height; row++) {
      for (let column = 0; column < sourceGrid.width; column++) {
        sourceValues[row * sourceGrid.width + column] = ROW_WEIGHT * row + column;
      }
    }
    const count = destinationGrid.width * destinationGrid.height;
    const run = async (method: 'nearest' | 'bilinear') => {
      const valuesBuffer = track(createOutputBuffer(device, count));
      const validityBuffer = track(createOutputBuffer(device, count));
      const contributor = new GPURasterReprojection({
        id: `reprojection-${method}`,
        sourceGrid,
        destinationGrid,
        input: {
          id: `source-${method}`,
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(
              graph,
              `source-values-${method}`,
              track(createInputBuffer(device, sourceValues)),
              'float32',
              sourceValues.length
            )
          }
        },
        inverseProjection: entry.inverseProjection,
        method,
        output: {
          values: importGraphBuffer(graph, `values-${method}`, valuesBuffer, 'float32', count),
          validity: importGraphBuffer(graph, `validity-${method}`, validityBuffer, 'uint32', count)
        }
      });
      graph.add(contributor);
      return {contributor, valuesBuffer, validityBuffer};
    };
    const bilinear = await run('bilinear');
    const nearest = await run('nearest');
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const bilinearValues = await readFloat32(bilinear.valuesBuffer, count);
      const nearestValues = await readFloat32(nearest.valuesBuffer, count);
      expect(Array.from(await readUint32(bilinear.validityBuffer, count))).toEqual(
        Array(count).fill(1)
      );
      expect(Array.from(await readUint32(nearest.validityBuffer, count))).toEqual(
        Array(count).fill(1)
      );
      let largestPixelError = 0;
      for (let row = 0; row < destinationGrid.height; row++) {
        for (let column = 0; column < destinationGrid.width; column++) {
          const index = row * destinationGrid.width + column;
          const [sourceColumn, sourceRow] = entry.getSourcePixel(column, row);
          // Area pixels: the centre of source pixel (c, r) is at pixel coordinate (c, r).
          const expected = ROW_WEIGHT * sourceRow + sourceColumn;
          largestPixelError = Math.max(
            largestPixelError,
            Math.abs(bilinearValues[index] - expected)
          );
          expect(nearestValues[index]).toBe(
            ROW_WEIGHT * Math.floor(sourceRow + 0.5) + Math.floor(sourceColumn + 0.5)
          );
        }
      }
      // A row error of 0.01 cell moves the plane by 0.64. The old kernel was off by 0.25 cell (16)
      // on the 1 m grid and by up to 0.9 cell on the 0.25 m grid.
      expect(largestPixelError).toBeLessThan(0.05);
    } finally {
      compiled.destroy();
      bilinear.contributor.destroy();
      nearest.contributor.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  });
}
