// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH,
  GPURasterFractionalCoverage,
  getGPURasterFractionalCoverageParameterValues
} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {GPURasterReprojection} from '../../src/raster-analysis/gpu-raster-reprojection';
import {makeGPURasterMetadata} from '../../src/raster-analysis/raster-grid/gpu-raster-grid';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

/** A western- and southern-hemisphere grid: negative origin words used to break the WGSL. */
const GRID = makeGPURasterMetadata({width: 2, height: 1, origin: [-120, -30]});

function createCoverage(device: Device, grid: GPURasterMetadata) {
  const graph = new GPUCommandGraph(device, {id: 'fractional-coverage-georeference'});
  // A unit square straddling both cells: x in [-119.5, -118.5], y in [-31, -30].
  const inputs = [
    createInputBuffer(device, Float32Array.of(-119.5, -31, -118.5, -31, -118.5, -30, -119.5, -30)),
    createInputBuffer(device, Uint32Array.of(0, 1)),
    createInputBuffer(device, Uint32Array.of(0, 1)),
    createInputBuffer(device, Uint32Array.of(0, 4))
  ];
  const outputs = {
    zoneIds: createOutputBuffer(device, 2),
    cellIds: createOutputBuffer(device, 2),
    fractions: createOutputBuffer(device, 2),
    count: createOutputBuffer(device, 1),
    requiredCount: createOutputBuffer(device, 1),
    overflow: createOutputBuffer(device, 1),
    zoneValidity: createOutputBuffer(device, 1)
  };
  const settings = new GPUParameterBuffer(device, {
    id: 'coverage-settings',
    format: 'float32',
    length: GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH,
    values: getGPURasterFractionalCoverageParameterValues(grid)
  });
  graph.add(
    new GPURasterFractionalCoverage({
      grid,
      zones: {
        positions: importGraphBuffer(graph, 'positions', inputs[0], 'float32x2', 4),
        featureOffsets: importGraphBuffer(graph, 'features', inputs[1], 'uint32', 2),
        polygonOffsets: importGraphBuffer(graph, 'polygons', inputs[2], 'uint32', 2),
        ringOffsets: importGraphBuffer(graph, 'rings', inputs[3], 'uint32', 2)
      },
      parameters: settings.importToGraph(graph),
      output: {
        zoneIds: importGraphBuffer(graph, 'zone-ids', outputs.zoneIds, 'uint32', 2),
        cellIds: importGraphBuffer(graph, 'cell-ids', outputs.cellIds, 'uint32', 2),
        fractions: importGraphBuffer(graph, 'fractions', outputs.fractions, 'float32', 2),
        status: {
          count: importGraphBuffer(graph, 'count', outputs.count, 'uint32', 1),
          requiredCount: importGraphBuffer(graph, 'required', outputs.requiredCount, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1)
        },
        zoneValidity: importGraphBuffer(graph, 'zone-validity', outputs.zoneValidity, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  const read = async () => {
    submitGraph(device, compiled, undefined);
    const count = (await readUint32(outputs.count, 1))[0];
    return {
      count,
      cellIds: Array.from(await readUint32(outputs.cellIds, 2)).slice(0, count),
      fractions: Array.from(await readFloat32(outputs.fractions, 2)).slice(0, count)
    };
  };
  const destroy = () => {
    compiled.destroy();
    settings.destroy();
    [...inputs, ...Object.values(outputs)].forEach((buffer: Buffer) => buffer.destroy());
  };
  return {read, settings, destroy};
}

it('GPURasterFractionalCoverage reads its affine from parameters and accepts negative origins', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const parameterized = createCoverage(device, GRID);
  expect(await parameterized.read()).toEqual({count: 2, cellIds: [0, 1], fractions: [0.5, 0.5]});
  // Same compiled graph, grid moved one cell east: the square now covers half of cell 0 only.
  const moved = makeGPURasterMetadata({width: 2, height: 1, origin: [-119, -30]});
  parameterized.settings.write(getGPURasterFractionalCoverageParameterValues(moved));
  expect(await parameterized.read()).toEqual({count: 1, cellIds: [0], fractions: [0.5]});
  parameterized.destroy();

  expect(() =>
    getGPURasterFractionalCoverageParameterValues({...GRID, affine: [1, 2, 0, 2, 4, 0]})
  ).toThrow(
    /^getGPURasterFractionalCoverageParameterValues grid\.affine must be invertible \(finite nonzero determinant\)/
  );
});

it('GPURasterReprojection compiles and samples grids with negative origin words', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'reprojection-negative-origin'});
  const source = createInputBuffer(device, Float32Array.of(10, 20));
  const values = createOutputBuffer(device, 3);
  const validity = createOutputBuffer(device, 3);
  graph.add(
    new GPURasterReprojection({
      sourceGrid: GRID,
      destinationGrid: makeGPURasterMetadata({width: 3, height: 1, origin: [-120, -30]}),
      input: {
        id: 'source',
        format: 'float32',
        storage: {kind: 'buffer', values: importGraphBuffer(graph, 'source', source, 'float32', 2)}
      },
      method: 'nearest',
      output: {
        values: importGraphBuffer(graph, 'values', values, 'float32', 3),
        validity: importGraphBuffer(graph, 'validity', validity, 'uint32', 3)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = Array.from(await readFloat32(values, 3));
  expect(result.slice(0, 2)).toEqual([10, 20]);
  expect(result[2]).toBeNaN();
  expect(Array.from(await readUint32(validity, 3))).toEqual([1, 1, 0]);
  compiled.destroy();
  [source, values, validity].forEach(buffer => buffer.destroy());
});
