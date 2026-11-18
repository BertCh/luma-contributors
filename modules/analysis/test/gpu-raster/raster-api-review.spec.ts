// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../src/raster-analysis/raster-grid/index';
import {GPURasterNBR} from '../../src/raster-analysis/gpu-raster-spectral-indices';
import {
  GPURasterReprojection,
  getGPURasterReprojectionParameterValues
} from '../../src/raster-analysis/gpu-raster-reprojection';
import {GPURasterHANTS} from '../../src/raster-analysis/raster-time-series/gpu-raster-hants';
import {GPURasterTemporalResample} from '../../src/raster-analysis/raster-time-series/gpu-raster-temporal-resample';
import {GPUTerrainStreamNetwork} from '../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';
import {WeightsRig} from '../gpu-spatial-analysis/spatial-weights/spatial-weights-harness';
import {createVectorView, readFloat32, readUint32} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

it('publishes the 0 to 1 stream reach with separate buffers and rejects pooled endpoints', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const sourceCells = rig.output('uint32', 1);
    const targetCells = rig.output('uint32', 1);
    const vertexOffsets = rig.output('uint32', 2);
    const positionStorage = rig.output('float32', 4);
    const positions = rig.graph.createDataView(positionStorage.view.buffer, {
      format: 'float32x2',
      length: 2
    });
    const lengths = rig.output('float32', 1);
    const reachCount = rig.output('uint32', 1);
    const vertexCount = rig.output('uint32', 1);
    const overflow = rig.output('uint32', 1);
    const unresolvedCount = rig.output('uint32', 1);
    const props = {
      grid: makeGPURasterMetadata({width: 2, height: 1}),
      directionEncoding: 'esri' as const,
      flowDirections: rig.input(Uint32Array.of(1, 0), 'uint32', 2),
      streams: rig.input(Uint32Array.of(1, 1), 'uint32', 2),
      output: {
        sourceCells: sourceCells.view,
        targetCells: targetCells.view,
        vertexOffsets: vertexOffsets.view,
        positions,
        lengths: lengths.view,
        reachCount: reachCount.view,
        vertexCount: vertexCount.view,
        status: {overflow: overflow.view},
        unresolvedCount: unresolvedCount.view
      }
    };
    const pool = rig.output('uint32', 128);
    expect(
      () =>
        new GPUTerrainStreamNetwork({
          ...props,
          output: {
            ...props.output,
            sourceCells: rig.graph.createDataView(pool.view.buffer, {format: 'uint32', length: 1}),
            targetCells: rig.graph.createDataView(pool.view.buffer, {
              format: 'uint32',
              byteOffset: 256,
              length: 1
            })
          }
        })
    ).toThrow(/^terrain-stream-network .*without explicit per-dispatch accesses/);
    rig.run(new GPUTerrainStreamNetwork(props));
    expect(await readUint32(reachCount.buffer, 1)).toEqual([1]);
    expect(await readUint32(vertexCount.buffer, 1)).toEqual([2]);
    expect(await readUint32(sourceCells.buffer, 1)).toEqual([0]);
    expect(await readUint32(targetCells.buffer, 1)).toEqual([1]);
    expect(await readUint32(vertexOffsets.buffer, 2)).toEqual([0, 2]);
    expect(await readFloat32(positionStorage.buffer, 4)).toEqual([0.5, -0.5, 1.5, -0.5]);
    expect((await readFloat32(lengths.buffer, 1))[0]).toBeCloseTo(1, 6);
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
    expect(await readUint32(unresolvedCount.buffer, 1)).toEqual([0]);
  } finally {
    rig.destroy();
  }
});

it('composes padded reprojection outputs at nonzero offsets into a spectral producer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const grid = makeGPURasterMetadata({width: 3, height: 1});
  const pooledValues = rig.output('float32', 6);
  const pooledValidity = rig.output('uint32', 7);
  const values = rig.graph.createDataView(pooledValues.view.buffer, {
    format: 'float32',
    byteOffset: 4,
    length: 5
  });
  const validity = rig.graph.createDataView(pooledValidity.view.buffer, {
    format: 'uint32',
    byteOffset: 8,
    length: 5
  });
  const reprojection = new GPURasterReprojection({
    sourceGrid: grid,
    destinationGrid: grid,
    parameters: rig.input(
      getGPURasterReprojectionParameterValues({sourceGrid: grid, destinationGrid: grid}),
      'float32',
      20
    ),
    input: makeGPURasterBand(rig.input(Float32Array.of(4, 8, 0), 'float32', 3), grid, {
      validity: rig.input(Uint32Array.of(1, 1, 0), 'uint32', 3)
    }),
    output: {values, validity}
  });
  try {
    const output = rig.output('float32', 3);
    const outputValidity = rig.output('uint32', 3);
    rig.run(
      reprojection,
      new GPURasterNBR({
        width: 3,
        height: 1,
        nearInfrared: makeGPURasterBand(values, grid, {validity}),
        shortwaveInfrared2: makeGPURasterBand(
          rig.input(Float32Array.of(1, 2, 3), 'float32', 3),
          grid
        ),
        output: output.view,
        outputValidity: outputValidity.view
      })
    );
    const actual = await readFloat32(output.buffer, 3);
    expect(actual[0]).toBeCloseTo(0.6, 6);
    expect(actual[1]).toBeCloseTo(0.6, 6);
    expect(Number.isNaN(actual[2])).toBe(true);
    expect(await readUint32(outputValidity.buffer, 3)).toEqual([1, 1, 0]);
  } finally {
    rig.destroy();
    reprojection.destroy();
  }
});

it('reprojects read-only values and validity imported separately from one physical buffer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const grid = makeGPURasterMetadata({width: 1, height: 1});
  const data = new Float32Array(128);
  data[0] = 7;
  new Uint32Array(data.buffer)[64] = 1;
  const source = rig.input(data, 'float32', data.length);
  const physical = rig.bufferOf(source);
  const second = rig.graph.importBuffer(
    {id: 'read-only-validity-alias', byteLength: physical.byteLength, usage: physical.usage},
    physical
  );
  const output = rig.output('float32', 1);
  const validity = rig.output('uint32', 1);
  const contributor = new GPURasterReprojection({
    sourceGrid: grid,
    destinationGrid: grid,
    input: {
      id: 'read-only-source',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: rig.graph.createDataView(source.buffer, {format: 'float32', length: 1})
      },
      validity: rig.graph.createDataView(second, {format: 'uint32', byteOffset: 256, length: 1})
    },
    parameters: rig.input(
      getGPURasterReprojectionParameterValues({sourceGrid: grid, destinationGrid: grid}),
      'float32',
      20
    ),
    output: {values: output.view, validity: validity.view}
  });
  try {
    rig.run(contributor);
    expect(await readFloat32(output.buffer, 1)).toEqual([7]);
    expect(await readUint32(validity.buffer, 1)).toEqual([1]);
  } finally {
    rig.destroy();
    contributor.destroy();
  }
});

it('composes HANTS and temporal resampling using aligned pooled outputs and a one-chunk adapter', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const pool = rig.output('float32', 192);
    const smoothed = rig.graph.createDataView(pool.view.buffer, {format: 'float32', length: 8});
    const coefficients = rig.graph.createDataView(pool.view.buffer, {
      format: 'float32',
      byteOffset: 256,
      length: 3
    });
    const status = rig.graph.createDataView(pool.view.buffer, {
      format: 'uint32',
      byteOffset: 512,
      length: 1
    });
    const hants = new GPURasterHANTS({
      cellCount: 1,
      sliceCount: 8,
      harmonicCount: 1,
      values: createVectorView('source', 'float32', [
        rig.input(new Float32Array(8).fill(2), 'float32', 8)
      ]),
      parameters: rig.input(Float32Array.of(8, 1, 0), 'float32', 3),
      output: {smoothed, coefficients, status}
    });
    const resampled = rig.output('float32', 65);
    const resampledValues = rig.graph.createDataView(resampled.view.buffer, {
      format: 'float32',
      length: 1
    });
    const resampledStatus = rig.graph.createDataView(resampled.view.buffer, {
      format: 'uint32',
      byteOffset: 256,
      length: 1
    });
    const resample = new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 8,
      gridCount: 1,
      values: createVectorView('fitted', 'float32', [smoothed]),
      parameters: rig.input(Float32Array.of(0, 1, 1), 'float32', 3),
      dateAxis: {
        acquisitionTimes: rig.input(
          Uint32Array.from({length: 16}, (_, index) => (index % 2 === 0 ? index / 2 : 0)),
          'uint32x2',
          8
        ),
        temporalContext: {
          encoding: 'int64-words',
          unit: 'second',
          epoch: 'unix',
          axis: {kind: 'irregular'}
        }
      },
      output: {values: resampledValues, status: resampledStatus}
    });
    rig.run(hants, resample);
    const fitted = await readFloat32(pool.buffer, 192);
    expect(fitted.slice(0, 8)).toHaveLength(8);
    for (const value of fitted.slice(0, 8)) expect(value).toBeCloseTo(2, 5);
    expect((await readUint32(pool.buffer, 192))[128]).toBe(0);
    expect((await readFloat32(resampled.buffer, 65))[0]).toBeCloseTo(2, 5);
    expect((await readUint32(resampled.buffer, 65))[64]).toBe(0);
  } finally {
    rig.destroy();
  }
});
