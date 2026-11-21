// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUDistanceField,
  getGPUDistanceFieldParameterValues
} from '../../../src/raster-analysis/distance-field';
import {
  GPURasterSampling,
  GPURasterProfile,
  getGPURasterSamplingParameterValues,
  getGPURasterProfileParameterValues
} from '../../../src/raster-analysis/raster-sampling';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

for (const fixture of [
  {
    name: 'negative side of zero',
    width: 2,
    step: 1,
    origin: [-1, 0] as const,
    position: [-(2 ** -25), -0.5],
    cell: 0,
    value: 10
  },
  {
    name: 'exact zero tie',
    width: 2,
    step: 1,
    origin: [-1, 0] as const,
    position: [0, -0.5],
    cell: 1,
    value: 20
  },
  {
    name: 'positive side of zero',
    width: 2,
    step: 1,
    origin: [-1, 0] as const,
    position: [2 ** -25, -0.5],
    cell: 1,
    value: 20
  },
  {name: 'outermost edge', width: 2, step: 41, position: [82, -20.5], cell: -1, value: 20},
  {name: '41 m column edge', width: 2, step: 41, position: [41, -20.5], cell: 1, value: 20},
  {name: '.85 m column edge', width: 21, step: 0.85, position: [17, -0.425], cell: 20, value: 210},
  {name: 'negative row step', width: 2, step: 41, position: [20.5, -41], cell: 2, value: 1010}
])
  it(`FLT-02 agrees across seeding, sampling and profiles at ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {width, step} = fixture;
    const localOrigin = 'origin' in fixture ? ([0, 0] as const) : ([10000000, 10000000] as const);
    const grid = makeGPURasterMetadata({
      width,
      height: 2,
      cellSize: [step, step],
      origin: 'origin' in fixture ? fixture.origin : localOrigin
    });
    const values = Float32Array.from(
      {length: width * 2},
      (_, index) => Math.floor(index / width) * 1000 + (index % width) * 10 + 10
    );
    const graph = new GPUCommandGraph(device);
    const raster = createInputBuffer(device, values);
    const positions = createInputBuffer(device, Float32Array.from(fixture.position));
    const offsets = createInputBuffer(device, Uint32Array.of(0, 1));
    const distanceParameters = createInputBuffer(
      device,
      getGPUDistanceFieldParameterValues({localOrigin}, grid)
    );
    const samplingParameters = createInputBuffer(
      device,
      getGPURasterSamplingParameterValues({localOrigin, method: 'nearest'}, grid)
    );
    const profileParameters = createInputBuffer(
      device,
      getGPURasterProfileParameterValues({localOrigin, method: 'nearest', spacing: 1}, grid)
    );
    const distances = createOutputBuffer(device, values.length);
    const samples = createOutputBuffer(device, 1);
    const profileSamples = createOutputBuffer(device, 1);
    const count = createOutputBuffer(device, 1);
    const overflow = createOutputBuffer(device, 1);
    const rasterView = importGraphBuffer(graph, 'raster', raster, 'float32', values.length);
    const positionView = importGraphBuffer(graph, 'positions', positions, 'float32x2', 1);
    graph.add(
      new GPUDistanceField({
        grid,
        seedPositions: positionView,
        parameters: importGraphBuffer(
          graph,
          'distance-parameters',
          distanceParameters,
          'float32',
          8
        ),
        output: {
          distances: importGraphBuffer(graph, 'distances', distances, 'float32', values.length)
        }
      })
    );
    graph.add(
      new GPURasterSampling({
        grid,
        values: rasterView,
        positions: positionView,
        parameters: importGraphBuffer(
          graph,
          'sampling-parameters',
          samplingParameters,
          'float32',
          12
        ),
        output: {values: importGraphBuffer(graph, 'samples', samples, 'float32', 1)}
      })
    );
    graph.add(
      new GPURasterProfile({
        grid,
        values: rasterView,
        positions: positionView,
        lineOffsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', 2),
        parameters: importGraphBuffer(
          graph,
          'profile-parameters',
          profileParameters,
          'float32',
          12
        ),
        output: {
          sampleValues: importGraphBuffer(graph, 'profile-samples', profileSamples, 'float32', 1),
          status: {
            count: importGraphBuffer(graph, 'count', count, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
          }
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      const distanceValues = await readFloat32(distances);
      expect(
        distanceValues.every(value =>
          fixture.cell < 0 ? value === Infinity : Number.isFinite(value)
        )
      ).toBe(true);
      expect(distanceValues).toHaveLength(width * 2);
      expect(distanceValues.flatMap((value, index) => (value === 0 ? [index] : []))).toEqual(
        fixture.cell < 0 ? [] : [fixture.cell]
      );
      // Half-open cell edges: [41,82) owns 41, [.85*20,.85*21) owns 17;
      // row 1 begins at -41 when the signed row step is -41.
      expect(await readFloat32(samples)).toEqual([fixture.value]);
      expect(await readFloat32(profileSamples)).toEqual([fixture.value]);
      expect(await readUint32(count)).toEqual([1]);
    } finally {
      compiled.destroy();
      for (const buffer of [
        raster,
        positions,
        offsets,
        distanceParameters,
        samplingParameters,
        profileParameters,
        distances,
        samples,
        profileSamples,
        count,
        overflow
      ])
        buffer.destroy();
    }
  });

it('WRT-07 reuses a three-row allocation at capacity one with a one-row readable prefix', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 3, height: 1, origin: [0, 1]});
  const raster = createInputBuffer(device, Float32Array.of(10, 20, 30));
  const positions = createInputBuffer(device, Float32Array.of(0.5, 0.5, 2.5, 0.5));
  const offsets = createInputBuffer(device, Uint32Array.of(0, 2));
  const parameters = createInputBuffer(
    device,
    getGPURasterProfileParameterValues({spacing: 1, method: 'nearest'}, grid)
  );
  const samples = createOutputBuffer(device, 3);
  const count = createOutputBuffer(device, 1);
  const overflow = createOutputBuffer(device, 1);
  try {
    for (const sampleCapacity of [3, 1]) {
      if (sampleCapacity === 1) raster.write(Float32Array.of(40, 50, 60));
      const graph = new GPUCommandGraph(device);
      graph.add(
        new GPURasterProfile({
          grid,
          sampleCapacity,
          values: importGraphBuffer(graph, 'raster', raster, 'float32', 3),
          positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 2),
          lineOffsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', 2),
          parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 12),
          output: {
            sampleValues: importGraphBuffer(graph, 'samples', samples, 'float32', 3),
            status: {
              count: importGraphBuffer(graph, 'count', count, 'uint32', 1),
              overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
            }
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled);
        const liveCount = (await readUint32(count))[0];
        const sampleValues = Float32Array.from(await readFloat32(samples));
        expect(liveCount).toBe(sampleCapacity);
        expect(Array.from(sampleValues.subarray(0, liveCount))).toEqual(
          sampleCapacity === 3 ? [10, 20, 30] : [40]
        );
        expect(Array.from(sampleValues)).toEqual(
          sampleCapacity === 3 ? [10, 20, 30] : [40, 20, 30]
        );
        expect(await readUint32(overflow)).toEqual([sampleCapacity === 3 ? 0 : 1]);
      } finally {
        compiled.destroy();
      }
    }
  } finally {
    for (const buffer of [raster, positions, offsets, parameters, samples, count, overflow])
      buffer.destroy();
  }
});

it('WRT-11 status-only geometry counts agree for finite and NaN rasters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 1, height: 1});
  const graph = new GPUCommandGraph(device);
  const raster = createInputBuffer(device, Float32Array.of(10));
  const positions = createInputBuffer(device, Float32Array.of(0, 0, 2, 0));
  const offsets = createInputBuffer(device, Uint32Array.of(0, 2));
  const parameters = createInputBuffer(
    device,
    getGPURasterProfileParameterValues({spacing: 1}, grid)
  );
  const count = createOutputBuffer(device, 1);
  const overflow = createOutputBuffer(device, 1);
  const requiredCount = createOutputBuffer(device, 1);
  const profile = new GPURasterProfile({
    grid,
    sampleCapacity: 1,
    values: importGraphBuffer(graph, 'raster', raster, 'float32', 1),
    positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 2),
    lineOffsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', 2),
    parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 12),
    output: {
      status: {
        count: importGraphBuffer(graph, 'count', count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
        requiredCount: importGraphBuffer(graph, 'required', requiredCount, 'uint32', 1)
      }
    }
  });
  const nodes = profile.getCommandNodes(graph);
  expect(nodes.map(node => node.id)).not.toContain('raster-profile-sample');
  for (const node of nodes) graph.add(node);
  const compiled = graph.compile();
  try {
    for (const value of [10, NaN]) {
      raster.write(Float32Array.of(value));
      submitGraph(device, compiled);
      expect(await readUint32(count)).toEqual([1]);
      expect(await readUint32(overflow)).toEqual([1]);
      // length two at spacing one includes 0, 1, 2, independent of elevation.
      expect(await readUint32(requiredCount)).toEqual([3]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of [raster, positions, offsets, parameters, count, overflow, requiredCount])
      buffer.destroy();
  }
});
