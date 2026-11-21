// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it} from 'vitest';
import {
  GPUGeometryMeasures,
  type GPUGeometryMeasuresOutput
} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures';
import {GPUGeodesicPairs} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs';
import {computeGeodesicPairsOnCPU} from '../../../src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation';
import type {GPUSpatialFrame} from '../../../src/analysis-core/contracts/spatial-context';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';

useCPUExecutorTestHooks();
const device = createNullWebGPUDevice();
afterAll(() => device.destroy());

it('rejects known angular planar measurement while retaining coordinate-only columns', () => {
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 2);
  const lineOffsets = createTransientView(graph, 'offsets', 'uint32', 2);
  const measurement = createTransientView(graph, 'measurement', 'float32', 1);
  for (const frame of [
    {id: 'registry', kind: 'crs', crs: 'EPSG:4326'},
    {id: 'explicit', kind: 'crs', unit: {name: 'degrees', degreesPerUnit: 1}}
  ] satisfies GPUSpatialFrame[]) {
    const spatialContext = {coordinateSpace: 'planar', metric: 'none', frame} as const;
    for (const output of [{lengths: measurement}, {areas: measurement}]) {
      expect(
        () =>
          new GPUGeometryMeasures({
            geometryType: 'lines',
            positions,
            lineOffsets,
            spatialContext,
            output
          })
      ).toThrow(/angular.*project.*planar linear frame/);
    }
    expect(
      () =>
        new GPUGeometryMeasures({
          geometryType: 'lines',
          positions,
          lineOffsets,
          spatialContext,
          output: {bounds: createTransientView(graph, frame.id, 'float32x4', 1)}
        })
    ).not.toThrow();
  }
});

it('reads the inherited lengths destination once and ignores prototype methods', () => {
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const positions = createTransientView(graph, 'positions', 'float32x2', 2);
  const lineOffsets = createTransientView(graph, 'offsets', 'uint32', 2);
  const lengths = createTransientView(graph, 'lengths', 'float32', 1);
  let reads = 0;
  class Destinations implements GPUGeometryMeasuresOutput {
    get lengths() {
      reads++;
      return lengths;
    }
    toJSON() {
      return {};
    }
  }
  memory.write(positions, Float32Array.of(0, 0, 2, 0));
  memory.write(lineOffsets, Uint32Array.of(0, 2));
  const contributor = new GPUGeometryMeasures({
    geometryType: 'lines',
    positions,
    lineOffsets,
    output: new Destinations()
  });
  executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
  expect(reads).toBe(1);
  expect(Array.from(memory.readFloat32(lengths))).toEqual([2]);
  const unknown = Object.create({signedAreas: lengths}) as GPUGeometryMeasuresOutput;
  expect(
    () => new GPUGeometryMeasures({geometryType: 'lines', positions, lineOffsets, output: unknown})
  ).toThrow(/signedAreas/);
});

// Independent meridional cutoff: the two stored latitudes times pi/180 are
// 0.019999979843629 and 0.020000013133143 radians. Sphere length is
// R*(pi-lat*pi/180), R=6371008.8; pyproj Geod(ellps='WGS84').inv gives
// 19877222 after f32 publication outside the cutoff. No twin supplies expectations.
for (const budget of [2, 3, 32]) {
  for (const [latitude, converged, approximation, expected] of [
    [1.1459144353866577, 0, 1, 19887694],
    [1.1459163427352905, 1, 0, 19877222]
  ]) {
    it(`selects the meridional antipodal model from stored latitude ${latitude} at budget ${budget}`, () => {
      const origins = Float32Array.of(0, 0);
      const targets = Float32Array.of(180, latitude);
      const spatialContext = {
        coordinateSpace: 'longitude-latitude',
        metric: 'ellipsoidal'
      } as const;
      const result = computeGeodesicPairsOnCPU({origins, targets}, spatialContext, budget);
      expect(Array.from(result.converged)).toEqual([converged]);
      expect(Array.from(result.approximation)).toEqual([approximation]);
      expect(Array.from(result.iterationLimitReached)).toEqual([0]);
      expect(Array.from(result.distances)).toEqual([expected]);
      const graph = new GPUCommandGraph(device);
      const memory = createCPUGraphMemory();
      const originView = createTransientView(graph, 'origins', 'float32x2', 1);
      const targetView = createTransientView(graph, 'targets', 'float32x2', 1);
      const distances = createTransientView(graph, 'distances', 'float32', 1);
      const status = {
        converged: createTransientView(graph, 'converged', 'uint32', 1),
        approximation: createTransientView(graph, 'approximation', 'uint32', 1),
        iterationLimitReached: createTransientView(graph, 'limit', 'uint32', 1)
      };
      memory.write(originView, origins);
      memory.write(targetView, targets);
      executeCommandNodesOnCPU(
        new GPUGeodesicPairs({
          origins: originView,
          targets: targetView,
          spatialContext,
          maximumIterations: budget,
          output: {distances, status}
        }).getCommandNodes(graph),
        memory
      );
      expect(Array.from(memory.readUint32(status.converged))).toEqual([converged]);
      expect(Array.from(memory.readUint32(status.approximation))).toEqual([approximation]);
      expect(Array.from(memory.readUint32(status.iterationLimitReached))).toEqual([0]);
      const actualDistances = memory.readFloat32(distances);
      expect(actualDistances).toHaveLength(1);
      const distance = actualDistances[0];
      expect(Number.isFinite(distance)).toBe(true);
      expect(Math.abs(distance - expected)).toBeLessThan(12);
    });
  }
}

it('preserves the independent two versus three iteration control', () => {
  for (const budget of [2, 3]) {
    const result = computeGeodesicPairsOnCPU(
      {origins: Float32Array.of(0, 20), targets: Float32Array.of(170, 60)},
      {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
      budget
    );
    expect(Array.from(result.converged)).toEqual([budget === 3 ? 1 : 0]);
    expect(Array.from(result.approximation)).toEqual([budget === 3 ? 0 : 1]);
    expect(Array.from(result.iterationLimitReached)).toEqual([budget === 3 ? 0 : 1]);
    expect(Array.from(result.distances)).toEqual([budget === 3 ? 11091265 : 11073359]);
  }
});

it('B1 retains the stored equatorial longitude supplement at the antipodal cutoff', () => {
  // Exact equatorial gap = stored longitude*pi/180 = 0.019999994407791207.
  // Sphere R*(pi-gap), R=6371008.8, rounds to 19887694 m.
  const origins = Float32Array.of(1.1459152698516846, 0);
  const targets = Float32Array.of(180, 0);
  const spatialContext = {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'} as const;
  const cpu = computeGeodesicPairsOnCPU({origins, targets}, spatialContext, 32);
  expect(Array.from(cpu.converged)).toEqual([0]);
  expect(Array.from(cpu.approximation)).toEqual([1]);
  expect(Array.from(cpu.iterationLimitReached)).toEqual([0]);
  expect(Array.from(cpu.distances)).toEqual([19887694]);
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const originView = createTransientView(graph, 'equatorial-origin', 'float32x2', 1);
  const targetView = createTransientView(graph, 'equatorial-target', 'float32x2', 1);
  const distances = createTransientView(graph, 'equatorial-distance', 'float32', 1);
  const status = {
    converged: createTransientView(graph, 'equatorial-converged', 'uint32', 1),
    approximation: createTransientView(graph, 'equatorial-approximation', 'uint32', 1),
    iterationLimitReached: createTransientView(graph, 'equatorial-limit', 'uint32', 1)
  };
  memory.write(originView, origins);
  memory.write(targetView, targets);
  executeCommandNodesOnCPU(
    new GPUGeodesicPairs({
      origins: originView,
      targets: targetView,
      spatialContext,
      maximumIterations: 32,
      output: {distances, status}
    }).getCommandNodes(graph),
    memory
  );
  expect(Array.from(memory.readUint32(status.converged))).toEqual([0]);
  expect(Array.from(memory.readUint32(status.approximation))).toEqual([1]);
  expect(Array.from(memory.readUint32(status.iterationLimitReached))).toEqual([0]);
  const actual = memory.readFloat32(distances);
  expect(actual).toHaveLength(1);
  expect(Number.isFinite(actual[0])).toBe(true);
  expect(Math.abs(actual[0] - 19887694)).toBeLessThan(12);
});

it('B5 registry angular units survive a name-only explicit unit', () => {
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'partial-unit-positions', 'float32x2', 2);
  const lineOffsets = createTransientView(graph, 'partial-unit-offsets', 'uint32', 2);
  const spatialContext = {
    coordinateSpace: 'planar',
    metric: 'none',
    frame: {id: 'partial-unit', kind: 'crs', crs: 'EPSG:4326', unit: {name: 'degrees'}}
  } as const;
  expect(
    () =>
      new GPUGeometryMeasures({
        geometryType: 'lines',
        positions,
        lineOffsets,
        spatialContext,
        output: {lengths: createTransientView(graph, 'partial-unit-length', 'float32', 1)}
      })
  ).toThrow(/angular.*project/);
  expect(
    () =>
      new GPUGeometryMeasures({
        geometryType: 'lines',
        positions,
        lineOffsets,
        spatialContext,
        output: {bounds: createTransientView(graph, 'partial-unit-bounds', 'float32x4', 1)}
      })
  ).not.toThrow();
});

it('N1 permits adjacent status words only across the actual output dispatch split', () => {
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const origins = createTransientView(graph, 'alias-origins', 'float32x2', 1);
  const targets = createTransientView(graph, 'alias-targets', 'float32x2', 1);
  const words = createTransientView(graph, 'alias-words', 'uint32', 2);
  const converged = graph.createDataView(words.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 0
  });
  const iterationLimitReached = graph.createDataView(words.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  const approximation = createTransientView(graph, 'alias-approximation', 'uint32', 1);
  const output = {
    distances: createTransientView(graph, 'alias-distance', 'float32', 1),
    initialBearings: createTransientView(graph, 'alias-initial', 'float32', 1),
    finalBearings: createTransientView(graph, 'alias-final', 'float32', 1),
    midpoints: createTransientView(graph, 'alias-midpoint', 'float32x2', 1),
    status: {converged, approximation, iterationLimitReached}
  };
  const props = {
    origins,
    targets,
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'} as const,
    maximumIterations: 1,
    output
  };
  memory.write(origins, Float32Array.of(0, 0));
  memory.write(targets, Float32Array.of(1, 0));
  const contributor = new GPUGeodesicPairs(props);
  executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
  expect(Array.from(memory.readUint32(words))).toEqual([0, 1]);
  expect(
    () =>
      new GPUGeodesicPairs({
        ...props,
        output: {...output, status: {...output.status, iterationLimitReached: converged}}
      })
  ).toThrow(/overlap/);
  expect(
    () =>
      new GPUGeodesicPairs({
        ...props,
        output: {
          ...output,
          status: {
            ...output.status,
            approximation: iterationLimitReached,
            iterationLimitReached: approximation
          }
        }
      })
  ).toThrow(/overlap/);
});
