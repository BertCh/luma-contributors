// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUGeodesicPairs} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs';
import {GPUGeodesicPairsOperation} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs-operation';
import {CPUGeodesicPairsOperation} from '../../../src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

for (const budget of [2, 3, 32]) {
  it(`KI-oracle-cells-tessellation-R2 resolves stored-coordinate cutoff and iteration budget ${budget}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const spatialContext = {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'} as const;
    const origins = Float32Array.of(0, 0, 0, 0, 0, 20);
    const targets = Float32Array.of(180, 1.1459144353866577, 180, 1.1459163427352905, 170, 60);
    // Meridional gap is |target latitude|*pi/180, on either side of .02.
    // Independent sphere closed forms and PROJ inverse values are in final-wave.node.spec.ts.
    const expectedConverged = [0, 1, budget >= 3 ? 1 : 0];
    const expectedApproximation = [1, 0, budget >= 3 ? 0 : 1];
    const expectedLimit = [0, 0, budget >= 3 ? 0 : 1];
    const expectedDistances = [19887694, 19877222, budget >= 3 ? 11091265 : 11073359];
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const distances = output('distances', 'float32', 3);
    const status = {
      converged: output('converged', 'uint32', 3),
      approximation: output('approximation', 'uint32', 3),
      iterationLimitReached: output('limit', 'uint32', 3)
    };
    graph.add(
      new GPUGeodesicPairs({
        origins: input('origins', 'float32x2', origins),
        targets: input('targets', 'float32x2', targets),
        spatialContext,
        maximumIterations: budget,
        output: {distances, status}
      })
    );
    let compiled: ReturnType<typeof graph.compile> | undefined;
    try {
      compiled = graph.compile();
      const result = await fixture.submitTwice(compiled, async () => ({
        distances: await fixture.read(distances),
        converged: await fixture.read(status.converged),
        approximation: await fixture.read(status.approximation),
        limit: await fixture.read(status.iterationLimitReached)
      }));
      expect(result.converged).toEqual(expectedConverged);
      expect(result.approximation).toEqual(expectedApproximation);
      expect(result.limit).toEqual(expectedLimit);
      expect(result.distances).toHaveLength(3);
      result.distances.forEach((distance, index) => {
        expect(Number.isFinite(distance)).toBe(true);
        expect(Math.abs(distance - expectedDistances[index])).toBeLessThan(12);
      });
    } finally {
      compiled?.destroy();
      fixture.destroy();
    }
    for (const Operation of [GPUGeodesicPairsOperation, CPUGeodesicPairsOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          origins: {format: 'float32x2', values: origins},
          targets: {format: 'float32x2', values: targets}
        },
        operations: [
          new Operation({
            origins: 'origins',
            targets: 'targets',
            pairCount: 3,
            spatialContext,
            maximumIterations: budget,
            output: {
              distances: 'distances',
              status: {
                converged: 'converged',
                approximation: 'approximation',
                iterationLimitReached: 'limit'
              }
            }
          })
        ],
        outputs: ['distances', 'converged', 'approximation', 'limit']
      });
      expect(Array.from(result['converged'])).toEqual(expectedConverged);
      expect(Array.from(result['approximation'])).toEqual(expectedApproximation);
      expect(Array.from(result['limit'])).toEqual(expectedLimit);
      expect(result['distances']).toHaveLength(3);
      Array.from(result['distances']).forEach((distance, index) => {
        expect(Number.isFinite(distance)).toBe(true);
        expect(Math.abs(distance - expectedDistances[index])).toBeLessThan(12);
      });
    }
  });
}

it('B1 equatorial stored longitude supplement selects spherical fallback on every form', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const origins = Float32Array.of(1.1459152698516846, 0),
    targets = Float32Array.of(180, 0);
  const spatialContext = {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'} as const;
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const distances = output('distances', 'float32', 1),
    converged = output('converged', 'uint32', 1),
    approximation = output('approximation', 'uint32', 1),
    limit = output('limit', 'uint32', 1);
  graph.add(
    new GPUGeodesicPairs({
      origins: input('origins', 'float32x2', origins),
      targets: input('targets', 'float32x2', targets),
      spatialContext,
      maximumIterations: 32,
      output: {distances, status: {converged, approximation, iterationLimitReached: limit}}
    })
  );
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    compiled = graph.compile();
    const result = await fixture.submitTwice(compiled, async () => ({
      distances: await fixture.read(distances),
      converged: await fixture.read(converged),
      approximation: await fixture.read(approximation),
      limit: await fixture.read(limit)
    }));
    expect(result.distances).toHaveLength(1);
    expect(Number.isFinite(result.distances[0])).toBe(true);
    expect(Math.abs(result.distances[0] - 19887694)).toBeLessThan(12);
    expect(result.converged).toEqual([0]);
    expect(result.approximation).toEqual([1]);
    expect(result.limit).toEqual([0]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
  for (const Operation of [GPUGeodesicPairsOperation, CPUGeodesicPairsOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        origins: {format: 'float32x2', values: origins},
        targets: {format: 'float32x2', values: targets}
      },
      operations: [
        new Operation({
          origins: 'origins',
          targets: 'targets',
          spatialContext,
          pairCount: 1,
          maximumIterations: 32,
          output: {
            distances: 'distances',
            status: {
              converged: 'converged',
              approximation: 'approximation',
              iterationLimitReached: 'limit'
            }
          }
        })
      ],
      outputs: ['distances', 'converged', 'approximation', 'limit']
    });
    expect(result['distances']).toHaveLength(1);
    expect(Number.isFinite(result['distances'][0])).toBe(true);
    expect(Math.abs(result['distances'][0] - 19887694)).toBeLessThan(12);
    expect(Array.from(result['converged'])).toEqual([0]);
    expect(Array.from(result['approximation'])).toEqual([1]);
    expect(Array.from(result['limit'])).toEqual([0]);
  }
});

it('N1 writes adjacent logically disjoint status words in different dispatches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const words = output('words', 'uint32', 2);
  const converged = graph.createDataView(words.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: words.byteOffset
  });
  const limit = graph.createDataView(words.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: words.byteOffset + 4
  });
  graph.add(
    new GPUGeodesicPairs({
      origins: input('origins', 'float32x2', Float32Array.of(0, 0)),
      targets: input('targets', 'float32x2', Float32Array.of(1, 0)),
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
      maximumIterations: 1,
      output: {
        distances: output('distances', 'float32', 1),
        initialBearings: output('initial', 'float32', 1),
        finalBearings: output('final', 'float32', 1),
        midpoints: output('midpoint', 'float32x2', 1),
        status: {
          converged,
          approximation: output('approximation', 'uint32', 1),
          iterationLimitReached: limit
        }
      }
    })
  );
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    compiled = graph.compile();
    expect(await fixture.submitTwice(compiled, () => fixture.read(words))).toEqual([0, 1]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});
