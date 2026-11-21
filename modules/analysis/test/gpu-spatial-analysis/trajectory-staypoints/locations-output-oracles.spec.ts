// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTrajectoryLocations} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations';
import {GPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations-operation';
import {CPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {poisonOperation} from '../trajectory-cleaning/trajectory-output-oracle-utils';
import {GraphHarness} from './movement-spec-utils';

it('parallel locations compact first visits with empty active input and unwritten capacity tails poisoned', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two rounds settle the six-row fixture into components {0,1,2,3,4} and {100}.
  // Stable first visits give means [2,100]; the final two input rows remain inactive.
  for (const active of [0, 3, 6])
    for (const Operation of [GPUTrajectoryLocationsOperation, CPUTrajectoryLocationsOperation]) {
      const operation = new Operation({
        centroids: 'points',
        tracks: 'tracks',
        count: 'active',
        parameters: 'parameters',
        capacity: 8,
        maximumIterations: 2,
        output: {
          locationIds: 'ids',
          centers: 'centers',
          status: {count: 'count', iterationLimitReached: 'limit'}
        }
      });
      const result = await runOperationGraph(device, {
        inputs: {
          points: {
            format: 'float32x2',
            values: Float32Array.of(0, 0, 1, 0, 2, 0, 100, 0, 3, 0, 4, 0, 900, 0, 901, 0)
          },
          tracks: {format: 'uint32', values: new Uint32Array(8)},
          active: {format: 'uint32', values: Uint32Array.of(active)},
          parameters: {format: 'float32', values: Float32Array.of(1, 0, 0, 0)}
        },
        operations: [
          operation instanceof GPUTrajectoryLocationsOperation
            ? poisonOperation(operation)
            : operation
        ],
        outputs: ['ids', 'centers', 'count', 'limit']
      });
      const expectedIds = active === 0 ? [] : active === 3 ? [0, 0, 0] : [0, 0, 0, 1, 0, 0];
      const count = active === 0 ? 0 : active === 3 ? 1 : 2;
      expect(Array.from(result['ids'])).toEqual([
        ...expectedIds,
        ...new Array(8 - active).fill(0xffffffff)
      ]);
      expect(Array.from(result['count'])).toEqual([count]);
      expect(Array.from(result['centers']).slice(0, count * 2)).toEqual(
        active === 0 ? [] : active === 3 ? [1, 0] : [2, 0, 100, 0]
      );
      expect(Array.from(result['limit'])).toEqual([0]);
    }
});

it('locations reports executed rounds with and without centers or convergence outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chain of [false, true])
    for (const withCenters of [false, true]) {
      const harness = new GraphHarness(device, `locations-rounds-${chain}-${withCenters}`);
      const capacity = chain ? 5 : 1;
      const ids = harness.output('ids', 'uint32', capacity);
      const count = harness.output('count', 'uint32', 1);
      const rounds = harness.output('rounds', 'uint32', 1);
      rounds.buffer.write(Uint32Array.of(93));
      const centers = withCenters ? harness.output('centers', 'float32x2', capacity) : undefined;
      centers?.buffer.write(new Float32Array(capacity * 2).fill(93));
      harness.graph.add(
        new GPUTrajectoryLocations({
          centroids: harness.input(
            'points',
            chain ? Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0, 4, 0) : Float32Array.of(7, 9),
            'float32x2',
            capacity
          ),
          tracks: harness.input('tracks', new Uint32Array(capacity), 'uint32', capacity),
          count: harness.input('active', Uint32Array.of(capacity), 'uint32', 1),
          parameters: harness.input('parameters', Float32Array.of(1, 0, 0, 0), 'float32', 4),
          maximumIterations: chain ? 1 : 2,
          output: {
            locationIds: ids.view,
            status: {
              count: count.view,
              iterationLimitReached: harness.output('iterationLimitReached', 'uint32', 1).view,
              iterationCount: rounds.view
            },
            ...(centers ? {centers: centers.view} : {})
          }
        })
      );
      const result = await harness.run(async () => ({
        rounds: await readUint32(rounds.buffer, 1),
        ids: await readUint32(ids.buffer, capacity),
        centers: centers ? await readFloat32(centers.buffer, capacity * 2) : undefined
      }));
      // There is no early exit: one hook/compress pair per configured round, excluding the check.
      expect(result.rounds).toEqual(chain ? [1] : [2]);
      if (!chain) expect(result.ids).toEqual([0]);
      if (!chain && centers) expect(result.centers).toEqual([7, 9]);
      if (chain) expect(result.ids).toEqual(new Array(5).fill(0xffffffff));
      if (chain && centers) expect(result.centers).toEqual(new Array(capacity * 2).fill(93));
    }
});

it('R2 zero iteration budget publishes zero executed rounds', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'locations-zero-rounds');
  const ids = harness.output('ids', 'uint32', 1);
  const count = harness.output('count', 'uint32', 1);
  const rounds = harness.output('rounds', 'uint32', 1);
  rounds.buffer.write(Uint32Array.of(93));
  harness.graph.add(
    new GPUTrajectoryLocations({
      maximumIterations: 0,
      centroids: harness.input('points', Float32Array.of(7, 9), 'float32x2', 1),
      tracks: harness.input('tracks', Uint32Array.of(0), 'uint32', 1),
      count: harness.input('active', Uint32Array.of(1), 'uint32', 1),
      parameters: harness.input('parameters', Float32Array.of(1, 0, 0, 0), 'float32', 4),
      output: {
        locationIds: ids.view,
        status: {
          count: count.view,
          iterationLimitReached: harness.output('iterationLimitReached', 'uint32', 1).view,
          iterationCount: rounds.view
        }
      }
    })
  );
  expect(await harness.run(() => readUint32(rounds.buffer, 1))).toEqual([0]);
});

it('R3 iterationCount writes only its one-row offset destination', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'locations-offset-rounds');
  const ids = harness.output('ids', 'uint32', 1);
  const count = harness.output('count', 'uint32', 1);
  const rounds = harness.output('rounds', 'uint32', 3);
  rounds.buffer.write(Uint32Array.of(93, 93, 93));
  const offsetRounds = harness.graph.createDataView(rounds.view.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  harness.graph.add(
    new GPUTrajectoryLocations({
      maximumIterations: 2,
      centroids: harness.input('points', Float32Array.of(7, 9), 'float32x2', 1),
      tracks: harness.input('tracks', Uint32Array.of(0), 'uint32', 1),
      count: harness.input('active', Uint32Array.of(1), 'uint32', 1),
      parameters: harness.input('parameters', Float32Array.of(1, 0, 0, 0), 'float32', 4),
      output: {
        locationIds: ids.view,
        status: {
          count: count.view,
          iterationLimitReached: harness.output('iterationLimitReached', 'uint32', 1).view,
          iterationCount: offsetRounds
        }
      }
    })
  );
  expect(await harness.run(() => readUint32(rounds.buffer, 3))).toEqual([93, 2, 93]);
});

it('R4 both graph operations publish iterationCount as their sole optional status', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const maximumIterations of [2, undefined])
    for (const Operation of [GPUTrajectoryLocationsOperation, CPUTrajectoryLocationsOperation]) {
      const operation = new Operation({
        centroids: 'points',
        tracks: 'tracks',
        count: 'active',
        parameters: 'parameters',
        capacity: 1,
        maximumIterations,
        output: {
          locationIds: 'ids',
          status: {count: 'count', iterationLimitReached: 'limit', iterationCount: 'rounds'}
        }
      });
      const result = await runOperationGraph(device, {
        inputs: {
          points: {format: 'float32x2', values: Float32Array.of(7, 9)},
          tracks: {format: 'uint32', values: Uint32Array.of(0)},
          active: {format: 'uint32', values: Uint32Array.of(1)},
          parameters: {format: 'float32', values: Float32Array.of(1, 0, 0, 0)}
        },
        operations: [
          operation instanceof GPUTrajectoryLocationsOperation
            ? poisonOperation(operation)
            : operation
        ],
        outputs: ['ids', 'count', 'rounds', 'limit']
      });
      expect(Array.from(result['rounds'])).toEqual([maximumIterations ?? 32]);
      expect(Array.from(result['limit'])).toEqual([0]);
      expect(Array.from(result['ids'])).toEqual([0]);
      expect(Array.from(result['count'])).toEqual([1]);
    }
});

it('location operation stopping flags distinguish an exhausted one-round chain from two-round convergence', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const maximumIterations of [1, 2])
    for (const optional of ['converged', 'limit', 'none'] as const) {
      for (const Operation of [GPUTrajectoryLocationsOperation, CPUTrajectoryLocationsOperation]) {
        const operation = new Operation({
          centroids: 'points',
          tracks: 'tracks',
          count: 'active',
          parameters: 'parameters',
          capacity: 5,
          maximumIterations,
          output: {
            locationIds: 'ids',
            status: {
              count: 'count',
              ...(optional === 'converged' ? {converged: 'flag'} : {}),
              iterationLimitReached: optional === 'limit' ? 'flag' : 'limit'
            }
          }
        });
        const outputs = ['ids', 'count', ...(optional !== 'none' ? ['flag'] : [])];
        const result = await runOperationGraph(device, {
          inputs: {
            points: {format: 'float32x2', values: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0, 4, 0)},
            tracks: {format: 'uint32', values: new Uint32Array(5)},
            active: {format: 'uint32', values: Uint32Array.of(5)},
            parameters: {format: 'float32', values: Float32Array.of(1, 0, 0, 0)}
          },
          operations: [
            operation instanceof GPUTrajectoryLocationsOperation
              ? poisonOperation(operation)
              : operation
          ],
          outputs
        });
        // First round labels [0,0,0,1,2] can still change; the second makes every label 0.
        expect(Array.from(result['count'])).toEqual([maximumIterations === 1 ? 0 : 1]);
        expect(Array.from(result['ids'])).toEqual(
          maximumIterations === 1 ? new Array(5).fill(0xffffffff) : [0, 0, 0, 0, 0]
        );
        if (optional !== 'none')
          expect(Array.from(result['flag'])).toEqual([
            optional === 'converged'
              ? Number(maximumIterations === 2)
              : Number(maximumIterations === 1)
          ]);
      }
    }
});

it('R1 converged location IDs and centers stay aligned across tracks and repeated submissions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'locations-converged-repeat'});
  const buffers = {
    points: createInputBuffer(device, Float32Array.of(50, 0, 0, 0, 1, 0, 2, 0, 4, 0)),
    tracks: createInputBuffer(device, Uint32Array.of(0, 1, 1, 1, 1)),
    active: createInputBuffer(device, Uint32Array.of(5)),
    parameters: createInputBuffer(device, Float32Array.of(0.25, 0, 0, 0)),
    ids: createOutputBuffer(device, 5),
    centers: createOutputBuffer(device, 10),
    count: createOutputBuffer(device, 1),
    limit: createOutputBuffer(device, 1)
  };
  graph.add(
    new GPUTrajectoryLocations({
      maximumIterations: 2,
      centroids: importGraphBuffer(graph, 'points', buffers.points, 'float32x2', 5),
      tracks: importGraphBuffer(graph, 'tracks', buffers.tracks, 'uint32', 5),
      count: importGraphBuffer(graph, 'active', buffers.active, 'uint32', 1),
      parameters: importGraphBuffer(graph, 'parameters', buffers.parameters, 'float32', 4),
      output: {
        locationIds: importGraphBuffer(graph, 'ids', buffers.ids, 'uint32', 5),
        centers: importGraphBuffer(graph, 'centers', buffers.centers, 'float32x2', 5),
        status: {
          count: importGraphBuffer(graph, 'count', buffers.count, 'uint32', 1),
          iterationLimitReached: importGraphBuffer(graph, 'limit', buffers.limit, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const epsilon of [0.25, 1]) {
      buffers.parameters.write(Float32Array.of(epsilon, 0, 0, 0));
      buffers.ids.write(new Uint32Array(5).fill(93));
      buffers.centers.write(new Float32Array(10).fill(93));
      buffers.count.write(Uint32Array.of(93));
      buffers.limit.write(Uint32Array.of(93));
      submitGraph(device, compiled, undefined);
      const count = epsilon === 1 ? 3 : 5;
      expect(await readUint32(buffers.count, 1)).toEqual([count]);
      expect(await readUint32(buffers.ids, 5)).toEqual(
        epsilon === 1 ? [0, 1, 1, 1, 2] : [0, 1, 2, 3, 4]
      );
      // Exact binary32 coordinates and means: 50, (0+1+2)/3=1, and 4.
      expect(await readFloat32(buffers.centers, count * 2)).toEqual(
        epsilon === 1 ? [50, 0, 1, 0, 4, 0] : [50, 0, 0, 0, 1, 0, 2, 0, 4, 0]
      );
      expect(await readUint32(buffers.limit, 1)).toEqual([0]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of Object.values(buffers)) buffer.destroy();
  }
});

it('R1 graph operation publishes all converged two-track locations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const operation = new GPUTrajectoryLocationsOperation({
    centroids: 'points',
    tracks: 'tracks',
    count: 'active',
    parameters: 'parameters',
    capacity: 5,
    maximumIterations: 2,
    output: {
      locationIds: 'ids',
      centers: 'centers',
      status: {count: 'count', iterationLimitReached: 'limit'}
    }
  });
  const result = await runOperationGraph(device, {
    inputs: {
      points: {format: 'float32x2', values: Float32Array.of(50, 0, 0, 0, 1, 0, 2, 0, 4, 0)},
      tracks: {format: 'uint32', values: Uint32Array.of(0, 1, 1, 1, 1)},
      active: {format: 'uint32', values: Uint32Array.of(5)},
      parameters: {format: 'float32', values: Float32Array.of(1, 0, 0, 0)}
    },
    operations: [poisonOperation(operation)],
    outputs: ['ids', 'centers', 'count', 'limit']
  });
  expect(Array.from(result['ids'])).toEqual([0, 1, 1, 1, 2]);
  expect(Array.from(result['centers']).slice(0, 6)).toEqual([50, 0, 1, 0, 4, 0]);
  expect(Array.from(result['count'])).toEqual([3]);
  expect(Array.from(result['limit'])).toEqual([0]);
});

it('location operation poisoned means and stopping flags match distinct component arithmetic', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const optional of ['centers', 'converged', 'limit', 'none'] as const) {
    const operation = new GPUTrajectoryLocationsOperation({
      centroids: 'points',
      tracks: 'tracks',
      count: 'active',
      parameters: 'parameters',
      capacity: 7,
      output: {
        locationIds: 'ids',
        ...(optional === 'centers' ? {centers: 'centers'} : {}),
        status: {
          count: 'count',
          ...(optional === 'converged' ? {converged: 'flag'} : {}),
          iterationLimitReached: optional === 'limit' ? 'flag' : 'limit'
        }
      }
    });
    const result = await runOperationGraph(device, {
      inputs: {
        points: {
          format: 'float32x2',
          values: Float32Array.of(0, 0, 2, 0, 4, 0, 0, 0, 10, 0, 0, 0, 2, 0)
        },
        tracks: {format: 'uint32', values: Uint32Array.of(0, 0, 0, 0, 0, 1, 1)},
        active: {format: 'uint32', values: Uint32Array.of(7)},
        parameters: {format: 'float32', values: Float32Array.of(2, 0, 0, 0)}
      },
      operations: [poisonOperation(operation)],
      outputs: Object.values(operation.outputs).map(output => output.value)
    });
    expect(Array.from(result['ids'])).toEqual([0, 0, 0, 0, 1, 2, 2]);
    expect(Array.from(result['count'])).toEqual([3]);
    // Component centers average distinct coordinates: (0+2+4)/3, 10, and (0+2)/2.
    // Only the count prefix is defined; the capacity tail intentionally stays poisoned.
    if (optional === 'centers')
      expect(Array.from(result['centers']).slice(0, 6)).toEqual([2, 0, 10, 0, 1, 0]);
    if (optional === 'converged') expect(Array.from(result['flag'])).toEqual([1]);
    if (optional === 'limit') expect(Array.from(result['flag'])).toEqual([0]);
  }
});
