// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUTrajectoryLocations} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations';
import {GPUTrajectoryStaypointsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints-operation';
import {CPUTrajectoryStaypointsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {GPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations-operation';
import {CPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {GraphHarness} from './movement-spec-utils';

it('breaks observed dwell at a missing middle fix in both named forms', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUTrajectoryStaypointsOperation, CPUTrajectoryStaypointsOperation])
    for (const masked of [false, true]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {
            format: 'float32x2',
            values: Float32Array.of(0, 0, masked ? 0 : NaN, masked ? 0 : NaN, 1000, 0)
          },
          timestamps: {format: 'float32', values: Float32Array.of(0, 300000, 600000)},
          offsets: {format: 'uint32', values: Uint32Array.of(0, 3)},
          parameters: {format: 'float32', values: Float32Array.of(100, 300000, 900000, 1)},
          mask: {format: 'uint32', values: Uint32Array.of(1, masked ? 0 : 1, 1)}
        },
        operations: [
          new Operation({
            positions: 'positions',
            timestamps: 'timestamps',
            trackOffsets: 'offsets',
            parameters: 'parameters',
            mask: 'mask',
            missingFixPolicy: 'break',
            rowCount: 3,
            capacity: 3,
            output: {
              ids: 'ids',
              status: {
                count: 'count',
                overflow: 'overflow',
                candidateOverflow: 'candidate',
                missingFixCount: 'missing'
              }
            }
          })
        ],
        outputs: ['count', 'overflow', 'candidate', 'missing']
      });
      expect(Array.from(result['count'])).toEqual([0]);
      expect(Array.from(result['missing'])).toEqual([1]);
      expect(Array.from(result['overflow'])).toEqual([0]);
      expect(Array.from(result['candidate'])).toEqual([0]);
    }
});
it('publishes location completeness independently of convergence and recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUTrajectoryLocationsOperation, CPUTrajectoryLocationsOperation])
    for (const capacity of [1, 2])
      for (const flag of ['overflow', 'candidateOverflow'] as const)
        for (const loss of [1, 0]) {
          const result = await runOperationGraph(device, {
            inputs: {
              positions: {
                format: 'float32x2',
                values: Float32Array.of(0, 0, 10, 0).subarray(0, capacity * 2)
              },
              tracks: {format: 'uint32', values: Uint32Array.of(0, 1).subarray(0, capacity)},
              active: {format: 'uint32', values: Uint32Array.of(capacity)},
              parameters: {format: 'float32', values: Float32Array.of(1, 0, 0, 0)},
              loss: {format: 'uint32', values: Uint32Array.of(loss)}
            },
            operations: [
              new Operation({
                centroids: 'positions',
                tracks: 'tracks',
                count: 'active',
                parameters: 'parameters',
                capacity,
                staypointStatus: {[flag]: 'loss'},
                output: {
                  locationIds: 'ids',
                  status: {
                    count: 'count',
                    converged: 'converged',
                    complete: 'complete',
                    iterationLimitReached: 'limit'
                  }
                }
              })
            ],
            outputs: ['ids', 'count', 'converged', 'complete']
          });
          expect(Array.from(result['count'])).toEqual([loss ? 0 : capacity]);
          expect(Array.from(result['converged'])).toEqual([1]);
          expect(Array.from(result['complete'])).toEqual([1 - loss]);
          expect(Array.from(result['ids'])).toEqual(
            loss
              ? new Array(capacity).fill(0xffffffff)
              : Array.from({length: capacity}, (_, row) => row)
          );
        }
});
it('tests radius membership before rounded distance in the actual hook and convergence check', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [firstX, offset] of [
    [0, 0],
    [0, 2 ** -13],
    [0, 2 ** -26],
    [-(2 ** -25), 0]
  ]) {
    const harness = new GraphHarness(device, `radius-${offset}`);
    const ids = harness.output('ids', 'uint32', 2);
    const count = harness.output('count', 'uint32', 1);
    const converged = harness.output('converged', 'uint32', 1);
    harness.graph.add(
      new GPUTrajectoryLocations({
        centroids: harness.input(
          'centroids',
          Float32Array.of(firstX, 0, 1, offset),
          'float32x2',
          2
        ),
        tracks: harness.input('tracks', Uint32Array.of(0, 0), 'uint32', 2),
        count: harness.input('active', Uint32Array.of(2), 'uint32', 1),
        parameters: harness.input('parameters', Float32Array.of(1, 0, 0, 0), 'float32', 4),
        maximumIterations: 2,
        output: {
          locationIds: ids.view,
          status: {
            count: count.view,
            iterationLimitReached: harness.output('iterationLimitReached', 'uint32', 1).view,
            converged: converged.view
          }
        }
      })
    );
    await harness.run(async () => {
      // Exact squared distances: 1 and 1+2^-26. Equality is inclusive.
      expect(Array.from(await readUint32(ids.buffer, 2))).toEqual(
        offset === 0 && firstX === 0 ? [0, 0] : [0, 1]
      );
      expect(Array.from(await readUint32(count.buffer, 1))).toEqual([
        offset === 0 && firstX === 0 ? 1 : 2
      ]);
      expect(Array.from(await readUint32(converged.buffer, 1))).toEqual([1]);
    });
  }
});
it('review reuses one compiled location graph across loss and recovery frames', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacity of [1, 2])
    for (const flag of ['overflow', 'candidateOverflow'] as const) {
      const harness = new GraphHarness(device, `location-recovery-${capacity}-${flag}`);
      const loss = harness.output('loss', 'uint32', 1),
        count = harness.output('count', 'uint32', 1),
        complete = harness.output('complete', 'uint32', 1),
        converged = harness.output('converged', 'uint32', 1),
        ids = harness.output('ids', 'uint32', capacity);
      loss.buffer.write(Uint32Array.of(1));
      harness.graph.add(
        new GPUTrajectoryLocations({
          centroids: harness.input(
            'points',
            Float32Array.of(0, 0, 10, 0).subarray(0, capacity * 2),
            'float32x2',
            capacity
          ),
          tracks: harness.input(
            'tracks',
            Uint32Array.of(0, 1).subarray(0, capacity),
            'uint32',
            capacity
          ),
          count: harness.input('active', Uint32Array.of(capacity), 'uint32', 1),
          parameters: harness.input('parameters', Float32Array.of(1, 0, 0, 0), 'float32', 4),
          staypointStatus: {[flag]: loss.view},
          output: {
            locationIds: ids.view,
            status: {
              count: count.view,
              iterationLimitReached: harness.output('iterationLimitReached', 'uint32', 1).view,
              complete: complete.view,
              converged: converged.view
            }
          }
        })
      );
      await harness.run(async compiled => {
        for (const lost of [1, 0, 1, 0]) {
          loss.buffer.write(Uint32Array.of(lost));
          submitGraph(device, compiled, undefined);
          expect(Array.from(await readUint32(count.buffer, 1))).toEqual([lost ? 0 : capacity]);
          expect(Array.from(await readUint32(complete.buffer, 1))).toEqual([1 - lost]);
          expect(Array.from(await readUint32(converged.buffer, 1))).toEqual([1]);
          expect(Array.from(await readUint32(ids.buffer, capacity))).toEqual(
            lost
              ? new Array(capacity).fill(0xffffffff)
              : Array.from({length: capacity}, (_, row) => row)
          );
        }
      });
    }
});
it('review ignores unowned input padding in the GPU missing-fix count', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUTrajectoryStaypointsOperation, CPUTrajectoryStaypointsOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        points: {format: 'float32x2', values: new Float32Array(10)},
        times: {format: 'float32', values: Float32Array.of(0, 1, 0, 0, 0)},
        offsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
        mask: {format: 'uint32', values: Uint32Array.of(1, 1, 0, 0, 0)},
        parameters: {format: 'float32', values: Float32Array.of(1, 1, 0, 1)}
      },
      operations: [
        new Operation({
          positions: 'points',
          timestamps: 'times',
          trackOffsets: 'offsets',
          mask: 'mask',
          parameters: 'parameters',
          rowCount: 5,
          capacity: 1,
          output: {
            ids: 'ids',
            status: {
              count: 'count',
              overflow: 'overflow',
              candidateOverflow: 'candidateOverflow',
              missingFixCount: 'missing'
            }
          }
        })
      ],
      outputs: ['count', 'missing']
    });
    expect(Array.from(result['count'])).toEqual([1]);
    expect(Array.from(result['missing'])).toEqual([0]);
  }
});
