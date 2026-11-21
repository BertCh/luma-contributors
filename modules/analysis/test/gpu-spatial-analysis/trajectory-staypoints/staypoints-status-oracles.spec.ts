// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUTrajectoryLocations} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations';
import {GPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations-operation';
import {CPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
import {GPUTrajectoryStaypoints} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints';
import {GPUTrajectoryStaypointsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints-operation';
import {CPUTrajectoryStaypointsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {GraphHarness} from './movement-spec-utils';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

// Four points at x=0,1,2,3 with epsilon=1 form a path. Initial labels
// [0,1,2,3] hook to [0,0,1,2] and halve to [0,0,0,1]. Edge (2,3)
// remains unresolved: row 3 still has label 1, adjacent row 2 has label 0.
// A second round gives [0,0,0,0], stable. This hand derivation fixes the
// positive-budget stopping decision independently of either implementation.
for (const scenario of [
  {rounds: 1, status: [0, 0, 1], labels: new Array(4).fill(0xffffffff)},
  {rounds: 2, status: [1, 1, 0], labels: [0, 0, 0, 0]}
]) {
  it(`locations overwrite poisoned status with unresolved-edge oracle after ${scenario.rounds} rounds`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'locations-status');
    const count = harness.output('count', 'uint32', 1);
    const converged = harness.output('converged', 'uint32', 1);
    const limit = harness.output('limit', 'uint32', 1);
    const labels = harness.output('labels', 'uint32', 4);
    const centers = harness.output('centers', 'float32x2', 4);
    centers.buffer.write(new Float32Array(8).fill(93));
    for (const output of [count, converged, limit, labels])
      output.buffer.write(new Uint32Array(output.view.length).fill(0xdeadbeef));
    harness.graph.add(
      new GPUTrajectoryLocations({
        centroids: harness.input(
          'centroids',
          new Float32Array([0, 0, 1, 0, 2, 0, 3, 0]),
          'float32x2',
          4
        ),
        tracks: harness.input('tracks', new Uint32Array(4), 'uint32', 4),
        count: harness.input('input-count', new Uint32Array([4]), 'uint32', 1),
        parameters: harness.input('parameters', new Float32Array([1, 0, 0, 0]), 'float32', 4),
        maximumIterations: scenario.rounds,
        output: {
          locationIds: labels.view,
          centers: centers.view,
          status: {
            count: count.view,
            converged: converged.view,
            iterationLimitReached: limit.view
          }
        }
      })
    );
    await harness.run(async () => {
      expect([
        ...(await readUint32(count.buffer, 1)),
        ...(await readUint32(converged.buffer, 1)),
        ...(await readUint32(limit.buffer, 1))
      ]).toEqual(scenario.status);
      expect(await readUint32(labels.buffer, 4)).toEqual(scenario.labels);
      expect(await readFloat32(centers.buffer, 8)).toEqual(
        scenario.rounds === 1 ? new Array(8).fill(93) : [1.5, 0, 93, 93, 93, 93, 93, 93]
      );
    });
  });
  for (const Operation of [GPUTrajectoryLocationsOperation, CPUTrajectoryLocationsOperation]) {
    it(`${Operation.name} pins unresolved-edge status after ${scenario.rounds} rounds`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const result = await runOperationGraph(device, {
        inputs: {
          centroids: {format: 'float32x2', values: new Float32Array([0, 0, 1, 0, 2, 0, 3, 0])},
          tracks: {format: 'uint32', values: new Uint32Array(4)},
          count: {format: 'uint32', values: new Uint32Array([4])},
          parameters: {format: 'float32', values: new Float32Array([1, 0, 0, 0])}
        },
        operations: [
          new Operation({
            centroids: 'centroids',
            tracks: 'tracks',
            count: 'count',
            parameters: 'parameters',
            capacity: 4,
            maximumIterations: scenario.rounds,
            output: {
              locationIds: 'labels',
              status: {count: 'outputCount', converged: 'converged', iterationLimitReached: 'limit'}
            }
          })
        ],
        outputs: ['labels', 'outputCount', 'converged', 'limit']
      });
      expect([...result['outputCount'], ...result['converged'], ...result['limit']]).toEqual(
        scenario.status
      );
      expect(Array.from(result['labels'])).toEqual(scenario.labels);
    });
  }
}

it('locations suppress and recover on the same compiled graph and output buffers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'locations-status-recovery');
  const parameters = harness.output('parameters', 'float32', 4);
  parameters.buffer.write(Float32Array.of(1, 0, 0, 0));
  const labels = harness.output('labels', 'uint32', 4);
  const centers = harness.output('centers', 'float32x2', 4);
  const status = ['count', 'converged', 'limit'].map(name => harness.output(name, 'uint32', 1));
  harness.graph.add(
    new GPUTrajectoryLocations({
      maximumIterations: 1,
      centroids: harness.input('points', Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0), 'float32x2', 4),
      tracks: harness.input('tracks', new Uint32Array(4), 'uint32', 4),
      count: harness.input('active', Uint32Array.of(4), 'uint32', 1),
      parameters: parameters.view,
      output: {
        locationIds: labels.view,
        centers: centers.view,
        status: {
          count: status[0].view,
          converged: status[1].view,
          iterationLimitReached: status[2].view
        }
      }
    })
  );
  await harness.run(async compiled => {
    for (const epsilon of [1, 2, 1, 2]) {
      parameters.buffer.write(Float32Array.of(epsilon, 0, 0, 0));
      centers.buffer.write(new Float32Array(8).fill(93));
      submitGraph(device, compiled, undefined);
      expect(
        (await Promise.all(status.map(output => readUint32(output.buffer, 1)))).flat()
      ).toEqual(epsilon === 1 ? [0, 0, 1] : [1, 1, 0]);
      expect(await readUint32(labels.buffer, 4)).toEqual(
        new Array(4).fill(epsilon === 1 ? 0xffffffff : 0)
      );
      expect(await readFloat32(centers.buffer, 8)).toEqual(
        epsilon === 1 ? new Array(8).fill(93) : [1.5, 0, 93, 93, 93, 93, 93, 93]
      );
    }
  });
});

// Two independent stationary two-fix tracks, each lasting 1, with includeLast=1
// and minimum duration 1, require exactly two stays. Capacity 1 loses one stay;
// capacity 2 fits exactly. A search of 4 rows covers both tracks, so candidate=0.
for (const scenario of [
  {capacity: 1, status: [1, 2, 1, 0, 1]},
  {capacity: 2, status: [2, 2, 0, 0, 2]}
]) {
  it(`staypoints poison exact capacity status at ${scenario.capacity} slots`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'staypoint-capacity-status');
    const outputs = ['count', 'required', 'overflow', 'candidate', 'draw'].map(name =>
      harness.output(name, 'uint32', 1)
    );
    for (const output of outputs) output.buffer.write(new Uint32Array([0xdeadbeef]));
    harness.graph.add(
      new GPUTrajectoryStaypoints({
        positions: harness.input(
          'positions',
          new Float32Array([0, 0, 0, 0, 10, 0, 10, 0]),
          'float32x2',
          4
        ),
        timestamps: harness.input('timestamps', new Float32Array([0, 1, 0, 1]), 'float32', 4),
        trackOffsets: harness.input('offsets', new Uint32Array([0, 2, 4]), 'uint32', 3),
        parameters: harness.input('parameters', new Float32Array([1, 1, 0, 1]), 'float32', 4),
        maximumSearchRows: 4,
        output: {
          ids: harness.output('ids', 'uint32', scenario.capacity).view,
          drawInstanceCount: outputs[4].view,
          status: {
            count: outputs[0].view,
            requiredCount: outputs[1].view,
            overflow: outputs[2].view,
            candidateOverflow: outputs[3].view
          }
        }
      })
    );
    await harness.run(async () => {
      const words = await Promise.all(outputs.map(output => readUint32(output.buffer, 1)));
      expect(words.flat()).toEqual(scenario.status);
    });
  });
  for (const Operation of [GPUTrajectoryStaypointsOperation, CPUTrajectoryStaypointsOperation]) {
    it(`${Operation.name} independently requires two stays at capacity ${scenario.capacity}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array([0, 0, 0, 0, 10, 0, 10, 0])},
          timestamps: {format: 'float32', values: new Float32Array([0, 1, 0, 1])},
          offsets: {format: 'uint32', values: new Uint32Array([0, 2, 4])},
          parameters: {format: 'float32', values: new Float32Array([1, 1, 0, 1])}
        },
        operations: [
          new Operation({
            positions: 'positions',
            timestamps: 'timestamps',
            trackOffsets: 'offsets',
            parameters: 'parameters',
            rowCount: 4,
            capacity: scenario.capacity,
            maximumSearchRows: 4,
            output: {
              ids: 'ids',
              drawInstanceCount: 'draw',
              status: {
                count: 'count',
                requiredCount: 'required',
                overflow: 'overflow',
                candidateOverflow: 'candidate'
              }
            }
          })
        ],
        outputs: ['count', 'required', 'overflow', 'candidate', 'draw']
      });
      expect(
        ['count', 'required', 'overflow', 'candidate', 'draw'].flatMap(name =>
          Array.from(result[name])
        )
      ).toEqual(scenario.status);
    });
  }
}
