// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUTrajectoryRestriction} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction';
import {GPUTrajectoryRestrictionOperation} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction-operation';
import {CPUTrajectoryRestrictionOperation} from '../../../src/gpu-spatial-analysis/trajectory-restriction/cpu-trajectory-restriction-operation';
import {GraphHarness} from '../trajectory-staypoints/movement-spec-utils';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {RESTRICTION_STATUS_CASES} from './restriction-status-oracles-fixture';

for (const scenario of [
  {name: 'active singleton pair', active: 1, first: 0, status: [50, 0]},
  {name: 'inactive pair', active: 0, first: 0, status: [0, 0]},
  {name: 'invalid track', active: 1, first: 99, status: [4, 0]}
]) {
  it(`restriction clears poisoned pair status: ${scenario.name}`, async () => {
    // Closed temporal support at t=0, two singleton tracks 0.5 apart:
    // 32 temporal + 16 fully inside + 2 singleton = 50; inactive clears all.
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'restriction-pair-status');
    const status = harness.output('status', 'uint32', 2);
    status.buffer.write(new Uint32Array(2).fill(0xdeadbeef));
    const positionStorage = harness.output('output-positions', 'float32x2', 4);
    const relationships = harness.output('relationships', 'uint32', 4);
    harness.graph.add(
      new GPUTrajectoryRestriction({
        positions: harness.input('positions', new Float32Array([0, 0, 0, 0.5]), 'float32x2', 2),
        times: harness.input('times', new Float32Array([0, 0]), 'float32', 2),
        offsets: harness.input('offsets', new Uint32Array([0, 1, 2]), 'uint32', 3),
        parameters: harness.input(
          'parameters',
          new Float32Array([0, 4, 1, 0, 0, 0, 0]),
          'float32',
          7
        ),
        pairs: {
          first: harness.input('first', new Uint32Array([scenario.first]), 'uint32', 1),
          second: harness.input('second', new Uint32Array([1]), 'uint32', 1),
          count: harness.input('pair-count', new Uint32Array([scenario.active]), 'uint32', 1)
        },
        intervalsPerPair: 2,
        output: {
          status: status.view,
          intervals: harness.output('intervals', 'float32x2', 2).view,
          positions: harness.graph.createDataView(positionStorage.view.buffer, {
            format: 'float32x4',
            length: 2
          }),
          relationships: harness.graph.createDataView(relationships.view.buffer, {
            format: 'uint32x4',
            length: 1
          })
        }
      })
    );
    await harness.run(async () =>
      expect(await readUint32(status.buffer, 2)).toEqual(scenario.status)
    );
  });
  for (const Operation of [GPUTrajectoryRestrictionOperation, CPUTrajectoryRestrictionOperation]) {
    it(`${Operation.name} independent pair status: ${scenario.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array([0, 0, 0, 0.5])},
          times: {format: 'float32', values: new Float32Array([0, 0])},
          offsets: {format: 'uint32', values: new Uint32Array([0, 1, 2])},
          parameters: {format: 'float32', values: new Float32Array([0, 4, 1, 0, 0, 0, 0])},
          first: {format: 'uint32', values: new Uint32Array([scenario.first])},
          second: {format: 'uint32', values: new Uint32Array([1])},
          count: {format: 'uint32', values: new Uint32Array([scenario.active])}
        },
        operations: [
          new Operation({
            positions: 'positions',
            timestamps: 'times',
            trackOffsets: 'offsets',
            parameters: 'parameters',
            rowCount: 2,
            trackCount: 2,
            pairs: {first: 'first', second: 'second', count: 'count'},
            pairCapacity: 1,
            intervalsPerPair: 2,
            output: {
              intervals: 'intervals',
              positions: 'outputPositions',
              status: 'status',
              relationships: 'relationships'
            }
          })
        ],
        outputs: ['status']
      });
      expect(Array.from(result['status'])).toEqual(scenario.status);
    });
  }
}

for (const scenario of RESTRICTION_STATUS_CASES) {
  it(`restriction overwrites every poisoned status word: ${scenario.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'restriction-status');
    const status = harness.output('status', 'uint32', 4);
    status.buffer.write(new Uint32Array(4).fill(0xdeadbeef));
    const positionStorage = harness.output('output-positions', 'float32x2', 8);
    harness.graph.add(
      new GPUTrajectoryRestriction({
        positions: harness.input(
          'positions',
          Float32Array.from(scenario.positions),
          'float32x2',
          2
        ),
        times: harness.input('times', new Float32Array([0, 4]), 'float32', 2),
        offsets: harness.input('offsets', new Uint32Array([0, 2]), 'uint32', 2),
        parameters: harness.input(
          'parameters',
          Float32Array.from(scenario.parameters),
          'float32',
          scenario.parameters.length
        ),
        geometryBoundary: scenario.boundary,
        polygonRingVertexCounts: scenario.rings,
        intervalsPerSegment: 2,
        output: {
          intervals: harness.output('intervals', 'float32x2', 4).view,
          positions: harness.graph.createDataView(positionStorage.view.buffer, {
            format: 'float32x4',
            length: 4
          }),
          status: status.view,
          relationships: harness.graph.createDataView(
            harness.output('relationships', 'float32x2', 2).view.buffer,
            {format: 'uint32x4', length: 1}
          )
        }
      })
    );
    await harness.run(async () =>
      expect(await readUint32(status.buffer, 4)).toEqual(scenario.status)
    );
  });
  for (const Operation of [GPUTrajectoryRestrictionOperation, CPUTrajectoryRestrictionOperation]) {
    it(`${Operation.name} whole status oracle: ${scenario.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: Float32Array.from(scenario.positions)},
          times: {format: 'float32', values: new Float32Array([0, 4])},
          offsets: {format: 'uint32', values: new Uint32Array([0, 2])},
          parameters: {format: 'float32', values: Float32Array.from(scenario.parameters)}
        },
        operations: [
          new Operation({
            positions: 'positions',
            timestamps: 'times',
            trackOffsets: 'offsets',
            parameters: 'parameters',
            rowCount: 2,
            trackCount: 1,
            geometryBoundary: scenario.boundary,
            polygonRingVertexCounts: scenario.rings,
            intervalsPerSegment: 2,
            output: {
              intervals: 'intervals',
              positions: 'outputPositions',
              status: 'status',
              relationships: 'relationships'
            }
          })
        ],
        outputs: ['status']
      });
      expect(Array.from(result['status'])).toEqual(scenario.status);
    });
  }
}
