// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {CPUTrajectoryThinningOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation';
import {GPUTrajectoryThinning} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning';
import {GPUTrajectoryThinningOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning-operation';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GraphHarness, importTrajectory} from '../trajectory-staypoints/movement-spec-utils';

// Hand oracle, times [0,1,2,3]: tolerance 2 keeps rows [0,2,3], including
// the forced endpoint; capacity 2 loses one of three rows. Tolerance 10 with
// budget 1 inspects only row 1 from anchor 0, leaving rows 2 and 3 unresolved:
// it resumes at row 2, keeps [0,2,3] and reports candidateOverflow=1. Budget 4
// inspects the whole track, keeps only [0,3] and reports no candidate loss.
// Empty offsets select no rows. Status order: count, required, overflow, candidate.
for (const scenario of [
  {name: 'exact fit', capacity: 3, tolerance: 2, budget: 4, empty: false, status: [3, 3, 0, 0]},
  {name: 'output loss', capacity: 2, tolerance: 2, budget: 4, empty: false, status: [2, 3, 1, 0]},
  {name: 'search loss', capacity: 3, tolerance: 10, budget: 1, empty: false, status: [3, 3, 0, 1]},
  {
    name: 'complete search',
    capacity: 3,
    tolerance: 10,
    budget: 4,
    empty: false,
    status: [2, 2, 0, 0]
  },
  {name: 'empty track', capacity: 3, tolerance: 2, budget: 4, empty: true, status: [0, 0, 0, 0]}
]) {
  it(`thinning contributor overwrites poisoned status: ${scenario.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, `thinning-status-${scenario.name}`);
    const trajectory = importTrajectory(harness, {
      positions: new Float32Array(8),
      times: new Float32Array([0, 1, 2, 3]),
      trackOffsets: scenario.empty ? [0, 0] : [0, 4]
    });
    const count = harness.output('count', 'uint32', 1);
    const required = harness.output('required', 'uint32', 1);
    const overflow = harness.output('overflow', 'uint32', 1);
    const candidate = harness.output('candidate', 'uint32', 1);
    for (const output of [count, required, overflow, candidate]) {
      output.buffer.write(new Uint32Array([0xdeadbeef]));
    }
    harness.graph.add(
      new GPUTrajectoryThinning({
        mode: 'minimum-time',
        maximumSearchRows: scenario.budget,
        positions: trajectory.positions,
        timestamps: trajectory.timestamps,
        trackOffsets: trajectory.trackOffsets,
        parameters: harness.input(
          'parameters',
          new Float32Array([scenario.tolerance, 0, 0, 0]),
          'float32',
          4
        ),
        output: {
          sourceRows: harness.output('rows', 'uint32', scenario.capacity).view,
          trackOffsets: harness.output('output-offsets', 'uint32', 2).view,
          status: {
            count: count.view,
            requiredCount: required.view,
            overflow: overflow.view,
            candidateOverflow: candidate.view
          }
        }
      })
    );
    const status = await harness.run(async () => [
      ...(await readUint32(count.buffer, 1)),
      ...(await readUint32(required.buffer, 1)),
      ...(await readUint32(overflow.buffer, 1)),
      ...(await readUint32(candidate.buffer, 1))
    ]);
    expect(status).toEqual(scenario.status);
  });

  for (const Operation of [GPUTrajectoryThinningOperation, CPUTrajectoryThinningOperation]) {
    it(`${Operation.name} publishes independent thinning status: ${scenario.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array(8)},
          timestamps: {format: 'float32', values: new Float32Array([0, 1, 2, 3])},
          trackOffsets: {
            format: 'uint32',
            values: new Uint32Array(scenario.empty ? [0, 0] : [0, 4])
          },
          parameters: {format: 'float32', values: new Float32Array([scenario.tolerance, 0, 0, 0])}
        },
        operations: [
          new Operation({
            mode: 'minimum-time',
            maximumSearchRows: scenario.budget,
            positions: 'positions',
            timestamps: 'timestamps',
            trackOffsets: 'trackOffsets',
            parameters: 'parameters',
            rowCount: 4,
            trackCount: 1,
            capacity: scenario.capacity,
            output: {
              sourceRows: 'rows',
              trackOffsets: 'outputOffsets',
              status: {
                count: 'count',
                requiredCount: 'required',
                overflow: 'overflow',
                candidateOverflow: 'candidate'
              }
            }
          })
        ],
        outputs: ['count', 'required', 'overflow', 'candidate']
      });
      expect([
        ...result['count'],
        ...result['required'],
        ...result['overflow'],
        ...result['candidate']
      ]).toEqual(scenario.status);
    });
  }
}
