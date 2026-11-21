// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {CPUTrajectorySplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-split-operation';
import {
  GPUTrajectorySplit,
  getGPUTrajectorySplitParameterValues
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split';
import {GPUTrajectorySplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split-operation';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GraphHarness, importTrajectory} from '../trajectory-staypoints/movement-spec-utils';

// Hand oracle: gaps > 2 split [0,1,4,5,9,10] into [0,1], [2,3], [4,5].
// Three segments require six rows. Count clamps only to segment capacity;
// either capacity loss sets overflow. With all gaps > 2 each run is a singleton
// and is dropped, giving count=requiredCount=overflow=0. All words are exact.
for (const scenario of [
  {name: 'exact fit', segmentCapacity: 3, rowCapacity: 6, empty: false, status: [3, 3, 0]},
  {name: 'segment loss', segmentCapacity: 2, rowCapacity: 6, empty: false, status: [2, 3, 1]},
  {name: 'row loss', segmentCapacity: 3, rowCapacity: 5, empty: false, status: [3, 3, 1]},
  {name: 'singletons dropped', segmentCapacity: 3, rowCapacity: 6, empty: true, status: [0, 0, 0]}
]) {
  it(`split contributor overwrites poisoned status: ${scenario.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, `split-status-${scenario.name}`);
    const trajectory = importTrajectory(harness, {
      positions: new Float32Array(12),
      times: new Float32Array(scenario.empty ? [0, 4, 8, 12, 16, 20] : [0, 1, 4, 5, 9, 10]),
      trackOffsets: [0, 6]
    });
    const count = harness.output('count', 'uint32', 1);
    const required = harness.output('required', 'uint32', 1);
    const overflow = harness.output('overflow', 'uint32', 1);
    for (const output of [count, required, overflow]) {
      output.buffer.write(new Uint32Array([0xdeadbeef]));
    }
    harness.graph.add(
      new GPUTrajectorySplit({
        mode: 'gap',
        positions: trajectory.positions,
        timestamps: trajectory.timestamps,
        trackOffsets: trajectory.trackOffsets,
        parameters: harness.input(
          'parameters',
          getGPUTrajectorySplitParameterValues({maximumGap: 2}),
          'float32',
          4
        ),
        output: {
          segmentOffsets: harness.output('segment-offsets', 'uint32', scenario.segmentCapacity + 1)
            .view,
          segmentTracks: harness.output('segment-tracks', 'uint32', scenario.segmentCapacity).view,
          rowIds: harness.output('rows', 'uint32', scenario.rowCapacity).view,
          status: {count: count.view, requiredCount: required.view, overflow: overflow.view}
        }
      })
    );
    const status = await harness.run(async () => [
      ...(await readUint32(count.buffer, 1)),
      ...(await readUint32(required.buffer, 1)),
      ...(await readUint32(overflow.buffer, 1))
    ]);
    expect(status).toEqual(scenario.status);
  });

  for (const Operation of [GPUTrajectorySplitOperation, CPUTrajectorySplitOperation]) {
    it(`${Operation.name} publishes independent split status: ${scenario.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array(12)},
          timestamps: {
            format: 'float32',
            values: new Float32Array(scenario.empty ? [0, 4, 8, 12, 16, 20] : [0, 1, 4, 5, 9, 10])
          },
          trackOffsets: {format: 'uint32', values: new Uint32Array([0, 6])},
          parameters: {
            format: 'float32',
            values: getGPUTrajectorySplitParameterValues({maximumGap: 2})
          }
        },
        operations: [
          new Operation({
            mode: 'gap',
            positions: 'positions',
            timestamps: 'timestamps',
            trackOffsets: 'trackOffsets',
            parameters: 'parameters',
            rowCount: 6,
            segmentCapacity: scenario.segmentCapacity,
            rowCapacity: scenario.rowCapacity,
            output: {
              segmentOffsets: 'offsets',
              segmentTracks: 'tracks',
              rowIds: 'rows',
              status: {count: 'count', requiredCount: 'required', overflow: 'overflow'}
            }
          })
        ],
        outputs: ['count', 'required', 'overflow']
      });
      expect([...result['count'], ...result['required'], ...result['overflow']]).toEqual(
        scenario.status
      );
    });
  }
}
