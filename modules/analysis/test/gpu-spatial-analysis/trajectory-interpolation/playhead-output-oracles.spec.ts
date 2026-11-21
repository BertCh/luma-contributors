// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTrajectoryPlayhead} from '../../../src/gpu-spatial-analysis/trajectory-interpolation/gpu-trajectory-playhead';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphHarness, importTrajectory} from '../trajectory-staypoints/movement-spec-utils';

it('playhead active compaction writes zero candidate loss with ordinary columns omitted', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const withCandidate of [true, false]) {
    const harness = new GraphHarness(device, `playhead-candidate-${withCandidate}`);
    const trajectory = importTrajectory(harness, {
      positions: [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
      ],
      times: Float32Array.of(0, 2, 4, 6),
      trackOffsets: [0, 2, 4]
    });
    const ids = harness.output('ids', 'uint32', 2);
    const count = harness.output('count', 'uint32', 1);
    const overflow = harness.output('overflow', 'uint32', 1);
    const candidate = withCandidate ? harness.output('candidate', 'uint32', 1) : undefined;
    for (const output of [ids, count, overflow, candidate])
      output?.buffer.write(new Uint32Array(output.buffer.byteLength / 4).fill(93));
    harness.graph.add(
      new GPUTrajectoryPlayhead({
        ...trajectory,
        parameters: harness.input('parameters', Float32Array.of(1, 0, 0, 0), 'float32', 4),
        output: {
          activeTracks: {
            ids: ids.view,
            status: {
              count: count.view,
              overflow: overflow.view,
              ...(candidate ? {candidateOverflow: candidate.view} : {})
            }
          }
        }
      })
    );
    const result = await harness.run(async () => ({
      ids: await readUint32(ids.buffer, 1),
      count: await readUint32(count.buffer, 1),
      overflow: await readUint32(overflow.buffer, 1),
      candidate: candidate ? await readUint32(candidate.buffer, 1) : undefined
    }));
    // Only track 0 contains t=1; compaction scans every track and has no candidate-loss source.
    expect(result.ids).toEqual([0]);
    expect(result.count).toEqual([1]);
    expect(result.overflow).toEqual([0]);
    if (candidate) expect(result.candidate).toEqual([0]);
  }
});
