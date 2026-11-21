// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUZoneEvents} from '../../../src/gpu-spatial-analysis/trajectory-zones/gpu-zone-events';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphHarness, importTrajectory} from '../trajectory-staypoints/movement-spec-utils';

it('zone candidate count includes the initial ray and both segment bounding-box probes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const withCandidate of [true, false]) {
    const harness = new GraphHarness(device, `zone-count-${withCandidate}`);
    const trajectory = importTrajectory(harness, {
      positions: [
        [-1, 1],
        [1, 1],
        [3, 1]
      ],
      times: Float32Array.of(0, 1, 2),
      trackOffsets: [0, 3]
    });
    const ids = harness.output('ids', 'uint32', 4);
    const count = harness.output('count', 'uint32', 1);
    const overflow = harness.output('overflow', 'uint32', 1);
    const candidate = withCandidate ? harness.output('candidate', 'uint32', 1) : undefined;
    for (const output of [ids, count, overflow, candidate])
      output?.buffer.write(new Uint32Array(output.buffer.byteLength / 4).fill(93));
    harness.graph.add(
      new GPUZoneEvents({
        tracks: {kind: 'trajectories', ...trajectory},
        zones: {
          kind: 'polygons',
          positions: harness.input(
            'square',
            Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2),
            'float32x2',
            4
          ),
          featureOffsets: harness.input('features', Uint32Array.of(0, 1), 'uint32', 2),
          polygonOffsets: harness.input('polygons', Uint32Array.of(0, 1), 'uint32', 2),
          ringOffsets: harness.input('rings', Uint32Array.of(0, 4), 'uint32', 2)
        },
        candidateCapacity: 8,
        trackEventCapacity: 4,
        output: {
          events: {ids: ids.view, status: {count: count.view, overflow: overflow.view}},
          ...(candidate ? {status: {candidateCount: candidate.view}} : {})
        }
      })
    );
    const result = await harness.run(async () => ({
      count: await readUint32(count.buffer, 1),
      ids: await readUint32(ids.buffer, 2),
      overflow: await readUint32(overflow.buffer, 1),
      candidate: candidate ? await readUint32(candidate.buffer, 1) : undefined
    }));
    // Ray at (-1,1) hits both vertical edge boxes (2), first segment hits x=0 (1),
    // second hits x=2 (1). Horizontal edge boxes have y=0 or y=2 and never overlap.
    if (candidate) expect(result.candidate).toEqual([4]);
    expect(result.count).toEqual([2]);
    expect(result.ids).toEqual([0, 0]);
    expect(result.overflow).toEqual([0]);
  }
});
