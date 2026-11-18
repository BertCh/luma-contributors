// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {
  LensCommandGraph,
  addLensViewSnapshot,
  submitLensGraph,
  type LensViewSnapshot
} from '../../src/computation-lenses/lens-view-snapshot';
import {runExample} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/zone-events.example';
import {getZoneEventLossCaptures} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/zone-event-loss-captures';
import {formatZoneEventReadout} from '../../../../website/src/gpu/runs/entries/recipes-trajectories/lib/zone-event-readout';

test.each([
  {cause: 'per-track event loss', candidateCapacity: 32, trackEventCapacity: 1, scratch: 0},
  {cause: 'candidate scratch loss', candidateCapacity: 1, trackEventCapacity: 4, scratch: 1}
])('spatial-4:F2 standalone consumer distinguishes $cause with zero final overflow', async ({
  candidateCapacity,
  trackEventCapacity,
  scratch
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new LensCommandGraph<undefined>(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    id: string,
    data: Float32Array | Uint32Array,
    format: Format,
    length: number
  ) => {
    const buffer = device.createBuffer({id, data, usage: Buffer.STORAGE | Buffer.COPY_DST});
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, length});
  };
  let compiled: ReturnType<typeof graph.compile> | undefined;
  let snapshot: LensViewSnapshot | undefined;
  try {
    // One square [0,2] x [0,2]; the track enters at t=1 and exits at t=3, dwelling 2 seconds.
    const output = await runExample(graph, {
      candidateCapacity,
      trackEventCapacity,
      tracks: {
        positions: input('positions', new Float32Array([-1, 1, 3, 1]), 'float32x2', 2),
        timestamps: input('times', new Float32Array([0, 4]), 'float32', 2),
        trackOffsets: input('offsets', new Uint32Array([0, 2]), 'uint32', 2)
      },
      zones: {
        positions: input('zones', new Float32Array([0, 0, 2, 0, 2, 2, 0, 2]), 'float32x2', 4),
        featureOffsets: input('features', new Uint32Array([0, 1]), 'uint32', 2),
        polygonOffsets: input('polygons', new Uint32Array([0, 1]), 'uint32', 2),
        ringOffsets: input('rings', new Uint32Array([0, 4]), 'uint32', 2)
      }
    });
    const loss = getZoneEventLossCaptures(output.events.status, output.status.candidateOverflow);
    snapshot = addLensViewSnapshot(graph, device, {
      id: 'zone-events-final1',
      views: {
        eventCount: output.events.status.count,
        eventOverflow: loss.eventOverflow.view,
        eventCandidateOverflow: loss.eventCandidateOverflow.view,
        candidateScratchOverflow: loss.scratchCandidateOverflow.view,
        dwell: output.dwellTimes,
        trackEventCount: output.trackEventCounts
      }
    });
    compiled = graph.compile();
    submitLensGraph(device, compiled, {parameters: undefined});
    const captures = await snapshot.arena.read({graph: 'compiled'});
    const read = (name: string) =>
      Number(captures.find(capture => capture.name === name)!.values[0]);
    expect(read('eventOverflow')).toBe(0);
    expect(read('eventCandidateOverflow')).toBe(1);
    expect(read('candidateScratchOverflow')).toBe(scratch);
    const dwell = read('dwell');
    if (!scratch) {
      expect(Number.isFinite(dwell)).toBe(true);
      expect(read('eventCount')).toBe(1);
      expect(read('trackEventCount')).toBe(2);
      expect(dwell).toBeCloseTo(2, 5);
    }
    const readout = formatZoneEventReadout(
      'A',
      read('eventCount'),
      {
        eventOverflow: read('eventOverflow'),
        eventCandidateOverflow: read('eventCandidateOverflow'),
        scratchCandidateOverflow: read('candidateScratchOverflow')
      },
      Math.round(dwell)
    );
    expect(readout).toContain('crossings (incomplete)');
    expect(readout.endsWith('s (incomplete)')).toBe(Boolean(scratch));
  } finally {
    snapshot?.arena.destroy();
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
