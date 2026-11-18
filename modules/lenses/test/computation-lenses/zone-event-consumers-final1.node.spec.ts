// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, expect, test, vi} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {Buffer} from '@luma.gl/core';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {createGPUComputationRecord} from '@luma-contributors/analysis/computation-record';
import * as example from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/zone-events.example';
import * as lensRun from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s3-lens-run';
import {getLensScenario} from '../../src/computation-lenses/registry-implementation';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/zone-events.lens';
import {formatZoneEventReadout} from '../../../../website/src/gpu/runs/entries/recipes-trajectories/lib/zone-event-readout';
import {entries} from '../../../../website/src/catalog/entries/recipes-trajectories';

afterEach(() => vi.restoreAllMocks());

test('spatial-4:F2 catalog example carries both loss flags and does not certify zero final overflow', () => {
  const entry = entries.find(candidate => candidate.id === 'zone-events')!;
  expect(entry.outputs.find(output => output.name === 'status')!.description).not.toContain(
    'zero means complete'
  );
  expect(entry.example).toContain('candidateOverflow: eventCandidateOverflow');
  expect(entry.example).toContain('candidateOverflow: candidateScratchOverflow');
});

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph<undefined>(device);
  // This test inspects consumer bindings. Shader execution is covered by the headless spec.
  vi.spyOn(graph, 'add').mockImplementation(() => {});
  const input: example.ZoneEventsExampleInput = {
    trackEventCapacity: 1,
    tracks: {
      positions: createTransientView(graph, 'fixture-positions', 'float32x2', 2),
      timestamps: createTransientView(graph, 'fixture-timestamps', 'float32', 2),
      trackOffsets: createTransientView(graph, 'fixture-trackOffsets', 'uint32', 2)
    },
    zones: {
      positions: createTransientView(graph, 'zones', 'float32x2', 4),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2)
    }
  };
  return {device, graph, input};
}

test('spatial-4:F2 standalone example exposes independent event-list and scratch loss views', async () => {
  const {device, graph, input} = createFixture();
  try {
    const result = await example.runExample(graph, input);
    expect(result.events.status).toHaveProperty('candidateOverflow');
    expect(result).toHaveProperty('status.candidateOverflow');
    expect(result.events.status.candidateOverflow).not.toBe(result.events.status.overflow);
    expect(result.status.candidateOverflow).not.toBe(result.events.status.candidateOverflow);
  } finally {
    device.destroy();
  }
});

test('spatial-4:F2 standalone lens captures the producer loss views instead of just final overflow', async () => {
  const {device, graph} = createFixture();
  let captured: Record<string, lensRun.S3CaptureSpec> = {};
  const runExample = vi.spyOn(example, 'runExample');
  vi.spyOn(lensRun, 'runS3Lens').mockImplementation(async (_device, props) => {
    const uploads: lensRun.S3Uploads = {
      own: resource => resource,
      view(id, data, format, length) {
        const components = Number(format.match(/x([234])$/)?.[1] ?? 1);
        const handle = graph.createTransientBuffer({
          id,
          byteLength: data.byteLength,
          usage: Buffer.STORAGE
        });
        return graph.createDataView(handle, {format, length: length ?? data.length / components});
      }
    };
    captured = await props.build(graph, uploads);
    return createGPUComputationRecord({
      id: 'fixture',
      domains: [],
      adapter: {backend: 'null'},
      graphs: []
    });
  });
  try {
    const scenario = await getLensScenario('gpu-spatial-analysis.zone-events')!.load();
    await scenario.run(device);
    expect(captured).toHaveProperty('eventCandidateOverflow');
    expect(captured).toHaveProperty('candidateScratchOverflow');
    const produced = await runExample.mock.results[0].value;
    expect(captured['eventCandidateOverflow'].view).toBe(produced.events.status.candidateOverflow);
    expect(captured['candidateScratchOverflow'].view).toBe(produced.status.candidateOverflow);
    expect(captured['dwellTimes'].meaning).toContain('candidateScratchOverflow');
    expect(captured['visitCounts'].meaning).toContain('candidateScratchOverflow');
    expect(captured['trackEventCounts'].meaning).toContain('candidateScratchOverflow');
    expect(captured['tracks'].keySpace).toBe('fix');
    expect(captured['tracks'].geometry).toHaveProperty('featureKeySpace', 'track');
    expect(captured['countedTracks'].geometry).toHaveProperty('featureKeySpace', 'track');
  } finally {
    device.destroy();
  }
});

test.each([
  [0, 1, 0, 'zone A · 1 crossings (incomplete) · dwell 2 s'],
  [0, 1, 1, 'zone A · 1 crossings (incomplete) · dwell 2 s (incomplete)']
])('spatial-4:F2 consumer preserves final=%i event-candidate=%i scratch=%i', (eventOverflow, eventCandidateOverflow, scratchCandidateOverflow, expected) => {
  expect(
    formatZoneEventReadout(
      'A',
      1,
      {eventOverflow, eventCandidateOverflow, scratchCandidateOverflow},
      2
    )
  ).toBe(expected);
});
