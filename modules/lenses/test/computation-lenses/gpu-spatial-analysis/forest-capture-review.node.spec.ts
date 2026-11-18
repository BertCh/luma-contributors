// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {fromRecordedLensFile} from '../../../src/computation-lenses/registry';
import {runExample as runForest} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-weights-minimum-spanning-tree.example';
import {runExample as runSkater} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/skater-regions.example';
import * as utilities from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-scenario-utils';

for (const [name, run] of [
  ['forest', runForest],
  ['SKATER', runSkater]
] as const)
  it(`R1 ${name} exposes the actual candidate flag to capture`, async () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const execution = await run(device, {
        graph,
        values: createTransientView(graph, 'values', 'float32', 3),
        parameters: createTransientView(graph, 'parameters', 'uint32', 2),
        tracts: {
          kind: 'polygons',
          positions: createTransientView(graph, 'positions', 'float32x2', 16),
          ringOffsets: createTransientView(graph, 'rings', 'uint32', 4),
          polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 4),
          featureOffsets: createTransientView(graph, 'features', 'uint32', 4)
        }
      });
      const views = execution.views as Record<string, unknown>;
      expect(views['weightsCandidateOverflow']).toBeDefined();
      expect(views['weightsCandidateOverflow']).toBe(execution.weights.status?.candidateOverflow);
      if (name === 'forest') expect(views['edgeCount']).toBeDefined();
    } finally {
      device.destroy();
    }
  });

it('R1 the capture boundary refuses either cause and accepts clean flags', () => {
  const captures = (finalLoss: number, candidateLoss: number) => [
    {name: 'weightsOverflow', values: Uint32Array.of(finalLoss)},
    {name: 'weightsCandidateOverflow', values: Uint32Array.of(candidateLoss)}
  ];
  expect(() => utilities.validateSpatialWeightsCaptures(captures(0, 9))).toThrow(
    /weightsCandidateOverflow/
  );
  expect(() => utilities.validateSpatialWeightsCaptures(captures(9, 0))).toThrow(/weightsOverflow/);
  expect(() => utilities.validateSpatialWeightsCaptures(captures(0, 0))).not.toThrow();
});

it('R1 forest capture sizes its geometry and key spaces from published edgeCount', () => {
  const record = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        'docs/lens-records/gpu-spatial-analysis.spatial-weights-minimum-spanning-tree.json',
        'utf8'
      )
    )
  ).record;
  expect(record.captures!.snapshots.filter(capture => capture.name === 'edgeCount')).toHaveLength(
    1
  );
  // Recaptured records already publish this count; replace it rather than adding a duplicate.
  record.captures = {
    ...record.captures!,
    snapshots: record.captures!.snapshots.map(capture =>
      capture.name === 'edgeCount' ? {...capture, values: Uint32Array.of(2)} : capture
    )
  };
  const finished = utilities.finishSpatialForestRecord(record);
  expect(finished.axes.keySpaces?.find(space => space.id === 'edge')?.size).toBe(2);
  expect(finished.axes.keySpaces?.find(space => space.id === 'edgeVertex')?.size).toBe(4);
  expect(
    finished.captures!.snapshots.find(capture => capture.name === 'treeSegments')!.values.length
  ).toBe(8);
  expect(
    finished.captures!.snapshots.find(capture => capture.name === 'treeSegmentOffsets')!.values
      .length
  ).toBe(3);
});
