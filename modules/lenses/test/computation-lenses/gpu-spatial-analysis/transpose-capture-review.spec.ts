// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {runExample} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-weights-transpose.example';
import {runExample as runContiguity} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/contiguity-weights.example';
import {runExample as runSummary} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-weights-summary.example';
import {
  addLensViewSnapshot,
  readLensViewSnapshot,
  submitLensGraph
} from '../../../src/computation-lenses/lens-view-snapshot';
import {
  createSpatialRecordingGraph,
  runSpatialScenario,
  uploadSpatialView
} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-scenario-utils';
import {
  getTractWeightsCaptureViews,
  readCompleteSpatialCaptures
} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-scenario-utils';
import {WEIGHTS_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/weights';

test.each([
  ['contiguity', runContiguity],
  ['summary', runSummary]
] as const)('R3 %s refuses actual candidate-only loss at readback', async (name, run) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = createSpatialRecordingGraph(device, `candidate-loss-${name}`);
  const uploads: {destroy(): void}[] = [];
  const upload = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const resource = uploadSpatialView(graph, id, values, format);
    uploads.push(resource);
    return resource.view;
  };
  try {
    // The three-square chain has four directed edges: output capacity four cannot overflow.
    // Its two undirected adjacent pairs cannot both fit pairCapacity one, so candidate loss is one.
    const execution = await run(device, {
      graph,
      capacity: 4,
      pairCapacity: 1,
      tracts: {
        kind: 'polygons',
        positions: upload(
          'positions',
          Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1, 2, 0, 3, 0, 3, 1, 2, 1),
          'float32x2'
        ),
        ringOffsets: upload('rings', Uint32Array.of(0, 4, 8, 12), 'uint32'),
        polygonOffsets: upload('polygons', Uint32Array.of(0, 1, 2, 3), 'uint32'),
        featureOffsets: upload('features', Uint32Array.of(0, 1, 2, 3), 'uint32')
      },
      values: upload('values', Float32Array.of(1, 2, 3), 'float32'),
      searchParameters: upload('search', Float32Array.of(0), 'float32'),
      colorParameters: upload('colors', Uint32Array.of(0), 'uint32')
    });
    const views = getTractWeightsCaptureViews(execution);
    const raw = addLensViewSnapshot(graph, device, {id: 'raw', views});
    const guarded = addLensViewSnapshot(graph, device, {id: 'guarded', views});
    const compiled = await graph.compileAsync();
    try {
      submitLensGraph(device, compiled, {parameters: undefined});
      const captures = await readLensViewSnapshot(raw, {graphId: graph.id});
      expect(
        Array.from(captures.find(capture => capture.name === 'weightsOverflow')!.values)
      ).toEqual([0]);
      expect(
        Array.from(captures.find(capture => capture.name === 'weightsCandidateOverflow')!.values)
      ).toEqual([1]);
      await expect(readCompleteSpatialCaptures(guarded, {graphId: graph.id})).rejects.toThrow(
        /weightsCandidateOverflow/
      );
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const resource of uploads) resource.destroy();
  }
});

test('transpose lens captures the hand-calculated three-square chain and complete source diagnostics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = createSpatialRecordingGraph(device, 'transpose-capture-review');
  const uploads: {destroy(): void}[] = [];
  const upload = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const resource = uploadSpatialView(graph, name, values, format);
    uploads.push(resource);
    return resource.view;
  };
  try {
    // Three unit squares share vertical edges in a chain: 0--1--2.
    // Row standardisation gives W01=W21=1, W10=W12=1/2. Thus W' has
    // CSR [0,1,3,4], [1,0,2,1], [1/2,1,1,1/2]. Directly from the definitions,
    // S0=3, S1=1/2 * 4 * (3/2)^2=9/2, S2=(3/2)^2+3^2+(3/2)^2=27/2.
    const execution = await runExample(device, {
      graph,
      tracts: {
        kind: 'polygons',
        positions: upload(
          'positions',
          Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1, 2, 0, 3, 0, 3, 1, 2, 1),
          'float32x2'
        ),
        ringOffsets: upload('rings', Uint32Array.of(0, 4, 8, 12), 'uint32'),
        polygonOffsets: upload('polygons', Uint32Array.of(0, 1, 2, 3), 'uint32'),
        featureOffsets: upload('features', Uint32Array.of(0, 1, 2, 3), 'uint32')
      },
      values: upload('values', Float32Array.of(1, 2, 3), 'float32'),
      searchParameters: upload('search', Float32Array.of(0), 'float32'),
      colorParameters: upload('colors', Uint32Array.of(0), 'uint32')
    });
    const record = await runSpatialScenario({
      id: 'transpose-capture-review',
      graph,
      views: execution.views,
      encodeOptions: {parameters: undefined},
      entry: WEIGHTS_OPERATORS['GPUSpatialWeightsTranspose'],
      choices: {columnCount: 3, symmetryTolerance: 0},
      datasetVersion: 'three-unit-squares',
      seed: 1,
      axes: {},
      tags: {}
    });
    const read = (name: string) =>
      Array.from(record.captures!.snapshots.find(capture => capture.name === name)!.values);
    expect(read('weightsRowOffsets')).toEqual([0, 1, 3, 4]);
    const neighbors = read('weightsNeighbors');
    expect(neighbors.slice(0, 4)).toEqual([1, 0, 2, 1]);
    expect(neighbors.slice(4).every(value => value === 0)).toBe(true);
    const weights = read('weightsValues');
    expect(weights.slice(0, 4)).toEqual([0.5, 1, 1, 0.5]);
    expect(weights.slice(4).every(value => value === 0)).toBe(true);
    expect(read('weightsOverflow')).toEqual([0]);
    expect(read('weightsCandidateOverflow')).toEqual([0]);
    expect(read('sourceWeightsOverflow')).toEqual([0]);
    expect(read('sourceWeightsRowOffsets')).toEqual([0, 1, 3, 4]);
    // The example row-standardises its source buffer in place before transposing it.
    expect(read('sourceWeightsValues').slice(0, 4)).toEqual([1, 0.5, 0.5, 1]);
    expect(read('neighborCounts')).toEqual([1, 2, 1]);
    expect(read('counts')).toEqual([4, 4, 0, 1, 2, 0]);
    // Powers of two make all expected moments exactly representable in float32.
    expect(read('statistics')).toEqual([3, 4.5, 13.5]);
  } finally {
    for (const resource of uploads) resource.destroy();
  }
});
