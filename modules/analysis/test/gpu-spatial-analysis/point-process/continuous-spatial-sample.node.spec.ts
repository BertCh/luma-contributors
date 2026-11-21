// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUContinuousSpatialSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('continuous designs expose intensity provenance and audit all generated line/area kernels', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32x2' | 'float32' | 'uint32' | 'uint32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `continuous-audit-${serial++}`, format, length);
  for (const type of ['line', 'area'] as const) {
    const positions = view('float32x2', 6);
    const output = {
      positions: view('float32x2', 32),
      designWeights: view('float32', 32),
      provenance: view('uint32x4', 32),
      status: view('uint32x4', 2)
    };
    const props = {
      frame:
        type === 'line'
          ? {type, positions, legacySites: view('float32', 2)}
          : {type, positions, bounds: [0, 0, 1, 1] as const, legacySites: view('float32x2', 2)},
      spacing: 0.5,
      seed: 7,
      waves: 2,
      capacity: 16,
      parameters: view('uint32', 2),
      output
    };
    const sample = new GPUContinuousSpatialSample(props);
    expect(sample.samplingIntensity).toBe(type === 'line' ? 2 : 4);
    for (const {id: identifier, source, constants} of getParsedKernelSources(
      sample.getCommandNodes(graph)
    ).configurations)
      expect(auditWGSLModule(source, constants), identifier).toEqual([]);
    expect(() => new GPUContinuousSpatialSample({...props, spacing: 0})).toThrow();
    expect(
      () => new GPUContinuousSpatialSample({...props, output: {...output, positions}})
    ).toThrow();
    expect(() => new GPUContinuousSpatialSample({...props, replicate: 0xffffffff})).toThrow();
  }
});

/** Builds one-wave props on a fresh graph; frameRows positions, legacySites legacy rows. */
function createSampleProps(
  type: 'line' | 'area',
  frameRows: number,
  legacySites: number,
  settings: {spacing: number; capacity?: number; maximumCandidates?: number},
  graph = new GPUCommandGraph(createNullWebGPUDevice())
) {
  let serial = 0;
  const view = <Format extends 'float32x2' | 'float32' | 'uint32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `continuous-bound-${serial++}`, format, length);
  const positions = view('float32x2', frameRows);
  const capacity = settings.capacity ?? 16;
  return {
    id: 'bounded-sample',
    frame:
      type === 'line'
        ? {
            type,
            positions,
            ...(legacySites ? {legacySites: view('float32', legacySites)} : {})
          }
        : {
            type,
            positions,
            bounds: [0, 0, 1, 1] as const,
            ...(legacySites ? {legacySites: view('float32x2', legacySites)} : {})
          },
    spacing: settings.spacing,
    seed: 7,
    capacity,
    maximumCandidates: settings.maximumCandidates,
    output: {
      positions: view('float32x2', capacity),
      designWeights: view('float32', capacity),
      provenance: view('uint32x4', capacity),
      status: view('uint32x4', 1)
    }
  };
}

it('PTS-112: an area lattice larger than maximumCandidates is rejected at construction', () => {
  // Bounds [0,0,1,1] at spacing 0.003: ceil(1 / 0.003) + 1 = 335 columns and rows, 335^2 = 112225
  // lattice cells, more than the default budget of 65536, so every wave would end in flag 4.
  expect(
    () => new GPUContinuousSpatialSample(createSampleProps('area', 3, 0, {spacing: 0.003}))
  ).toThrow(
    'bounded-sample maximumCandidates (65536) must be at least the area lattice of 335 x 335 = 112225 cells'
  );
  expect(
    () =>
      new GPUContinuousSpatialSample(
        createSampleProps('area', 3, 0, {spacing: 0.003, maximumCandidates: 112225})
      )
  ).not.toThrow();
  // Bounds [0,0,1,1] at spacing 0.5 is a 3 x 3 = 9 cell lattice: 9 candidates suffice, 8 do not.
  expect(
    () =>
      new GPUContinuousSpatialSample(
        createSampleProps('area', 3, 0, {spacing: 0.5, maximumCandidates: 9})
      )
  ).not.toThrow();
  expect(
    () =>
      new GPUContinuousSpatialSample(
        createSampleProps('area', 3, 0, {spacing: 0.5, maximumCandidates: 8})
      )
  ).toThrow('3 x 3 = 9 cells');
  // A line frame's arc length is per-frame data: a tiny budget still constructs and reports flag 4.
  expect(
    () =>
      new GPUContinuousSpatialSample(
        createSampleProps('line', 2, 0, {spacing: 0.003, maximumCandidates: 1})
      )
  ).not.toThrow();
});

it('PTS-111: the serial work of one wave invocation is bounded at construction', () => {
  // Work = (candidates + legacy) * (primitives + legacy * legacy-site cost + 1) + rows + capacity,
  // bounded by 2^24 = 16777216.
  // Largest validated area: 1365 triangles (4095 rows), 256 legacy sites, a 1001 x 1001 lattice
  // (bounds [0,0,1,1], spacing 0.001) under the 1048576 budget:
  // (1002001 + 256) * (1365 + 256 + 1) + 4095 + 16 = 1625664965.
  expect(
    () =>
      new GPUContinuousSpatialSample(
        createSampleProps('area', 4095, 256, {spacing: 0.001, maximumCandidates: 1048576})
      )
  ).toThrow('= 1625664965, must be at most 16777216');
  // Default budget on a 4096-row line: (65536 + 0) * (4095 + 1) + 4096 + 16 = 268439568.
  expect(
    () => new GPUContinuousSpatialSample(createSampleProps('line', 4096, 0, {spacing: 0.01}))
  ).toThrow('= 268439568, must be at most 16777216');
  // Line legacy sites are located by a segment scan: 255 segments, 1 site, budget 32768 gives
  // (32768 + 1) * (255 + 255 + 1) + 256 + 16 = 16745231, inside the bound; 2 sites give
  // (32768 + 2) * (255 + 510 + 1) + 256 + 16 = 25102092, outside it.
  expect(
    () =>
      new GPUContinuousSpatialSample(
        createSampleProps('line', 256, 1, {spacing: 0.01, maximumCandidates: 32768})
      )
  ).not.toThrow();
  expect(
    () =>
      new GPUContinuousSpatialSample(
        createSampleProps('line', 256, 2, {spacing: 0.01, maximumCandidates: 32768})
      )
  ).toThrow('= 25102092, must be at most 16777216');
  // The same largest area frame with a 0.1 spacing (11 x 11 = 121 cells) is cheap:
  // (121 + 256) * 1622 + 4095 + 16 = 615605.
  expect(
    () =>
      new GPUContinuousSpatialSample(
        createSampleProps('area', 4095, 256, {spacing: 0.1, maximumCandidates: 1048576})
      )
  ).not.toThrow();
});

it('DEF-14 continuous sampling accepts a resident-only seed and snapshots replay words', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createSampleProps(
    'line',
    2,
    0,
    {spacing: 1, maximumCandidates: 2, capacity: 2},
    graph
  );
  const sample = new GPUContinuousSpatialSample({
    ...props,
    seed: undefined,
    parameters: createTransientView(graph, 'resident-seed', 'uint32', 2)
  });
  expect(() => sample.getCommandNodes(graph)).not.toThrow();
  expect(sample.provenance.parameterSource).toBe('resident-view');
  const words = Uint32Array.of(17, 0);
  const recorded = sample.getReplayParameters(words);
  words[0] = 99;
  expect([...recorded]).toEqual([17, 0]);
  expect(() => sample.getReplayParameters()).toThrow(/parameters.*length 2/);
  expect(() => new GPUContinuousSpatialSample({...props, seed: undefined})).toThrow(
    /seed.*parameters/
  );
});
