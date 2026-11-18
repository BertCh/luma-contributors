// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../../../analysis/src/testing/gpu-contributor-test-utils';
import {GPUCellPyramid} from '../../../../analysis/src/gpu-spatial-analysis/cell-aggregation/index';
import {runExample} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/cell-pyramid.example';
import type {GPUComputationCapture} from '../../../../analysis/src/computation-record/index';
import {checkCellPyramidCompleteness} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/cell-display-check';

it('declares the published cell-pyramid scenario graph with candidate-loss destinations on every level', async () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const add = vi.spyOn(graph, 'add');
  try {
    const places = createTransientView(graph, 'places', 'float32x2', 2);
    const output = await runExample(graph, {places, capacities: [2, 2, 2]});
    expect(add).toHaveBeenCalledWith(expect.any(GPUCellPyramid));
    expect(output.levels.map(table => table.status.candidateOverflow?.length)).toEqual([1, 1, 1]);
  } finally {
    add.mockRestore();
    device.destroy();
  }
});

it('keeps the published cell-pyramid example region within 25 lines', () => {
  const source = readFileSync(
    new URL(
      '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/cell-pyramid.example.ts',
      import.meta.url
    ),
    'utf8'
  );
  const region = source.split('// #region example')[1].split('// #endregion example')[0].trim();
  expect(region.split('\n').length).toBeLessThanOrEqual(25);
});

it('refuses complete pyramid observations for source loss, destination loss or unavailable status', () => {
  const snapshots: GPUComputationCapture[] = Array.from({length: 3}, (_, level) =>
    ['Overflow', 'CandidateOverflow'].map(flag => ({
      id: `level${level}${flag}`,
      name: `level${level}${flag}`,
      values: [0],
      elementType: 'u32' as const,
      components: 1,
      provenance: 'gpu-readback' as const
    }))
  ).flat();
  expect(() => checkCellPyramidCompleteness(snapshots, 3)).not.toThrow();
  for (const capture of snapshots) {
    expect(() =>
      checkCellPyramidCompleteness(
        snapshots.map(entry => (entry === capture ? {...entry, values: [1]} : entry)),
        3
      )
    ).toThrow(/incomplete/);
  }
  expect(() => checkCellPyramidCompleteness(snapshots.slice(1), 3)).toThrow(/unavailable/);
});
