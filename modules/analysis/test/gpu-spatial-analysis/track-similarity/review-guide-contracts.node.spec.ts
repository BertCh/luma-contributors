// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  computeTrackSimilarityOnCPU,
  GPUTrackSimilaritySearch
} from '../../../src/gpu-spatial-analysis/track-similarity';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

const measuresGuide = () =>
  readFileSync('docs/api/gpu-spatial-analysis/track-similarity-measures.md', 'utf8');
const linesGuide = () => readFileSync('docs/api/gpu-spatial-analysis/operators-lines.md', 'utf8');

it('B3 documents the geographic bounds actually published for separated singletons', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const metric of [
      'hausdorff',
      'symmetric-segment-path',
      'edit-distance-on-real-sequence'
    ] as const) {
      const graph = new GPUCommandGraph(device);
      const positions = createTransientView(graph, 'positions', 'float32x2', 2);
      const offsets = createTransientView(graph, 'offsets', 'uint32', 3);
      const first = createTransientView(graph, 'first', 'uint32', 1);
      const second = createTransientView(graph, 'second', 'uint32', 1);
      const threshold = createTransientView(graph, 'threshold', 'float32', 1);
      const bound = createTransientView(graph, 'bound', 'float32', 1);
      const nodes = new GPUTrackSimilaritySearch({
        metric,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
        linesA: {kind: 'lines', positions, lineOffsets: offsets},
        pairA: first,
        pairB: second,
        threshold,
        output: {
          distance: createTransientView(graph, 'distance', 'float32', 1),
          lowerBound: bound,
          matched: createTransientView(graph, 'matched', 'uint32', 1),
          status: createTransientView(graph, 'status', 'uint32', 1)
        }
      }).getCommandNodes(graph);
      const memory = createCPUGraphMemory();
      memory.write(positions, Float32Array.of(0, 0, 90, 0));
      memory.write(offsets, Uint32Array.of(0, 1, 2));
      memory.write(first, Uint32Array.of(0));
      memory.write(second, Uint32Array.of(1));
      memory.write(threshold, Float32Array.of(1));
      executeCommandNodesOnCPU(
        nodes.filter(node => node.id.endsWith('-bound')),
        memory
      );
      // No box-distance pruning for spherical Hausdorff/SSPD; EDR count difference |1-1|=0.
      expect(Array.from(memory.readFloat32(bound))).toEqual([0]);
    }
    const table = measuresGuide()
      .split('## Lower bounds and pruned search')[1]
      ?.split('Pruning is exact:')[0];
    expect(table).toMatch(/Great-circle lower bound/);
    expect(table).toMatch(/\| Hausdorff \|[^\n]*\| `0` \|/);
    expect(table).toMatch(/\| SSPD \|[^\n]*\| `0` \|/);
    expect(table).toMatch(/\| EDR \|[^\n]*\| `abs\(n - m\)` \|/);
    expect(
      linesGuide().includes('Geographic Hausdorff and SSPD use a zero lower bound'),
      'lines guide geographic alternatives'
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('S1 qualifies agreement when virtual densified vertices differ between backends', () => {
  const result = computeTrackSimilarityOnCPU(
    {
      linesA: {
        positions: Float32Array.of(1048576, 0, 1048576.125, 4),
        lineOffsets: Uint32Array.of(0, 2)
      },
      linesB: {
        positions: Float32Array.of(1048576.125, 0, 1048576.25, 4),
        lineOffsets: Uint32Array.of(0, 2)
      },
      pairA: Uint32Array.of(0),
      pairB: Uint32Array.of(0),
      frechetEpsilon: Float32Array.of(0.125)
    },
    {
      densify: 0.5,
      editDistanceTolerance: 0.125,
      lcssTolerance: 0.125,
      measures: ['frechetWithin', 'editDistanceOnRealSequence', 'longestCommonSubsequence']
    }
  );
  // CPU exact midpoints differ by 1/8; ties-to-even f32 midpoints differ by 1/4.
  expect(Array.from(result.frechetWithin)).toEqual([1]);
  expect(Array.from(result.editDistanceOnRealSequence)).toEqual([0]);
  expect(Array.from(result.longestCommonSubsequence)).toEqual([1]);
  expect(Math.fround(1048576 + 0.125 * 0.5)).toBe(1048576);
  expect(Math.fround(1048576.125 + 0.125 * 0.5)).toBe(1048576.25);
  expect(
    measuresGuide().includes('Virtual densified vertices can differ between CPU and GPU'),
    'densification qualification'
  ).toBe(true);
});

it('B2 distinguishes the published prefix from consumer endpoint identity', () => {
  // Distinct final vertices (127/2^20 and 128/2^20) round to the same measure.
  expect(Math.fround(1024 + 127 * 2 ** -20)).toBe(1024.0001220703125);
  expect(Math.fround(1024 + 128 * 2 ** -20)).toBe(1024.0001220703125);
  expect(
    linesGuide().includes('A published measure can plateau between distinct moving vertices'),
    'prefix publication versus endpoint identity'
  ).toBe(true);
});

it('N2 documents integer planar EDR slack separately from summed-distance slack', () => {
  const guide = measuresGuide();
  expect(guide.includes('The EDR and SSPD bounds subtract'), 'no summed EDR slack claim').toBe(
    false
  );
  expect(guide).toContain('tolerance * 1.000001');
  expect(guide).toContain('The SSPD and ERP bounds');
});
