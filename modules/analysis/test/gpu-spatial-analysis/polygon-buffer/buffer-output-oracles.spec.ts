// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createBufferOutputOracle, BUFFER_OPTIONAL_STATUS} from './buffer-output-oracles';
import {
  expectBufferNodingDiagnostics,
  expectBufferOutputOracle
} from './buffer-output-oracles.expect';

it('buffers a finite zero-length stored line according to its cap style', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capStyle of ['square', 'round', 'flat'] as const) {
    const fixture = createBufferOutputOracle(device, 'ringCount', {
      positions: [3, 7, 3, 7],
      lineOffsets: [0, 0, 2, 2],
      capStyle,
      distance: 1,
      vertexTolerance: capStyle === 'round' ? 1 / 65536 : 0.25,
      boundaryCapacity: 32,
      vertexCapacity: 33,
      intersections: 128,
      nodedSegments: 64
    });
    try {
      const result = await fixture.runGPU();
      for (const name of [
        'overflow',
        'candidateOverflow',
        'boundaryOverflow',
        'boundaryCandidateOverflow'
      ])
        expect(result[name]).toEqual([0]);
      expect(result['count']).toEqual([capStyle === 'flat' ? 0 : 1]);
      expect(result['ringCount']).toEqual([capStyle === 'flat' ? 0 : 1]);
      if (capStyle === 'flat') continue;
      const count = result['ringOffsets'][1];
      let doubledArea = 0;
      for (let row = 0; row < count; row++) {
        const next = (row + 1) % count;
        doubledArea +=
          (result['positions'][2 * row] - 3) * (result['positions'][2 * next + 1] - 7) -
          (result['positions'][2 * next] - 3) * (result['positions'][2 * row + 1] - 7);
      }
      if (capStyle === 'square') {
        expect(result['boundaryCount']).toEqual([4]);
        expect(Math.abs(doubledArea) / 2).toBe(4);
        expect(
          Array.from({length: 4}, (_, row) =>
            result['positions'].slice(2 * row, 2 * row + 2)
          ).sort()
        ).toEqual(
          [
            [2, 6],
            [4, 6],
            [4, 8],
            [2, 8]
          ].sort()
        );
      } else {
        expect(result['boundaryCount']).toEqual([32]);
        // Shapely quad_segs=8: 32-sided unit circle area. Grid 2^-16 plus f32 trigonometry
        // changes area by <1e-4; this bound is much smaller than a missing/incorrect cap.
        expect(Math.abs(Math.abs(doubledArea) / 2 - 3.121445152258052)).toBeLessThan(1e-4);
      }
    } finally {
      fixture.destroy();
    }
  }
});
it('preserves the area-five translated square buffer with leading and trailing empty parts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A length-4 vertical line, square caps and radius 0.5 give a 1 by 5 rectangle.
  // Empty parts add no points to that surface or its lattice bounds. Eight collinear/raw
  // boundary edges survive; nine positions provide their closing row during assembly.
  for (const lineOffsets of [
    [0, 2],
    [0, 0, 2],
    [0, 2, 2],
    [0, 0, 2, 2]
  ]) {
    const fixture = createBufferOutputOracle(device, 'ringCount', {
      positions: [0, 5000000, 0, 5000004],
      lineOffsets,
      capStyle: 'square',
      distance: 0.5,
      vertexTolerance: 0.5,
      boundaryCapacity: 8,
      vertexCapacity: 9
    });
    try {
      const result = await fixture.runGPU();
      for (const [name, value] of Object.entries({
        count: 1,
        ringCount: 1,
        boundaryCount: 8,
        boundaryRequiredCount: 8,
        boundaryOverflow: 0,
        boundaryCandidateOverflow: 0,
        overflow: 0,
        candidateOverflow: 0
      }))
        expect(result[name], name).toEqual([value]);
      const count = result['ringOffsets'][1];
      let doubledArea = 0;
      for (let row = 0; row < count; row++) {
        const next = (row + 1) % count;
        doubledArea +=
          result['positions'][2 * row] * (result['positions'][2 * next + 1] - 5000000) -
          result['positions'][2 * next] * (result['positions'][2 * row + 1] - 5000000);
      }
      expect(Math.abs(doubledArea) / 2).toBe(5);
    } finally {
      fixture.destroy();
    }
  }
});

it('matches the hand-derived rectangular buffer with poisoned outputs and optional diagnostics alone', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const optional of [undefined, ...BUFFER_OPTIONAL_STATUS]) {
    const fixture = createBufferOutputOracle(device, optional);
    try {
      expectBufferOutputOracle(await fixture.runGPU());
      expectBufferOutputOracle(await fixture.runGPU());
    } finally {
      fixture.destroy();
    }
  }
});

it('publishes exact retry requirements when boundary or assembly capacity truncates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const boundaryCapacity of [3, 4]) {
    const fixture = createBufferOutputOracle(device, undefined, {
      boundaryCapacity,
      vertexCapacity: 4,
      ringCounts: true
    });
    try {
      const result = await fixture.runGPU();
      // Four rectangle edges are required. Three boundary slots retain only three; with all
      // four edges a whole closed ring requires FIVE assembly positions and four cannot fit it.
      if (boundaryCapacity === 3) {
        expect(result['boundaryCount']).toEqual([3]);
        expect(result['boundaryRequiredCount']).toEqual([4]);
        expect(result['boundaryOverflow']).toEqual([1]);
      } else {
        expect(result['boundaryCount']).toEqual([4]);
        expect(result['ringCount']).toEqual([0]);
        expect(result['requiredRingCount']).toEqual([1]);
        expect(result['overflow']).toEqual([1]);
      }
    } finally {
      fixture.destroy();
    }
  }
});

it('reports invalid singleton and deduplicated parts while retaining the valid four-edge buffer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const shortPart of [
    [Number.NaN, 0],
    [Infinity, 0, Infinity, 0]
  ]) {
    const fixture = createBufferOutputOracle(device, 'uncertainCount', {
      positions: [...shortPart, 0, 0, 4, 0],
      lineOffsets: [0, shortPart.length / 2, shortPart.length / 2 + 2]
    });
    try {
      const input = fixture.columns.find(entry => entry.name === 'input')!;
      for (const invalid of [true, false]) {
        input.buffer.write(
          new Float32Array([...(invalid ? shortPart : shortPart.map(() => 0)), 0, 0, 4, 0])
        );
        const result = await fixture.runGPU();
        // Short/degenerate parts contribute no stroke; the valid segment still has four edges.
        expect(result['boundaryCount']).toEqual([4]);
        expect(result['candidateOverflow']).toEqual([invalid ? 1 : 0]);
        expect(result['boundaryCandidateOverflow']).toEqual([invalid ? 1 : 0]);
        expect(result['uncertainCount']).toEqual([0]);
        expect(result['overflow']).toEqual([0]);
      }
    } finally {
      fixture.destroy();
    }
  }
});

it('separates nonfinite input loss from predicate uncertainty and recovers on the same graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const counter of [true, false]) {
    const fixture = createBufferOutputOracle(device, counter ? 'uncertainCount' : undefined);
    try {
      const input = fixture.columns.find(entry => entry.name === 'input')!;
      for (const usable of [true, false, true]) {
        input.buffer.write(
          new Float32Array(usable ? [0, 0, 4, 0] : [Number.NaN, 0, Number.NaN, 0])
        );
        for (const entry of fixture.columns.filter(
          entry =>
            entry.output &&
            (entry.name.startsWith('boundary') ||
              entry.name === 'uncertainCount' ||
              entry.name === 'candidateOverflow')
        ))
          entry.buffer.write(entry.values);
        const actual = await fixture.runGPU();
        expectBufferNodingDiagnostics(actual, usable);
        expect(actual['candidateOverflow']).toEqual([usable ? 0 : 1]);
        expect(actual['overflow']).toEqual([0]);
        if (usable) expectBufferOutputOracle(actual);
        else expect(actual['count']).toEqual([0]);
      }
    } finally {
      fixture.destroy();
    }
  }
});
