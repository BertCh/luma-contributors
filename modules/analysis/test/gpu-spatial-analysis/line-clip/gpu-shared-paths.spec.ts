// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {runSharedPaths, type Point} from './line-clip-harness';
import {SHAPELY_SHARED_PATHS_FIXTURE} from './shapely-shared-paths-fixture';
import {requireTestDevice} from '../../utils/require-test-device';

function getLength(run: readonly (readonly number[])[]): number {
  let total = 0;
  for (let index = 1; index < run.length; index++) {
    total += Math.hypot(run[index][0] - run[index - 1][0], run[index][1] - run[index - 1][1]);
  }
  return total;
}

it('GPUSharedPaths#reports forward and backward runs along the left line', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runSharedPaths(
    device,
    [
      [
        [0, 0],
        [2, 0],
        [4, 0]
      ],
      [
        [0, 3],
        [4, 3]
      ]
    ],
    [
      // Runs backward over [1, 3] of line 0, across its middle vertex.
      [
        [3, 0],
        [1, 0]
      ],
      // Forward over [1, 4] of line 1; the end beyond the left line is not shared.
      [
        [1, 3],
        [5, 3]
      ],
      // Crosses line 0 at a single point: no path.
      [
        [2, -1],
        [2, 1]
      ]
    ]
  );
  expect(result.count).toBe(2);
  expect(result.leftSourceIds).toEqual([0, 1]);
  expect(result.rightSourceIds).toEqual([0, 1]);
  expect(result.forward).toEqual([0, 1]);
  expect(result.runs).toEqual([
    [
      [1, 0],
      [2, 0],
      [3, 0]
    ],
    [
      [1, 3],
      [4, 3]
    ]
  ]);
  expect(result.overflow).toBe(0);
});

it('GPUSharedPaths#matches Shapely shared_paths on lattice walks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {left, right, shared} = SHAPELY_SHARED_PATHS_FIXTURE;
  const start = performance.now();
  const result = await runSharedPaths(device, left as never, right as never);
  const elapsed = performance.now() - start;
  expect(result.overflow).toBe(0);
  const used = new Set<number>();
  let forwardLength = 0;
  let backwardLength = 0;
  for (const reference of shared) {
    for (const [direction, expected] of [
      [1, reference.forward],
      [0, reference.backward]
    ] as const) {
      const mine: Point[][] = [];
      result.runs.forEach((run, index) => {
        if (
          result.leftSourceIds[index] === reference.left &&
          result.rightSourceIds[index] === reference.right &&
          result.forward[index] === direction
        ) {
          mine.push(run);
          used.add(index);
        }
      });
      const length = mine.reduce((sum, run) => sum + getLength(run), 0);
      expect(
        Math.abs(length - expected.length),
        `pair ${reference.left},${reference.right} direction ${direction} length`
      ).toBeLessThan(1e-4);
      expect(mine.length, `pair ${reference.left},${reference.right} run count`).toBe(
        expected.pieces.length
      );
      const unused = new Set(expected.pieces.map((_, index) => index));
      for (const run of mine) {
        const ends = [run[0], run[run.length - 1]];
        const match = [...unused].find(index => {
          const other = expected.pieces[index];
          const first = other[0];
          const last = other[other.length - 1];
          const same = (a: readonly number[], b: readonly number[]) =>
            a[0] === b[0] && a[1] === b[1];
          return (
            (same(ends[0], first) && same(ends[1], last)) ||
            (same(ends[0], last) && same(ends[1], first))
          );
        });
        expect(match).toBeDefined();
        unused.delete(match as number);
      }
      if (direction === 1) {
        forwardLength += length;
      } else {
        backwardLength += length;
      }
    }
  }
  // No run for a pair Shapely reports as empty.
  expect(used.size).toBe(result.count);
  console.log(
    `GPUSharedPaths: ${shared.length} pairs, ${result.count} runs, forward ${forwardLength.toFixed(3)}, ` +
      `backward ${backwardLength.toFixed(3)}, ${elapsed.toFixed(0)} ms`
  );
});

it('GPUSharedPaths#reports overflow and keeps a consistent prefix', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const left = [
    [
      [0, 0],
      [10, 0]
    ],
    [
      [0, 1],
      [10, 1]
    ]
  ];
  const right = [
    [
      [2, 0],
      [4, 0]
    ],
    [
      [2, 1],
      [4, 1]
    ]
  ];
  const limited = await runSharedPaths(device, left, right, {runCapacity: 1});
  expect(limited.count).toBe(1);
  expect(limited.requiredCount).toBe(2);
  expect(limited.overflow).toBe(1);
  // Run capacity is final: no candidate overflow.
  expect(limited.candidateOverflow).toBe(0);
  const pairs = await runSharedPaths(device, left, right, {intersectionCapacity: 1});
  expect(pairs.overflow).toBe(1);
  // The pair list is intermediate, so rule 9 reports it as candidateOverflow as well.
  expect(pairs.candidateOverflow).toBe(1);
  const vertices = await runSharedPaths(device, left, right, {vertexCapacity: 3});
  expect(vertices.count).toBe(1);
  expect(vertices.overflow).toBe(1);
  expect(vertices.candidateOverflow).toBe(0);
});

it('GPUSharedPaths#compiles within the default storage buffer limit', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const result = await runSharedPaths(
    device,
    [
      [
        [0, 0],
        [4, 0]
      ]
    ],
    [
      [
        [1, 0],
        [3, 0]
      ]
    ]
  );
  expect(result.runs).toEqual([
    [
      [1, 0],
      [3, 0]
    ]
  ]);
});

it('GPUSharedPaths#sizes and writes long chains and shared tails without walking them', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A 600 vertex road shared with a right line that splits every left segment in two and with a
  // second right line that runs it backward over 300 long segments: two chains of 1,198 spans.
  const left = [Array.from({length: 600}, (_, index) => [index, 0])];
  const right = [
    [[0, 0], ...Array.from({length: 599}, (_, index) => [index + 0.5, 0]), [599, 0]],
    [[599, 0], ...Array.from({length: 300}, (_, index) => [598 - 2 * index, 0])]
  ];
  const result = await runSharedPaths(device, left, right, {
    intersectionCapacity: 8192,
    runCapacity: 8,
    vertexCapacity: 4096
  });
  expect(result.overflow).toBe(0);
  expect(result.count).toBe(2);
  expect(result.rightSourceIds).toEqual([0, 1]);
  expect(result.forward).toEqual([1, 0]);
  expect(result.vertexCount).toBe(1200);
  for (const run of result.runs) {
    expect(run.map(point => point[0])).toEqual(Array.from({length: 600}, (_, index) => index));
  }
  // A right line that retraces itself starts two runs that share their tail; each run keeps it.
  const retraced = await runSharedPaths(
    device,
    [
      [
        [0, 0],
        [2, 0],
        [4, 0]
      ]
    ],
    [
      [
        [0, 0],
        [1, 0],
        [3, 0],
        [1, 0],
        [4, 0]
      ]
    ],
    {intersectionCapacity: 64}
  );
  expect(retraced.overflow).toBe(0);
  expect(retraced.count).toBeGreaterThanOrEqual(1);
  expect(retraced.vertexCount).toBe(retraced.runs.reduce((sum, run) => sum + run.length, 0));
});
