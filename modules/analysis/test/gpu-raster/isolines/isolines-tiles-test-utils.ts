// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import type {GPUIsolinesTile} from '../../../src/raster-analysis/isolines/isolines-tile';
import {getGPUIsolinesParameterValues} from '../../../src/raster-analysis/isolines/isolines-parameters';
import fixtures from './fixtures/isolines-tiles.json';

export const tileFixtures = fixtures.cases;
export type IsolinesTileFixture = (typeof tileFixtures)[number];

export function getWholeScene(fixture: IsolinesTileFixture) {
  const grid = makeGPURasterMetadata({width: fixture.width, height: fixture.height});
  return {
    grid,
    tile: undefined as GPUIsolinesTile | undefined,
    values: Float32Array.from(fixture.values),
    mask: Uint32Array.from(fixture.mask),
    levels: Float32Array.from(fixture.levels),
    parameters: getGPUIsolinesParameterValues(
      {
        levelCount: fixture.levels.length,
        extent: fixture.extent as [number, number, number, number],
        localOrigin: fixture.origin as [number, number]
      },
      grid
    )
  };
}

/** Four cores with overlapping cells and one sample of halo on every available side. */
export function getTileScenes(fixture: IsolinesTileFixture) {
  const whole = getWholeScene(fixture);
  const columnBreaks = [0, Math.floor((fixture.width - 1) / 2), fixture.width - 1];
  const rowBreaks = [0, Math.floor((fixture.height - 1) / 2), fixture.height - 1];
  return [0, 1, 2, 3].map(index => {
    const minColumn = columnBreaks[index % 2];
    const maxColumn = columnBreaks[(index % 2) + 1];
    const minRow = rowBreaks[Math.floor(index / 2)];
    const maxRow = rowBreaks[Math.floor(index / 2) + 1];
    const columnOffset = Math.max(0, minColumn - 1);
    const rowOffset = Math.max(0, minRow - 1);
    const width = Math.min(fixture.width, maxColumn + 2) - columnOffset;
    const height = Math.min(fixture.height, maxRow + 2) - rowOffset;
    const values = new Float32Array(width * height);
    const mask = new Uint32Array(width * height);
    for (let row = 0; row < height; row++) {
      const source = (rowOffset + row) * fixture.width + columnOffset;
      values.set(whole.values.subarray(source, source + width), row * width);
      mask.set(whole.mask.subarray(source, source + width), row * width);
    }
    return {
      ...whole,
      grid: makeGPURasterMetadata({width, height}),
      values,
      mask,
      tile: {
        domainWidth: fixture.width,
        domainHeight: fixture.height,
        columnOffset,
        rowOffset,
        ownedCells: [
          minColumn - columnOffset,
          minRow - rowOffset,
          maxColumn - columnOffset,
          maxRow - rowOffset
        ] as const
      }
    };
  });
}

export type IsolinesTileScene = ReturnType<typeof getWholeScene>;
export type IsolinesRows = {
  count: ArrayLike<number>;
  segments: ArrayLike<number>;
  segmentLevels: ArrayLike<number>;
  segmentEdges: ArrayLike<number>;
};

/** Level, directed domain edges, coordinates; unique even at zero-length corner bridges. */
export function getSegmentRows(result: IsolinesRows): number[][] {
  return Array.from({length: result.count[0]}, (_, index) => [
    result.segmentLevels[index],
    result.segmentEdges[2 * index],
    result.segmentEdges[2 * index + 1],
    ...Array.from({length: 4}, (_, component) => result.segments[4 * index + component])
  ]);
}

/** Compare topology exactly, coordinates within a fixed local-frame float32 budget. */
export function expectSameSegments(actual: number[][], expected: number[][]): void {
  const order = (first: number[], second: number[]) =>
    first[0] - second[0] || first[1] - second[1] || first[2] - second[2];
  actual = [...actual].sort(order);
  expected = [...expected].sort(order);
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    expect(actual[index].slice(0, 3)).toEqual(expected[index].slice(0, 3));
    for (let coordinate = 3; coordinate < 7; coordinate++)
      expect(Math.abs(actual[index][coordinate] - expected[index][coordinate])).toBeLessThan(3e-5);
  }
}

/** Independent skimage ignores traversal direction; consume matches to catch duplicate segments. */
export function expectSkimageSegments(actual: number[][], fixture: IsolinesTileFixture): void {
  const remaining = fixture.segments.map(segment => [...segment]);
  expect(actual.length).toBe(remaining.length);
  for (const row of actual) {
    const found = remaining.findIndex(
      expected =>
        expected[0] === row[0] &&
        [row.slice(3), [row[5], row[6], row[3], row[4]]].some(coordinates =>
          coordinates.every((value, index) => Math.abs(value - expected[index + 1]) < 3e-5)
        )
    );
    expect(found, `unmatched segment ${JSON.stringify(row)}`).toBeGreaterThanOrEqual(0);
    remaining.splice(found, 1);
  }
}

/** Review regression: a resident 2x2 tile near sample 2^24, with a nearby shared world origin. */
export function getLargeOffsetScene(
  axis: 'column' | 'row',
  originSample = 16777218
): IsolinesTileScene {
  const domainWidth = axis === 'column' ? 16777220 : 2;
  const domainHeight = axis === 'row' ? 16777220 : 2;
  return {
    grid: makeGPURasterMetadata({width: 2, height: 2}),
    tile: {
      domainWidth,
      domainHeight,
      columnOffset: axis === 'column' ? 16777217 : 0,
      rowOffset: axis === 'row' ? 16777217 : 0,
      ownedCells: [0, 0, 1, 1]
    },
    values: axis === 'column' ? Float32Array.of(-1, 0, -1, -1) : Float32Array.of(-1, -1, 0, -1),
    mask: new Uint32Array(4).fill(1),
    levels: Float32Array.of(0),
    parameters: getGPUIsolinesParameterValues(
      {
        levelCount: 1,
        extent: [0, 0, domainWidth, domainHeight],
        localOrigin: axis === 'column' ? [originSample, 0] : [0, originSample]
      },
      makeGPURasterMetadata({width: domainWidth, height: domainHeight})
    )
  };
}
