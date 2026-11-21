// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import fixture from './fixtures/rts05-compositing-numpy.json';

it('B6 serializes compositing fixture sections in generator order', () => {
  const source = readFileSync(new URL('./rts05-compositing-numpy.py', import.meta.url), 'utf8');
  const publication = source.slice(source.lastIndexOf('fixture = {'));
  const keys = [...publication.matchAll(/^\s*'(\w+)':/gm)].map(match => match[1]);
  expect(Object.keys(fixture)).toEqual(keys);
});

it('TWN-02 records earliest scalar medoids and their selected values', () => {
  // For four scalar observations, the absolute-distance objective is constant
  // between the two central order statistics. Pick the earliest usable slice.
  expect(fixture.scalarMedoid.bandCount).toBe(1);
  expect(fixture.scalarMedoid.cells.map(cell => [cell.expectedSlice, cell.expectedValue])).toEqual([
    [1, 0.2],
    [1, 0.3],
    [1, 0.2],
    [0, 0.2],
    [2, 0.3]
  ]);
  expect(fixture.scalarMedoid.cells.map(cell => cell.values)).toEqual([
    [0.1, 0.2, 0.3, 0.4],
    [0.1, 0.3, 0.2, 0.4],
    [0.4, 0.2, 0.2, 0.1],
    [0.2, 0.1, 0.3, 0.2],
    [0.2, 0.1, 0.3, 0.4]
  ]);
  expect(fixture.scalarMedoid.cells[4].validity).toEqual([0, 1, 1, 1]);
});

it('TWN-06 records exact singleton and identical-stack geometric medians with zero last step', () => {
  const reference = fixture.geometricMedianIdentity;
  expect(reference.iterationCount).toBe(1);
  expect(reference.distanceEpsilon).toBe(Math.fround(1e-5));
  expect(reference.cells.map(cell => cell.validity)).toEqual([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
    [1, 1, 1]
  ]);
  // Sum of distances is exactly zero at the retained point, so it cannot move.
  expect(reference.cells.map(cell => [cell.expected, cell.expectedLastStep])).toEqual([
    [[7], 0],
    [[7], 0],
    [[456787.5], 0],
    [[456787.5], 0],
    [[0.1], 0],
    [[0.1], 0]
  ]);
  expect(reference.cells.map(cell => cell.values)).toEqual([
    [20, 7, -16],
    [7, 7, 7],
    [456800.5, 456787.5, 456764.5],
    [456787.5, 456787.5, 456787.5],
    [13.1, 0.1, -22.9],
    [0.1, 0.1, 0.1]
  ]);
});
