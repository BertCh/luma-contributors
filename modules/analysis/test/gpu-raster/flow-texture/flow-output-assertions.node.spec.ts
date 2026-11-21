// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  expectPublicationOrder,
  expectCompactStatus,
  expectLICValues,
  expectLostPosition
} from './flow-output-assertions';

it('S1 ordering assertions reject an absent clear node', () => {
  expect(() => expectPublicationOrder([{id: 'publication-counts'}])).toThrow();
});

it('S2 compact status assertions reject trailing status words', () => {
  expect(() =>
    expectCompactStatus({count: [0, 99], overflow: [0], candidateOverflow: [1]}, [0, 0, 1])
  ).toThrow();
});

it('S2 LIC assertions reject an incorrect second pixel in both support cases', () => {
  expect(() => expectLICValues([0, 0], 1)).toThrow();
  expect(() => expectLICValues([1 / 3, NaN], 2)).toThrow();
});

it('S2 lost-position assertions reject trailing coordinates', () => {
  expect(() => expectLostPosition([NaN, NaN, 1])).toThrow();
});
