// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  getArcLabelAnchor,
  getLineLabelAnchor,
  getMultiPolygonLabelAnchor,
  getPolygonLabelAnchor
} from '../../../examples/deck/kepler-compute/src/label-anchors';
import {
  packAcceptedLabelRows,
  placeLabelsGreedy
} from '../../../examples/deck/kepler-compute/src/label-collision';
import {
  DEFAULT_TIMESTAMP_LABEL_FORMAT,
  getTimestampMilliseconds,
  isTimestampLabelFieldType
} from '../../../examples/deck/kepler-compute/src/label-timestamp';

describe('label collision', () => {
  it('keeps the highest priority label and drops overlapping ones', () => {
    const accepted = placeLabelsGreedy(
      [
        {rowIndex: 0, x: 0, y: 0, width: 50, height: 10, priority: 1},
        {rowIndex: 1, x: 20, y: 2, width: 50, height: 10, priority: 5},
        {rowIndex: 2, x: 200, y: 0, width: 50, height: 10, priority: 0}
      ],
      {padding: 0}
    );
    expect(accepted).toEqual([1, 2]);
  });

  it('is bounded and breaks ties by row index', () => {
    const candidates = Array.from({length: 10}, (_, index) => ({
      rowIndex: index,
      x: index * 100,
      y: 0,
      width: 10,
      height: 10,
      priority: 0
    }));
    expect(placeLabelsGreedy(candidates, {maximumCandidates: 3})).toEqual([0, 1, 2]);
  });

  it('detects overlaps across spatial hash cells and packs bits', () => {
    const accepted = placeLabelsGreedy(
      [
        {rowIndex: 0, x: 60, y: 0, width: 20, height: 10, priority: 2},
        {rowIndex: 1, x: 70, y: 5, width: 20, height: 10, priority: 1}
      ],
      {padding: 0, cellSize: 64}
    );
    expect(accepted).toEqual([0]);
    expect([...packAcceptedLabelRows(33, [0, 32])]).toEqual([1, 1]);
  });
});

describe('label anchors', () => {
  it('finds the line midpoint by length', () => {
    expect(
      getLineLabelAnchor([
        [0, 0],
        [10, 0],
        [10, 10]
      ])
    ).toEqual([10, 0]);
    expect(
      getLineLabelAnchor([
        [0, 0],
        [4, 0]
      ])
    ).toEqual([2, 0]);
    expect(getLineLabelAnchor([])).toBeNull();
  });

  it('finds the great-circle arc midpoint', () => {
    const [longitude, latitude] = getArcLabelAnchor([0, 0], [90, 0]);
    expect(longitude).toBeCloseTo(45);
    expect(latitude).toBeCloseTo(0);
    const [, northLatitude] = getArcLabelAnchor([-60, 60], [60, 60]);
    expect(northLatitude).toBeGreaterThan(60);
  });

  it('finds polygon centroids and the largest multipolygon part', () => {
    const square = [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4],
        [0, 0]
      ]
    ];
    const anchor = getPolygonLabelAnchor(square)!;
    expect(anchor[0]).toBeCloseTo(2);
    expect(anchor[1]).toBeCloseTo(2);
    const small = [
      [
        [10, 10],
        [11, 10],
        [11, 11],
        [10, 11],
        [10, 10]
      ]
    ];
    expect(getMultiPolygonLabelAnchor([small, square])![0]).toBeCloseTo(2);
    expect(
      getPolygonLabelAnchor([
        [
          [1, 1],
          [3, 1],
          [1, 1]
        ]
      ])
    ).toEqual([2, 1]);
  });
});

describe('timestamp labels', () => {
  it('uses the native date-time tooltip format and normalizes cells', () => {
    expect(DEFAULT_TIMESTAMP_LABEL_FORMAT).toBe('L LTS');
    expect(isTimestampLabelFieldType('Timestamp')).toBe(true);
    expect(isTimestampLabelFieldType('date')).toBe(false);
    expect(getTimestampMilliseconds(new Date(5))).toBe(5);
    expect(getTimestampMilliseconds(7n)).toBe(7);
    expect(getTimestampMilliseconds(null)).toBeNull();
    expect(getTimestampMilliseconds(Number.NaN)).toBeNull();
  });
});
