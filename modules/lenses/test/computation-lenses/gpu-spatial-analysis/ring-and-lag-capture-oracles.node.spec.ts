// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {fromRecordedLensFile} from '../../../src/computation-lenses/registry';
import {assertRingCapture, assertLagMLCapture} from './ring-and-lag-capture-assertions';
import type {LensRecord} from '../../../src/computation-views/registry/view-registration';

function readRecord(slug: string) {
  return fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(
          `../../../../../docs/lens-records/gpu-spatial-analysis.${slug}.json`,
          import.meta.url
        ),
        'utf8'
      )
    )
  ).record;
}

function capture(record: LensRecord, name: string) {
  return record.captures!.snapshots.find(snapshot => snapshot.name === name)!;
}

test('ring capture oracle rejects uncancelled input polygons even when all close and conserve area', () => {
  const record = readRecord('segment-ring-assembly');
  assertRingCapture(record);
  const positions = capture(record, 'positions').values;
  const offsets = capture(record, 'areaRingOffsets').values;
  const normalized: number[] = [];
  const areas: number[] = [];
  for (let ring = 0; ring < offsets.length - 1; ring++) {
    const vertices = Array.from(
      {length: Number(offsets[ring + 1]) - Number(offsets[ring])},
      (_, vertex) => [
        Number(positions[2 * (Number(offsets[ring]) + vertex)]),
        Number(positions[2 * (Number(offsets[ring]) + vertex) + 1])
      ]
    );
    const signedArea =
      vertices
        .slice(0, -1)
        .reduce(
          (sum, vertex, index) =>
            sum + vertex[0] * vertices[index + 1][1] - vertices[index + 1][0] * vertex[1],
          0
        ) / 2;
    if (signedArea < 0) vertices.reverse();
    normalized.push(...vertices.flat());
    areas.push(Math.abs(signedArea));
  }
  capture(record, 'ringCount').values = [areas.length];
  capture(record, 'ringPositions').values = normalized;
  capture(record, 'ringOffsets').values = Array.from(offsets);
  capture(record, 'ringAreas').values = areas;
  expect(areas).toHaveLength(17);
  expect(() => assertRingCapture(record)).toThrow();
});

test('ring capture oracle rejects an area-preserving deformation of the union boundary', () => {
  const record = readRecord('segment-ring-assembly');
  const positions = capture(record, 'ringPositions');
  // Determinant-one affine map keeps signed area and closure but changes the perimeter.
  positions.values = Array.from(
    positions.values,
    (value, index) => Number(value) * (index % 2 ? 0.5 : 2)
  );
  expect(() => assertRingCapture(record)).toThrow();
});

test('lag ML capture oracle rejects a shifted coefficient-key window', () => {
  const record = readRecord('spatial-lag-maximum-likelihood');
  assertLagMLCapture(record);
  const estimates = capture(record, 'estimates');
  estimates.meaning = `${estimates.meaning ?? ''} first:1`;
  expect(() => assertLagMLCapture(record)).toThrow();
});
