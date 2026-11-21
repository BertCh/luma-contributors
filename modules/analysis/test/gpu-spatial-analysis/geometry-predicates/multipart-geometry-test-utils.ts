// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import type {GPUGeometryPredicatesGeometry} from '../../../src/gpu-spatial-analysis/geometry-predicates/index';
import FIXTURE from './multipart-geometry-fixture.json';

export {FIXTURE};
export const MULTIPART_COLUMNS = [
  'isSimple',
  'isRing',
  'isClosed',
  'isCcw',
  'numPoints',
  'numCoordinates',
  'numInteriorRings',
  'numGeometries'
] as const;
export type MultipartRecord = (typeof FIXTURE.lines)[number];

/** Assert the fixture actually uploads opposite IEEE-754 zero signs before equality runs. */
export function expectSignedZeroUpload(
  records: readonly {name: string; left: number[][][]; right: number[][][]}[],
  operand: 'left' | 'right',
  values: Float32Array | Uint32Array
): void {
  const row = records.findIndex(record => record.name === 'negative zero');
  if (row < 0) return;
  const offset = records
    .slice(0, row)
    .reduce((count, record) => count + record[operand].flat(2).length, 0);
  const bits = new Uint32Array(values.buffer, values.byteOffset, values.byteLength / 4);
  expect(bits[offset + (operand === 'left' ? 0 : 1)], `${operand} negative-zero input bits`).toBe(
    0x80000000
  );
  expect(bits[offset + (operand === 'left' ? 1 : 0)], `${operand} positive-zero input bits`).toBe(
    0
  );
}

export function createMultipartGeometry(
  kind: 'points' | 'lines',
  features: number[][][][],
  upload: <Format extends 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => GraphDataView<Format>
): Extract<GPUGeometryPredicatesGeometry, {kind: 'points' | 'lines'}> {
  const positions: number[] = [];
  const featureOffsets = [0];
  const lineOffsets = [0];
  let partCount = 0;
  for (const feature of features) {
    for (const part of feature) {
      for (const point of part) positions.push(...point);
      lineOffsets.push(positions.length / 2);
      partCount++;
    }
    featureOffsets.push(partCount);
  }
  const common = {
    positions: upload('positions', new Float32Array(positions), 'float32x2'),
    featureOffsets: upload('features', new Uint32Array(featureOffsets), 'uint32')
  };
  return kind === 'points'
    ? {kind, ...common}
    : {
        kind,
        ...common,
        lineOffsets: upload('lines', new Uint32Array(lineOffsets), 'uint32')
      };
}

export function getMultipartMeasureLayout(geometry: ReturnType<typeof createMultipartGeometry>) {
  return geometry.kind === 'lines'
    ? {...geometry, geometryType: 'lines' as const}
    : {...geometry, geometryType: 'points' as const};
}

/** All expected values are Shapely literals, produced by multipart-geometry.oracle.py. */
export function expectMultipartPredicates(
  records: MultipartRecord[],
  columns: Record<string, number[]>,
  offsets: number[],
  positions: number[],
  exact: number[],
  identical: number[],
  toleranceIndex: number
): void {
  for (const column of MULTIPART_COLUMNS)
    expect(columns[column], column).toEqual(records.map(record => record.expected[column]));
  expect(exact, 'equalsExact').toEqual(records.map(record => record.equalsExact[toleranceIndex]));
  expect(identical, 'equalsIdentical').toEqual(records.map(record => record.equalsIdentical));
  let expectedOffset = 0;
  const expectedOffsets = [0];
  for (const record of records) {
    expectedOffset += record.unique.length;
    expectedOffsets.push(expectedOffset);
  }
  expect(offsets, 'uniqueOffsets including terminal').toEqual(expectedOffsets);
  // Canonicalize signed zero only: Shapely and WGSL both merge it when extracting points.
  expect(
    positions.slice(0, expectedOffset * 2).map(value => (value === 0 ? 0 : value)),
    'uniquePositions'
  ).toEqual(
    records.flatMap(record => record.unique.flat()).map(value => (value === 0 ? 0 : value))
  );
}
