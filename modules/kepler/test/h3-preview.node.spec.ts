// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';

// This pure preview check does not ingest Kepler CPU datasets or load its browser-only utility stack.
vi.mock('../../../examples/deck/kepler-compute/src/native-dataset-source', () => ({
  makeArrowSourceFromNativeDataset: () => {
    throw new Error('Native ingestion is outside this preview test');
  }
}));
import {formatH3SourcePreview} from '../../../examples/deck/kepler-compute/src/source-workspace-panel';
import {preparePublishedH3} from '../../../examples/deck/kepler-compute/src/source-publication';

test('H3 source preview reports exact IDs separately from measured Float32 coordinate error', () => {
  const source = new arrow.Table({
    cell: arrow.vectorFromArray([null, '8928308280fffff', 'invalid'], new arrow.Utf8())
  });
  const prepared = preparePublishedH3(source, 'cell');
  const preview = formatH3SourcePreview(prepared);
  expect(preview).toContain('1 valid H3 cells; 1 missing and 1 invalid IDs');
  expect(preview).toContain('row 2:');
  expect(preview).toContain('Exact source IDs and batches retained');
  expect(preview).toContain('Geometry uses Float32 longitude/latitude');
  expect(preview).toContain(
    `boundary approximately ${prepared.precision.maximumCoordinateErrorDegrees.toExponential(3)} degrees`
  );
  expect(preview).toContain(
    `center approximately ${prepared.precision.maximumCenterErrorDegrees.toExponential(3)} degrees`
  );
  expect(prepared.precision.maximumCoordinateErrorDegrees).toBeGreaterThan(0);
  expect(preview).toContain('fp64 hi/lo center path');
});

test('valid H3 IDs with collapsed Float32 boundaries are admitted for the fp64 column path', () => {
  const source = new arrow.Table({
    cell: arrow.vectorFromArray([BigInt('0x8fdbb16cc2e02d9')], new arrow.Uint64())
  });
  const prepared = preparePublishedH3(source, 'cell');
  expect(prepared.precision.collapsedRows).toHaveLength(1);
});
