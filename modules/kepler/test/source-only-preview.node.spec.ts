// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {createRequire} from 'node:module';
import {expect, test, vi} from 'vitest';
import {SourcePreviewPanel} from '../../../examples/deck/kepler-compute/src/source-preview-panel';
import {previewArrowSource} from '../../../examples/deck/kepler-compute/src/source-preview';

// Match the example's React version rather than the repository toolchain's renderer.
const hostRequire = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {createElement} = hostRequire('react') as typeof import('react');
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');

for (const role of ['h3', 'geometry']) {
  test(`${role}-only source preview preserves retained row counts and batches without scalar normalization`, () => {
    const values =
      role === 'h3'
        ? arrow.vectorFromArray(
            [
              null,
              BigInt('0x8928308280fffff'),
              BigInt('0x8928308280bffff'),
              null,
              BigInt('0x89283082807ffff'),
              BigInt('0x89283082803ffff')
            ],
            new arrow.Uint64()
          )
        : arrow.vectorFromArray(
            [
              null,
              '{"type":"Point","coordinates":[0,0]}',
              null,
              null,
              '{"type":"Point","coordinates":[1,1]}',
              null
            ],
            new arrow.Utf8()
          );
    const original = new arrow.Table({[role]: values});
    const source = original.slice(0, 3).concat(original.slice(3));
    // Accessing role values here would accidentally attempt scalar preparation again.
    vi.spyOn(source, 'getChild').mockImplementation(() => {
      throw new Error('Source role passed to scalar normalization');
    });
    const preview = previewArrowSource(source, {columns: []}, [role]);
    expect(preview.rowCount).toBe(6);
    expect(preview.batchRowCounts).toEqual([3, 3]);
    expect(preview.fields).toEqual([]);
    expect(preview.canUpload).toBe(true);
    expect(source.schema.fields.map(field => field.name)).toEqual([role]);
    const html = renderToStaticMarkup(
      createElement(SourcePreviewPanel, {
        source,
        label: role,
        sourceOnlyColumns: [role],
        onPublish: async () => {}
      })
    );
    expect(html).toContain('6 rows · 2 source batches');
    expect(html).toContain('No scalar fields. Original source rows and role data are retained.');
    expect(html).not.toContain('type="checkbox"');
  });
}

test('source role exclusion preserves scalar choices and cannot silently normalize the role', () => {
  const source = new arrow.Table({
    h3: arrow.makeVector(new BigUint64Array([BigInt('0x8928308280fffff')])),
    amount: arrow.makeVector(new Float32Array([5]))
  });
  const preview = previewArrowSource(source, {columns: ['amount']}, ['h3']);
  expect(preview.fields.map(field => field.name)).toEqual(['amount']);
  expect(preview.canUpload).toBe(true);
  expect(() => previewArrowSource(source, {columns: ['h3']}, ['h3'])).toThrow(
    /cannot be scalar-normalized/
  );
  const html = renderToStaticMarkup(
    createElement(SourcePreviewPanel, {
      source,
      label: 'mixed',
      sourceOnlyColumns: ['h3'],
      onPublish: async () => {}
    })
  );
  expect(html.match(/type="checkbox"/g)).toHaveLength(1);
  expect(html).toContain('amount');
  expect(html).not.toContain('h3 (Uint64)');
});
