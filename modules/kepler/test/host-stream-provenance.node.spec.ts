// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getPopulationLabel,
  getStreamImportOptions,
  parseRemoteStreamUrl,
  parseStreamProvenance,
  serializeStreamProvenance
} from '../../../examples/deck/kepler-compute/src/remote-stream-source';

test('stream provenance round trips remote origin and partial counts and rejects malformed manifests', () => {
  const provenance = {
    origin: {kind: 'remote' as const, url: 'https://data.example/a.arrows'},
    partial: {rowCount: 5, batchCount: 2}
  };
  expect(parseStreamProvenance(serializeStreamProvenance(provenance))).toEqual(provenance);
  expect(parseStreamProvenance(undefined)).toBeUndefined();
  expect(() => parseStreamProvenance({partial: {rowCount: 5, batchCount: 0}})).toThrow();
  expect(() => parseStreamProvenance({origin: {kind: 'remote', url: 'file:///x'}})).toThrow();
  expect(() => parseStreamProvenance([])).toThrow();
  expect(getStreamImportOptions(provenance).population).toMatchObject({
    state: 'complete',
    partial: provenance.partial
  });
});

test('remote stream URLs must be http(s) without credentials', () => {
  expect(parseRemoteStreamUrl('https://data.example/a.arrows').href).toContain('data.example');
  expect(() => parseRemoteStreamUrl('https://user:pass@data.example/a')).toThrow();
  expect(() => parseRemoteStreamUrl('data:text/plain,1', undefined)).toThrow();
});

test('partial populations get one idempotent visible label and finite sources are unchanged', () => {
  const population = {
    kind: 'stream' as const,
    streamId: 's',
    state: 'complete' as const,
    partial: {rowCount: 1200, batchCount: 3}
  };
  const label = getPopulationLabel('Trips', population);
  expect(label).toBe('Trips (partial: 1,200 of ? rows, 3 batches)');
  expect(getPopulationLabel(label, population)).toBe(label);
  expect(getPopulationLabel('Trips', undefined)).toBe('Trips');
});
