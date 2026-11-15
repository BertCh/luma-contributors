// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {existsSync, readFileSync, statSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import earthquake from '../../oracles/fixtures/parity-wp6-earthquake-input.json';
import groupStatistics from '../../oracles/fixtures/parity-wp6-group-statistics.json';
import temporal from '../../oracles/fixtures/parity-wp6-temporal-reduction.json';
import histogram from '../../oracles/fixtures/parity-wp6-histogram.json';
import {decodeParityPayload} from '../../oracles/parity-wp6-reduction-harness';

const root = resolve(import.meta.dirname, '../../../../..');
const fixtureDirectory = resolve(root, 'modules/analysis/test/oracles/fixtures');
const digest = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
const SOURCE_DIRECTORY = 'examples/deck/spatial-analysis-showcase/public/data';

// Manifests are provenance metadata, not oracle inputs. Pin their recorded hashes even when
// the optional example tree is absent; compare the actual files as well in a full checkout.
const MANIFEST_HASHES: Record<string, string> = {
  [`${SOURCE_DIRECTORY}/chicago-tracts/manifest.json`]:
    '8dc3c174c05d3a8e47dcf6a815920948e6d2c699e1c720e9f7821ec4faec7b84',
  [`${SOURCE_DIRECTORY}/poopdeck-earthquakes/manifest.json`]:
    'c0cf0b19912c95a087982bb4d09bb3d58e4476103fd38c9040cfe63e6cfc215f'
};

/** Reconstruct every original binary input from the committed lossless fixture data. */
function getEmbeddedSources(): Map<string, Uint8Array> {
  const chicago = groupStatistics.cases.find(scene => scene.name === 'chicago-all-tracts-compact');
  if (!chicago) {
    throw new Error('Group statistics fixture needs the complete Chicago input');
  }
  const floatBytes = (values: (number | string | null)[]) => {
    const bytes = Buffer.alloc(values.length * 4);
    values.forEach((value, row) =>
      bytes.writeFloatLE(value === null ? NaN : Number(value), row * 4)
    );
    return bytes;
  };
  const cells = decodeParityPayload(earthquake.cells);
  expect(cells.every(value => Number.isInteger(value) && value >= 0 && value <= 255)).toBe(true);
  return new Map<string, Uint8Array>([
    [`${SOURCE_DIRECTORY}/chicago-tracts/communityArea.bin`, Uint8Array.from(chicago.keys)],
    [`${SOURCE_DIRECTORY}/chicago-tracts/perCapitaIncome.bin`, floatBytes(chicago.columns[0])],
    [`${SOURCE_DIRECTORY}/chicago-tracts/population.bin`, floatBytes(chicago.columns[1])],
    [
      `${SOURCE_DIRECTORY}/poopdeck-earthquakes/magnitude.bin`,
      Buffer.from(earthquake.magnitude.base64, 'base64')
    ],
    [
      `${SOURCE_DIRECTORY}/poopdeck-earthquakes/time.bin`,
      Buffer.from(earthquake.seconds.base64, 'base64')
    ],
    // The original categories are uint8; the GPU fixture widens them to little-endian uint32.
    [`${SOURCE_DIRECTORY}/poopdeck-earthquakes/mag_band.bin`, Uint8Array.from(cells)]
  ]);
}

it('WP6 reduction references retain versions, original source hashes and bounded fixture sizes', () => {
  const embeddedSources = getEmbeddedSources();
  for (const [topic, fixture] of Object.entries({
    'earthquake-input': earthquake,
    'group-statistics': groupStatistics,
    'temporal-reduction': temporal,
    histogram
  })) {
    expect(
      statSync(resolve(fixtureDirectory, `parity-wp6-${topic}.json`)).size
    ).toBeLessThanOrEqual(1500000);
    for (const version of Object.values(fixture.versions)) expect(version).toMatch(/^\d+\./);
    expect(fixture.generator).toBe('parity-wp6-dataframe-reductions.py');
    for (const [source, hash] of Object.entries(fixture.sources)) {
      const embedded = embeddedSources.get(source);
      const expectedHash = embedded ? digest(embedded) : MANIFEST_HASHES[source];
      expect(
        expectedHash,
        `${source} must have embedded inputs or pinned provenance`
      ).toBeDefined();
      expect(hash, `${topic}: ${source}`).toBe(expectedHash);
      // Keep byte-for-byte source verification when examples are included, while the committed
      // inputs above make the analysis package's oracle checks independent of example assets.
      const sourcePath = resolve(root, source);
      if (existsSync(sourcePath)) {
        expect(digest(readFileSync(sourcePath)), source).toBe(hash);
      }
    }
  }
});

it('WP6 full USGS payloads preserve every source row and little-endian byte', () => {
  expect(earthquake.rowCount).toBe(77231);
  for (const payload of [earthquake.magnitude, earthquake.seconds, earthquake.cells]) {
    const bytes = Buffer.from(payload.base64, 'base64');
    expect(bytes.length).toBe(earthquake.rowCount * 4);
    expect(digest(bytes)).toBe(payload.sha256);
    expect(decodeParityPayload(payload)).toEqual(
      Array.from({length: earthquake.rowCount}, (_, row) =>
        payload.dtype === '<f4' ? bytes.readFloatLE(row * 4) : bytes.readUInt32LE(row * 4)
      )
    );
  }
});

it('WP6 fixtures contain complete output arrays, retained capacities and source-row scores', () => {
  for (const scene of groupStatistics.cases) {
    expect(scene.expected.keys).toHaveLength(scene.capacity * (scene.keyBits / 32));
    expect(scene.expected.counts).toHaveLength(scene.capacity);
    for (const columns of Object.values(scene.expected.columns)) {
      for (const column of columns) {
        for (const [field, entries] of Object.entries(column)) {
          expect(entries, `${scene.name}/${field}`).toHaveLength(
            field === 'zScores'
              ? scene.mask.length
              : scene.capacity *
                  (field === 'sums' ? 2 : field === 'percentiles' ? scene.fractions.length : 1)
          );
        }
      }
    }
  }
  for (const scene of temporal.cases) {
    for (const field of ['counts', 'minimums', 'maximums', 'first', 'last'] as const) {
      expect(scene.expected[field]).toHaveLength(scene.cellCount * scene.bucketCount);
    }
    expect(scene.expected.ids).toHaveLength(scene.expected.status[0]);
    if (scene.source)
      expect(scene.expected.counts.reduce((total, count) => total + count, 0)).toBe(
        earthquake.rowCount
      );
  }
  for (const scene of histogram.cases) {
    expect(scene.expected).toHaveLength(scene.bins);
    if (scene.source)
      expect(scene.expected.reduce((total, count) => total + count, 0)).toBe(earthquake.rowCount);
  }
});

it('WP6 int64 temporal references retain nanosecond differences beyond JS integer precision', () => {
  const relative = temporal.cases.find(scene => scene.name === 'edges-float-0')!;
  const nanoseconds = temporal.cases.find(
    scene => scene.name === 'edges-word-1577836800000000000'
  )!;
  expect(nanoseconds.expected).toEqual(relative.expected);
  const firstTimestamp = nanoseconds.timestamps[0];
  const secondTimestamp = nanoseconds.timestamps[1];
  if (typeof firstTimestamp !== 'string' || typeof secondTimestamp !== 'string') {
    throw new Error('Int64 fixture timestamps must remain decimal strings');
  }
  expect(BigInt(secondTimestamp) - BigInt(firstTimestamp)).toBe(2n);
  expect(Number(nanoseconds.timestamps[1])).toBe(Number(nanoseconds.timestamps[0]));
});
