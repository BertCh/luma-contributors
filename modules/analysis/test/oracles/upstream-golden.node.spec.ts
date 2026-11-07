// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import manifest from './upstream-golden/manifest.json';
import changes from './fixtures/pdal-m3c2-upstream-golden.json';
import circuits from './fixtures/circuitscape-upstream-golden.json';

it('primary upstream fixture bytes match the pinned provenance manifest', () => {
  for (const file of manifest.files) {
    const contents = readFileSync(new URL(`./upstream-golden/${file.local}`, import.meta.url));
    expect(createHash('sha256').update(contents).digest('hex'), file.path).toBe(file.sha256);
    expect(file.url).toContain(`/${file.commit}/`);
  }
});
it('M3C2 expected values are upstream golden columns, with every row accounted for', () => {
  const lines = readFileSync(
    new URL('./upstream-golden/PDAL/autzen-bmx-m3c2.txt', import.meta.url),
    'utf8'
  )
    .trim()
    .split(/\r?\n/);
  const names = lines[0].replaceAll('"', '').split(',');
  const rows = lines.slice(1).map(line => line.split(',').map(Number));
  expect(changes.cores.length + changes.excluded.length).toBe(rows.length);
  expect(new Set([...changes.cores, ...changes.excluded].map(row => row.row)).size).toBe(
    rows.length
  );
  for (const core of changes.cores) {
    const row = rows[core.row];
    expect(core.distance).toBe(row[names.indexOf('m3c2_distance')]);
    expect(core.uncertainty).toBe(row[names.indexOf('m3c2_uncertainty')]);
    expect(core.counts).toEqual([
      row[names.indexOf('m3c2_count1')],
      row[names.indexOf('m3c2_count2')]
    ]);
  }
});
it('Circuitscape expected voltages and currents are copied from upstream golden files', () => {
  const read = (name: string) =>
    readFileSync(new URL(`./upstream-golden/Circuitscape.jl/${name}`, import.meta.url), 'utf8')
      .trim()
      .split(/\r?\n/)
      .map(line => line.trim().split(/\s+/).map(Number));
  for (const pair of circuits.pairs) {
    for (const [actual, expected] of [
      [pair.voltages, read(`sgNetworkVerify2_voltages_${pair.sink}_${pair.source}.txt`)],
      [
        pair.absoluteCurrents,
        read(`sgNetworkVerify2_branch_currents_${pair.sink}_${pair.source}.txt`)
      ]
    ]) {
      expect(actual.length).toBe(expected.length);
      for (let row = 0; row < actual.length; row++)
        for (let column = 0; column < actual[row].length; column++) {
          // Python JSON serialization and JS numeric-literal import may differ by one binary64 ulp.
          expect(Math.abs(actual[row][column] - expected[row][column])).toBeLessThanOrEqual(
            Number.EPSILON *
              Math.max(Math.abs(actual[row][column]), Math.abs(expected[row][column]))
          );
        }
    }
  }
});
