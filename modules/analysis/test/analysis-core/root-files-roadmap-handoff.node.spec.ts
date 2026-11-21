// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readdirSync, readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import {findRoadmapRows} from './roadmap-rows.test-utils';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '../../../..');

function readRepositoryFile(path: string): string {
  return readFileSync(resolve(REPOSITORY_ROOT, path), 'utf8');
}

function readRoadmapRow(
  identity: string,
  markdown = readRepositoryFile('dev-docs/roadmap.md')
): string {
  const rows = findRoadmapRows(
    markdown,
    identity,
    'Library review 2026-11-17 (LR)',
    'implementation'
  );
  expect(rows, `exact LR implementation record ${identity}`).toHaveLength(1);
  return rows[0].text;
}

function expectRoundedNumber(actual: string, expected: number): void {
  expect(Number.isFinite(Number(actual))).toBe(true);
  expect(Number.isFinite(expected)).toBe(true);
  const decimalPlaces = actual.split('.')[1]?.length ?? 0;
  expect(Number(actual)).toBe(Number(expected.toFixed(decimalPlaces)));
}

it('X9 keeps LR-02 limited to the outstanding numerical oracle regeneration', () => {
  const row = readRoadmapRow('LR-02');
  expect(row).not.toContain('unresolved memory layout');
  for (const dependency of ['GRASS', 'GDAL Python bindings']) expect(row).toContain(dependency);
  expect(row).toMatch(/\bR\b/);
  expect(row).toContain('Visibility oracle regeneration');
  expect(row).toMatch(/not regenerated numerically|still need numerical regeneration/);
  expect(row).not.toContain('CORE-12');
});

// These source and baseline checks pin current declarations, not executable boundary behavior.
it('X9 pins the root-file inventory and declared regression tile validation', () => {
  const baseline = JSON.parse(readRepositoryFile('scripts/contributor-contract-baseline.json')) as {
    'root-file': Record<string, unknown>;
  };
  for (const entry of ['pointcloud-analysis', 'geospatial-analysis']) {
    const directory = `modules/analysis/src/${entry}`;
    expect(
      readdirSync(resolve(REPOSITORY_ROOT, directory)).filter(name => name.endsWith('.ts'))
    ).toEqual(['index.ts']);
    expect(
      Object.keys(baseline['root-file']).filter(path => path.startsWith(`${directory}/`))
    ).toEqual([]);
  }
  for (const name of [
    'gpu-spatial-error-gm',
    'gpu-spatial-two-stage-least-squares',
    'gpu-spatial-weights-traces'
  ]) {
    const source = readRepositoryFile(
      `modules/analysis/src/gpu-spatial-analysis/spatial-regression/${name}.ts`
    );
    expect(source).toContain('REGRESSION_TILE_MAXIMUM_TILE_COUNT');
    expect(source).toContain(
      "validateMaximumCount(id, 'tileCount', this.tileCount, REGRESSION_TILE_MAXIMUM_TILE_COUNT)"
    );
  }
});

// Roadmap assertions are editorial pins and do not qualify an enforced runtime bound.
it('X9 records remaining root files and the hierarchical tile limit separately', () => {
  const baseline = JSON.parse(readRepositoryFile('scripts/contributor-contract-baseline.json')) as {
    'root-file': Record<string, unknown>;
  };
  const rootFiles = readRoadmapRow('LR-17');
  const remainingFamilies = new Set(
    Object.keys(baseline['root-file']).map(path => path.split('/')[3])
  );
  expect(remainingFamilies.size).toBeGreaterThan(0);
  for (const path of Object.keys(baseline['root-file'])) {
    expect(existsSync(resolve(REPOSITORY_ROOT, path)), path).toBe(true);
  }
  for (const family of remainingFamilies) expect(rootFiles).toContain(`\`${family}\``);
  expect(rootFiles).toContain('root-file');
  expect(rootFiles).not.toMatch(/point-cloud|pointcloud-analysis|geospatial|norm-merge|uncapped/i);
  const tileCeiling = readRoadmapRow('LR-26');
  for (const phrase of [
    'Error GM',
    '2SLS',
    'weights traces',
    'through 4096 tiles',
    'up to 4096² tiles',
    'tileRowCount',
    'chunks'
  ]) {
    expect(tileCeiling).toContain(phrase);
  }
});

it('X9 keeps LR-19 timings consistent with the measured table and preserves open RSS qualification', () => {
  const measurementRows = readRepositoryFile('dev-docs/release-package.md')
    .split('\n')
    .filter(line => line.startsWith('| Snapshot '));
  expect(measurementRows).toHaveLength(3);
  const row = readRoadmapRow('LR-19');
  const measurements = measurementRows.map(measurement => {
    const cells = measurement
      .split('|')
      .slice(1, -1)
      .map(cell => cell.trim());
    const coldSeconds = Number(cells[2]);
    const warmSeconds = Number(cells[3]);
    expect(Number.isFinite(coldSeconds)).toBe(true);
    expect(Number.isFinite(warmSeconds)).toBe(true);
    expect(cells[4]).toContain('unavailable');
    return {files: Number.parseInt(cells[1], 10), coldSeconds, warmSeconds};
  });
  const sample = row.match(
    /One sample: source (\d+) files ([\d.]+) s cold, references (\d+) files ([\d.]+) s cold \(([\d.]+)% slower\), warm ([\d.]+) s vs ([\d.]+) s/
  );
  expect(sample, 'source and reference cold/warm sample').not.toBeNull();
  expect(Number(sample![1])).toBe(measurements[0].files);
  expect(Number(sample![3])).toBe(measurements[1].files);
  expectRoundedNumber(sample![2], measurements[0].coldSeconds);
  expectRoundedNumber(sample![4], measurements[1].coldSeconds);
  expectRoundedNumber(
    sample![5],
    (measurements[1].coldSeconds / measurements[0].coldSeconds - 1) * 100
  );
  expectRoundedNumber(sample![6], measurements[0].warmSeconds);
  expectRoundedNumber(sample![7], measurements[1].warmSeconds);
  expect(row).not.toContain('Not done: cold, warm');
  expect(row).toContain('no peak RSS was observed');
  expect(row).toContain('project-reference decision is open');
  for (const qualification of [
    '20% lower sampled peak RSS',
    '10% cold or warm time regression',
    'same heap',
    'idle machine'
  ]) {
    expect(row).toContain(qualification);
  }
});

it('W2-02 selects the exact implementation row after repeated summary and decision IDs', () => {
  const markdown = `# Roadmap
## High-priority summary
| ID | Item |
| --- | --- |
| LR-19 | Summary |
## Library review 2026-11-17 (LR)
| ID | Item | Priority |
| --- | --- | --- |
| LR-190 | Prefix collision | H |
| LR-19 / LR-20 | Compound record | H |
| LR-19 | Implementation | M |
### Decisions for the maintainer
| ID | Item | Priority |
| --- | --- | --- |
| LR-19 | Decision | Decision |
`;
  expect(readRoadmapRow('LR-19', markdown)).toContain('Implementation');
  expect(readRoadmapRow('LR-20 / LR-19', markdown)).toContain('Compound record');
});
