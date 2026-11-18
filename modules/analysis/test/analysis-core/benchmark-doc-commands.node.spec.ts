// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import {listTestFiles} from '../../../../scripts/verification-policy.mjs';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '../../../..');
const scripts = JSON.parse(readFileSync(resolve(REPOSITORY_ROOT, 'package.json'), 'utf8'))
  .scripts as Record<string, string>;
const benchmarkFiles = new Set(listTestFiles(REPOSITORY_ROOT, 'benchmarks'));

it.each([
  [
    'docs/api/gpu-network/network-movement-benchmarks.md',
    'modules/analysis/test/gpu-network/network-optimization/network-movement-benchmark.spec.ts'
  ],
  [
    'docs/api/gpu-terrain/terrain-processing.md',
    'modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing-benchmark.spec.ts'
  ],
  [
    'modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/interpolation-performance-README.md',
    'modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/interpolation-performance-benchmark.spec.ts'
  ],
  [
    'modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing-benchmark.spec.ts',
    'modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing-benchmark.spec.ts'
  ],
  [
    'modules/analysis/test/gpu-raster/medial-axis/gpu-raster-medial-axis-benchmark.spec.ts',
    'modules/analysis/test/gpu-raster/medial-axis/gpu-raster-medial-axis-benchmark.spec.ts'
  ],
  [
    'docs/api/gpu-spatial-analysis/global-coordinates.md',
    'modules/analysis/test/gpu-spatial-analysis/global-coordinates/global-coordinates-benchmark.spec.ts'
  ],
  [
    'modules/analysis/test/gpu-raster/remote-sensing/raster-performance-README.md',
    'modules/analysis/test/gpu-raster/remote-sensing/raster-performance-benchmark.spec.ts'
  ],
  [
    'modules/analysis/test/geospatial-expansion/backlog-performance-README.md',
    'modules/analysis/test/geospatial-expansion/backlog-performance-benchmark.spec.ts'
  ],
  [
    'modules/lenses/test/computation-lenses/raster-terrain/coverage-tranche-one-benchmark.md',
    'modules/lenses/test/computation-lenses/raster-terrain/coverage-tranche-one-benchmark.spec.ts'
  ],
  [
    'modules/analysis/test/gpu-spatial-analysis/neighbor-search/neighbor-search-benchmark.spec.ts',
    'modules/analysis/test/gpu-spatial-analysis/neighbor-search/neighbor-search-benchmark.spec.ts'
  ],
  [
    'modules/analysis/test/gpu-spatial-analysis/spatial-regression/gwr-benchmark.spec.ts',
    'modules/analysis/test/gpu-spatial-analysis/spatial-regression/gwr-benchmark.spec.ts'
  ],
  [
    'modules/analysis/test/gpu-spatial-analysis/spatial-join/spatial-relate-benchmark.ts',
    'modules/analysis/test/gpu-spatial-analysis/spatial-join/spatial-join-benchmark.spec.ts'
  ],
  [
    'modules/analysis/test/gpu-dataframe/pair-statistics/ripley-isotropic-envelope-benchmark.spec.ts',
    'modules/analysis/test/gpu-dataframe/pair-statistics/ripley-isotropic-envelope-benchmark.spec.ts'
  ]
])('uses a published benchmark script and inventoried spec in %s', (documentation, benchmark) => {
  const source = readFileSync(resolve(REPOSITORY_ROOT, documentation), 'utf8');
  expect(
    source.includes('LUMA_TEST_BROWSER_BENCHMARKS'),
    `${documentation}: obsolete benchmark selector`
  ).toBe(false);
  const commands = [
    ...source.matchAll(/\byarn\s+(test-[\w-]*benchmarks|test-headless)\b([^\r\n`]*)/g)
  ];
  expect(commands.length, `${documentation}: missing benchmark command`).toBeGreaterThan(0);
  for (const [, script, arguments_] of commands) {
    expect(scripts[script], `${documentation}: missing package.json script ${script}`).toBeTypeOf(
      'string'
    );
    expect(script).toBe('test-benchmarks');
    const selectedFiles =
      arguments_.match(/modules\/[\w./-]+-(?:bench|benchmark)\.spec\.(?:ts|js)\b/g) ?? [];
    expect(selectedFiles, `${documentation}: command must select its benchmark file`).toContain(
      benchmark
    );
    for (const file of selectedFiles) {
      expect(
        benchmarkFiles.has(file),
        `${documentation}: ${file} is absent from the benchmark inventory`
      ).toBe(true);
    }
  }
});

it('uses the correctness runner for the non-benchmark neighbor-search A/B filename', () => {
  const file =
    'modules/analysis/test/gpu-spatial-analysis/neighbor-search/neighbor-search-structure-ab.spec.ts';
  const source = readFileSync(resolve(REPOSITORY_ROOT, file), 'utf8');
  expect(
    source.includes('LUMA_TEST_BROWSER_BENCHMARKS'),
    `${file}: obsolete benchmark selector`
  ).toBe(false);
  expect(source.includes(`yarn test-headless ${file}`)).toBe(true);
  expect(scripts['test-headless']).toBeTypeOf('string');
  expect(listTestFiles(REPOSITORY_ROOT, 'headless')).toContain(file);
  expect(benchmarkFiles.has(file)).toBe(false);
});
