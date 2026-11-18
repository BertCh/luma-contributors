// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import {resolveNodeHeapOptions} from '../../../../scripts/verification-policy.mjs';
import {GPU_NETWORK_COARSENING_MAXIMUM_GROUP_CAPACITY} from '../../src/gpu-network/network-coarsening/gpu-network-coarsening';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '../../../..');

function readRepositoryFile(path: string): string {
  return readFileSync(resolve(REPOSITORY_ROOT, path), 'utf8');
}

function readNumericConstant(path: string, name: string): number {
  const declaration = readRepositoryFile(path).match(
    new RegExp(`const ${name} = (0x[\\da-f]+|\\d+);`, 'i')
  );
  expect(declaration, `${path}: ${name}`).not.toBeNull();
  return Number(declaration![1]);
}

it('final2 G9 documents runtime-only exclusions independently from the 8192 MiB default typecheck heap', () => {
  const manifest = JSON.parse(readRepositoryFile('test/kepler-compute-example-specs.json'));
  const runtimeOnly = 'modules/kepler/test/interpolation-geotiff.node.spec.ts';
  expect(manifest.excludedFromRuns[runtimeOnly]).toBeDefined();
  expect(manifest.excludedFromTypecheck[runtimeOnly]).toBeUndefined();
  expect(manifest.description).toContain('runtime only');
  expect(manifest.description).toContain('8192 MiB');
  expect(manifest.description).not.toMatch(/16 GB|left out of vitest and of the test typecheck/);
  expect(resolveNodeHeapOptions('')).toBe('--max-old-space-size=8192');
});

it('documents every direct lenses dependency and luma peer', () => {
  const manifest = JSON.parse(readRepositoryFile('modules/lenses/package.json'));
  const dependencies = readRepositoryFile('modules/lenses/README.md')
    .split('## Dependencies')[1]
    .split('## Conventions')[0];
  for (const name of [
    ...Object.keys(manifest.dependencies),
    ...Object.keys(manifest.peerDependencies)
  ]) {
    expect(dependencies, name).toContain(name);
  }
});

it('aligns the deck package extension with the workspace graph pin', () => {
  const manifest = JSON.parse(readRepositoryFile('package.json'));
  const extension = readRepositoryFile('.yarnrc.yml').match(/"@luma.gl\/graph": "([^"]+)"/);
  expect(extension).not.toBeNull();
  expect(extension![1]).toBe(manifest.resolutions['@luma.gl/graph']);
});

it('distinguishes panel-error effects and shipped sorted rank-Markov support', () => {
  const roadmap = readRepositoryFile('dev-docs/roadmap.md');
  const panelRow = roadmap.split('\n').find(line => line.startsWith('| ST03 |'))!;
  expect(panelRow).toContain('Panel error ML supports unit fixed effects only');
  expect(panelRow).toContain('panel lag ML supports fixed and random effects');
  const rankRow = roadmap.split('\n').find(line => line.startsWith('| ST08 |'))!;
  expect(rankRow).toContain('Sorted ranking supports more than 64 states');
  expect(rankRow).toContain('reference diagnostics remain capped at 64 states');
});

it('documents the forest total-row maximum separately from per-batch work', () => {
  const maximumRows = readNumericConstant(
    'modules/analysis/src/dataframe-analysis/forest-inference/forest-inference-parameters.ts',
    'GPU_FOREST_INFERENCE_MAXIMUM_ROW_COUNT'
  );
  const documentation = readRepositoryFile('docs/api/forest-inference.md');
  const row = documentation.split('\n').find(line => line.startsWith('| Total rows |'));
  expect(row).toBeDefined();
  expect(Number(row!.split('|')[2].trim().replaceAll(',', ''))).toBe(maximumRows);
  expect(documentation).toContain('per batch');
  expect(documentation).not.toContain('All views\nare single packed buffers');
});

it('permits forest output subviews while documenting logical and aligned separation', () => {
  const documentation = readRepositoryFile('docs/api/forest-inference.md');
  expect(documentation).toContain('Output views may share a physical buffer');
  expect(documentation).toContain('logical output ranges');
  expect(documentation).toContain('aligned storage-binding ranges');
  expect(documentation).not.toContain('Output buffers must\nbe distinct');
});

it('documents total codec pixels and both sides of the per-contributor chunk budget', () => {
  const documentation = readRepositoryFile('docs/api/gpu-raster/raster-embedding-codecs.md');
  expect(documentation).toContain('total pixelCount is at most 4,294,967,295');
  expect(documentation).toContain('1 + dispatchesPerChunk × nonemptyChunks ≤ 32,768');
  for (const [name, dispatches] of [
    ['Encode', 2],
    ['Decode', 1]
  ] as const) {
    const accepted = Math.floor((32768 - 1) / dispatches);
    expect(documentation).toContain(
      `| ${name} | ${dispatches} | ${accepted.toLocaleString('en-US')} | ${(accepted + 1).toLocaleString('en-US')} |`
    );
  }
});

it('distinguishes executor CPU snapshots from the row helper packing copy', () => {
  const documentation = readRepositoryFile('dev-docs/contributor-contract.md');
  expect(documentation).toContain('CPU-only stages also materialize callback chunk snapshots');
  expect(documentation).toContain('Borrows the callback snapshot for one chunk');
  expect(documentation).toContain('additional packed copy');
});

it('routes primary setup instructions to the landed typecheck runner and exclusion manifest', () => {
  const readme = readRepositoryFile('README.md');
  const agents = readRepositoryFile('AGENTS.md');
  const runner = readRepositoryFile('scripts/typecheck-tests.mjs');
  for (const documentation of [readme, agents]) {
    expect(documentation).toContain('scripts/typecheck-tests.mjs');
    expect(documentation).toContain('test/kepler-compute-example-specs.json');
  }
  if (runner.includes("'example-backed specs'")) {
    expect(readme).toContain('example-backed');
    expect(readme).toContain('excludedFromTypecheck');
  }
  expect(readme).not.toContain(
    'tests, minus the kepler-compute example specs; same memory setting'
  );
  expect(agents).not.toContain('needs the same NODE_OPTIONS');
});

it('describes preserved caller options and the current default heap without obsolete overrides', () => {
  const callerOptions = '--trace-warnings --max-old-space-size=256';
  expect(resolveNodeHeapOptions(callerOptions)).toBe(callerOptions);
  expect(resolveNodeHeapOptions('')).toBe('--max-old-space-size=8192');
  const readme = readRepositoryFile('README.md');
  expect(readme).not.toMatch(/overrides\s+`NODE_OPTIONS`/);
  expect(readme).not.toMatch(/16,384,\s*16,384\s+and\s+8,192/);
  expect(readme).toContain('preserves all caller');
  expect(readme).toContain('8,192 MiB');
  expect(readme).toContain('family');
});

it('states the coarsening addressing ceiling and the existing 131072-group coverage', () => {
  const guide = readRepositoryFile('docs/api/gpu-network/operators.md')
    .split('### `GPUNetworkCoarsening`')[1]
    .split('\n### ')[0];
  expect(guide).toContain('GPU_NETWORK_COARSENING_MAXIMUM_GROUP_CAPACITY');
  expect(guide).toContain(GPU_NETWORK_COARSENING_MAXIMUM_GROUP_CAPACITY.toLocaleString('en-US'));
  expect(guide).toContain('device storage-binding');
  expect(guide).not.toContain('groupCapacity` (at most 65535)');
  expect(guide).toContain('131,072');
  const headless = readRepositoryFile(
    'modules/analysis/test/gpu-network/network-coarsening/gpu-network-coarsening.spec.ts'
  );
  expect(headless).toMatch(/groupCapacity:\s*131072/);
});

it('keeps the codec dimension maximum consistent across the page', () => {
  const maximumDimensions = readNumericConstant(
    'modules/analysis/src/raster-analysis/raster-embeddings/raster-embedding-common.ts',
    'MAXIMUM_RASTER_EMBEDDING_DIMENSIONS'
  );
  const documentation = readRepositoryFile('docs/api/gpu-raster/raster-embedding-codecs.md');
  const limits = documentation.split('## Limits and verification')[1];
  expect(limits).toBeDefined();
  expect(limits).toContain(`1–${maximumDimensions.toLocaleString('en-US')} dimensions`);
  expect(limits).toContain('per chunk');
  expect(limits).not.toContain('Views are single packed buffers');
});

it('qualifies the SCC reference bound with its exact route', () => {
  const maximumExactNodes = readNumericConstant(
    'modules/analysis/src/gpu-network/network-structure/gpu-network-structure.ts',
    'GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_NODE_COUNT'
  );
  const documentation = readRepositoryFile('dev-docs/coverage-matrix.md');
  const row = documentation.split('\n').find(line => line.startsWith('| NW-SCC |'));
  expect(row).toBeDefined();
  expect(row).toContain(
    `exact route is limited to ${maximumExactNodes.toLocaleString('en-US')} vertices`
  );
  expect(row).toContain('parallel');
});
