// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {
  chmodSync,
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {afterEach, expect, it} from 'vitest';
import type {Device} from '@luma.gl/core';
import {getWGSLModuleUsage} from '../../src/testing/wgsl-memory-usage';
import {
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getWorkgroupMemoryBytes,
  getStorageBindingCount,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES
} from '../../src/testing/gpu-contributor-test-utils';
import {auditWGSLModule} from './kernel-audit/kernel-audit-wgsl';

it('core:F10 counts a function-local copy of a fixed storage array', () => {
  const source = `@group(0) @binding(0) var<storage, read> data: array<u32, 4096>;
    @compute @workgroup_size(1) fn main() { var copy = data; }`;
  const usage = getWGSLModuleUsage(source);
  expect(usage.functionBytes).toBe(16384);
  expect(usage.localArrayBytes).toBe(16384);
  expect(usage.unresolvedTypes).toEqual([]);
  expect(getPrivateMemoryBytes(source)).toBe(16384);
  expect(auditWGSLModule(source).map(finding => finding.check)).toContain('function bytes');
});

it('core:G9 counts a fixed storage array copied through a pointer before a storage write', () => {
  const source = `@group(0) @binding(0) var<storage, read> data: array<u32, 4096>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32, 1>;
    @compute @workgroup_size(1) fn main() {
      let p = &data; var copy = *p; output[0] = copy[1];
    }`;
  const usage = getWGSLModuleUsage(source);
  expect(usage.functionBytes).toBe(16384);
  expect(usage.localArrayBytes).toBe(16384);
  expect(usage.unresolvedTypes).toEqual([]);
  expect(usage.unresolvedLengths).toEqual([]);
  expect(getPrivateMemoryBytes(source)).toBe(16384);
  expect(auditWGSLModule(source).map(finding => finding.check)).toContain('function bytes');
});

it('core:F10 resolves copied arrays through let bindings and preserves their declaration scope', () => {
  const source = `const N = 4096u;
    @group(0) @binding(0) var<storage, read> data: array<u32, N>;
    @compute @workgroup_size(1) fn main() {
      const N = 1u; let previous = data; var copy = previous;
    }`;
  expect(getWGSLModuleUsage(source).functionBytes).toBe(32768);
  expect(auditWGSLModule(source).map(finding => finding.check)).toContain('function bytes');
});

it('core:F10 resolves parameter copies and refuses an unresolved copied initializer', () => {
  const source = 'fn helper(data: array<u32, 4096>) { var copy = data; }';
  expect(getWGSLModuleUsage(source).functionBytes).toBe(16384);
  expect(getWGSLModuleUsage('fn helper(data: u32) { var copy = data; }').functionBytes).toBe(4);
  expect(
    getWGSLModuleUsage('fn helper(data: u32) { let index = (data + 1u) * 2u; var copy = index; }')
      .unresolvedTypes
  ).toEqual([]);
  expect(
    getWGSLModuleUsage('fn helper(data: vec4u) { let value = data + vec4u(1u); var copy = value; }')
      .functionBytes
  ).toBe(16);
  expect(
    getWGSLModuleUsage('fn helper() { let data = array(1u, 2u); var copy = data; }').localArrayBytes
  ).toBe(16);
  const unresolved = 'fn helper() { let data = missingType(); var copy = data; }';
  expect(() => getPrivateMemoryBytes(unresolved)).toThrow(/unresolved/);
  expect(auditWGSLModule(unresolved).map(finding => finding.check)).toContain(
    'unresolved memory layout'
  );
});

function requireRecordedWorkgroupBudget(nodes: readonly unknown[]): void {
  const sources = getParsedKernelSources(nodes);
  for (const {id, source, constants} of sources.configurations) {
    expect(getWorkgroupMemoryBytes(source, constants), id).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
}

function requireRecordedStorageBudget(nodes: readonly unknown[]): void {
  const sources = getParsedKernelSources(nodes);
  for (const {id, source} of sources.configurations)
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
}

it.each([
  [false, false],
  [true, false],
  [false, true],
  [true, true]
] as const)('final2 G10 rejects nine/one shared-ID bindings, reversed=%s, defaultRead=%s', (reversed, defaultRead) => {
  const counts = reversed ? [1, 9] : [9, 1];
  const requests = counts.map(count => ({
    compile: ({device}: {device: Device}) => {
      const source = Array.from(
        {length: count},
        (_, binding) =>
          `@group(0) @binding(${binding}) var<storage${defaultRead ? '' : ', read'}> data${binding}: array<u32>;`
      ).join('\n');
      const shader = device.createShader({stage: 'compute', source});
      device.createComputePipeline({id: 'shared', shader});
      return {destroy: () => shader.destroy()};
    }
  }));
  const sources = getParsedKernelSources(requests);
  expect(
    sources.configurations.map(configuration => getStorageBindingCount(configuration.source))
  ).toEqual(counts);
  expect(() => requireRecordedStorageBudget(requests)).toThrow();
});

it.each([
  ['source', false],
  ['source', true],
  ['overrides', false],
  ['overrides', true]
] as const)('core:F11 checks every shared-ID %s configuration, reversed=%s', (kind, reversed) => {
  const requests = [8192, 1].map(count => ({
    compile: ({device}: {device: Device}) => {
      const source =
        kind === 'source'
          ? `var<workgroup> scratch: array<u32, ${count}>;`
          : 'override N: u32 = 1u; var<workgroup> scratch: array<u32, N>;';
      const shader = device.createShader({stage: 'compute', source});
      device.createComputePipeline({
        id: 'shared',
        shader,
        constants: kind === 'overrides' ? {N: count} : undefined
      });
      return {destroy: () => shader.destroy()};
    }
  }));
  if (reversed) requests.reverse();
  const sources = getParsedKernelSources([...requests, requests[0]]);
  expect(sources.size).toBe(1);
  expect(sources.configurations).toHaveLength(2);
  expect(sources.getConfigurations('shared')).toHaveLength(2);
  expect(
    sources.configurations.map(({source, constants}) => getWorkgroupMemoryBytes(source, constants))
  ).toEqual(reversed ? [16, 32768] : [32768, 16]);
  expect(() => sources.getConfigurations('missing')).toThrow(/no recorded kernel/);
  expect(() => requireRecordedWorkgroupBudget(requests)).toThrow();
});

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const manifest = JSON.parse(readFileSync(join(ROOT, 'scripts/oracle-fixtures.json'), 'utf8'));
const directories: string[] = [];
const ROUND_EIGHT = [
  'groups',
  'profiles',
  'composite-min-max',
  'composite-z-score',
  'composite-rank',
  'histograms',
  'temporal',
  'correlogram',
  'quantiles',
  'composite-pca-selectors'
].map(name => `oracles/fixtures/parity-wp6-round8-${name}.json`);

function write(directory: string, file: string, contents: string) {
  mkdirSync(dirname(join(directory, file)), {recursive: true});
  writeFileSync(join(directory, file), contents);
}

function fixture() {
  const directory = realpathSync(mkdtempSync(join(tmpdir(), 'gate-review-oracles-')));
  directories.push(directory);
  mkdirSync(join(directory, 'scripts'));
  copyFileSync(
    join(ROOT, 'scripts/run-oracle-fixture.mjs'),
    join(directory, 'scripts/run-oracle-fixture.mjs')
  );
  return directory;
}

afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

it.each([
  [
    'parity-wp6-ripley.py',
    'oracles/fixtures/parity-wp6-point-patterns-positions.json',
    ['oracles/fixtures/parity-wp6-ripley.json', 'oracles/fixtures/parity-wp6-ripley-distance.json']
  ],
  [
    'parity-wp2-line-density.py',
    'oracles/fixtures/parity-wp2-prepared-nearest.json',
    ['oracles/fixtures/parity-wp2-line-density.json']
  ],
  [
    'parity-wp2-buffer-selection.py',
    'oracles/fixtures/parity-wp2-prepared-nearest.json',
    ['oracles/fixtures/parity-wp2-buffer-selection.json']
  ],
  [
    'parity-wp2-line-length-per-polygon.py',
    'oracles/fixtures/parity-wp2-prepared-nearest.json',
    ['oracles/fixtures/parity-wp2-line-length-per-polygon.json']
  ],
  [
    'parity-wp2-nearest-weights.py',
    'oracles/fixtures/parity-wp2-prepared-nearest.json',
    ['oracles/fixtures/parity-wp2-nearest-weights.json']
  ],
  [
    'parity-wp2-round8-spherical-lines.py',
    [
      'oracles/fixtures/parity-wp2-prepared-nearest.json',
      'oracles/fixtures/parity-wp2-line-length-per-polygon.json'
    ],
    [
      'oracles/fixtures/parity-wp2-round8-spherical-density.json',
      'oracles/fixtures/parity-wp2-round8-spherical-polygons.json'
    ]
  ],
  [
    'parity-wp3-cost-matrix-live.py',
    'oracles/fixtures/parity-wp3-cost-matrix-synthetic-input.json',
    ['oracles/fixtures/parity-wp3-cost-matrix-live.json']
  ],
  [
    'parity-wp4-change-modes.py',
    'oracles/fixtures/parity-wp4-change-detection.json',
    ['oracles/fixtures/parity-wp4-change-modes.json']
  ],
  ['parity-wp6-round8.py', 'oracles/fixtures/parity-wp6-correlogram.json', ROUND_EIGHT],
  [
    'parity-wp3-cost-matrix.py',
    'oracles/fixtures/parity-wp3-service-areas-chicago-input.json',
    [
      'chicago-forward-five-minutes',
      'chicago-forward-input',
      'chicago-forward-unlimited',
      'chicago-reverse-five-minutes',
      'chicago-reverse-input',
      'chicago-reverse-unlimited',
      'synthetic-boundary',
      'synthetic-input',
      'synthetic-unlimited',
      'synthetic-zero'
    ].map(name => `oracles/fixtures/parity-wp3-cost-matrix-${name}.json`)
  ],
  [
    'parity-wp3-topology-modes.py',
    [
      'oracles/fixtures/parity-wp3-graph-communities-undirected.json',
      'oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-input.json',
      'oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-outgoing.json'
    ],
    [
      'oracles/fixtures/parity-wp3-neighborhood-showcase.json',
      'oracles/fixtures/parity-wp3-noding-showcase-large-snap.json',
      'oracles/fixtures/parity-wp3-graph-communities-showcase.json'
    ]
  ],
  [
    'parity-wp4-contours.py',
    'oracles/fixtures/parity-wp4-isobands.json',
    ['oracles/fixtures/parity-wp4-contours.json']
  ],
  [
    'parity-wp4-stitched.py',
    'oracles/fixtures/parity-wp4-contours.json',
    ['oracles/fixtures/parity-wp4-stitched.json']
  ],
  [
    'statistics-gwr.py',
    'oracles/fixtures/statistics-regression-scenes.json',
    ['oracles/fixtures/statistics-gwr.json']
  ],
  [
    'statistics-spatial-regression.py',
    'oracles/fixtures/statistics-regression-scenes.json',
    ['oracles/fixtures/statistics-spatial-regression.json']
  ],
  [
    'statistics-upstream.py',
    'oracles/fixtures/statistics-coverage.json',
    ['oracles/fixtures/statistics-upstream.json']
  ]
] as const)('stages the real %s record preserving its input and requiring its exact writes', (generator, input, outputs) => {
  const directory = fixture();
  const declaration = manifest.generators[generator];
  const inputs = typeof input === 'string' ? [input] : [...input];
  // Use a stdlib stand-in for the numerical body, keeping the real manifest declaration.
  write(
    directory,
    'scripts/oracle-fixtures.json',
    JSON.stringify({
      version: 1,
      profiles: {[declaration.profile]: {runtime: 'python'}},
      generators: {[generator]: declaration}
    })
  );
  for (const file of new Set([
    ...inputs,
    ...outputs,
    ...declaration.outputs,
    ...(declaration.inputs ?? [])
  ]))
    write(
      directory,
      `modules/analysis/test/${file}`,
      inputs.includes(file) ? 'preserved-input' : '{}'
    );
  write(
    directory,
    `modules/analysis/test/oracles/${generator}`,
    `from pathlib import Path
root = Path.cwd() / 'modules/analysis/test'
for required_input in ${JSON.stringify(inputs)}:
    assert (root / required_input).read_text() == 'preserved-input'
for output in ${JSON.stringify(outputs)}:
    assert not (root / output).exists(), 'output must be recreated'
    (root / output).write_text('{}')
`
  );
  const result = spawnSync(
    process.execPath,
    [join(directory, 'scripts/run-oracle-fixture.mjs'), generator, '--python', '/usr/bin/python3'],
    {cwd: directory, encoding: 'utf8'}
  );
  expect(result.status, result.stdout + result.stderr).toBe(0);
  for (const requiredInput of inputs) expect(declaration.inputs).toContain(requiredInput);
  expect([...declaration.outputs].sort()).toEqual([...outputs].sort());
  expect(result.stdout.match(/IDENTICAL /g)).toHaveLength(outputs.length);
  if (generator === 'parity-wp3-topology-modes.py') {
    const showcase = 'oracles/fixtures/parity-wp3-graph-communities-showcase.json';
    const owners = Object.entries(manifest.generators)
      .filter(([, entry]) => (entry as {outputs?: string[]}).outputs?.includes(showcase))
      .map(([name]) => name);
    expect(owners).toEqual(['parity-wp3-topology-modes.py']);
  }
  for (const file of ROUND_EIGHT) {
    if (generator !== 'parity-wp6-round8.py') continue;
    const owners = Object.entries(manifest.generators)
      .filter(([, entry]) => (entry as {outputs?: string[]}).outputs?.includes(file))
      .map(([name]) => name);
    expect(owners, file).toEqual(['parity-wp6-round8.py']);
  }
});

it('lists legacy environments and excludes generators importing historical GRASS paths', () => {
  for (const file of [
    'nested-validation-oracle.py',
    'raster-round-two.py',
    'raster-spectral-indices.py',
    'statistics-upstream.py',
    'network-movement-coverage.py'
  ])
    expect(manifest.generators[file].profile).toBe('python-legacy');
  expect(manifest.profiles['python-legacy'].description).toContain('2.2.6');
  const result = spawnSync(
    process.execPath,
    [join(ROOT, 'scripts/run-oracle-fixture.mjs'), '--list'],
    {encoding: 'utf8'}
  );
  expect(result.status, result.stderr).toBe(0);
  expect(result.stdout).toContain('nested-validation-oracle.py [python-legacy]');
  for (const file of [
    'parity-wp5-deep-horizon.py',
    'parity-wp5-deep-geomorphons.py',
    'parity-wp5-round10-grass.py'
  ]) {
    expect(manifest.generators[file].kind).toBe('unconfigured');
    expect(result.stdout).not.toContain(`${file} [`);
  }
});

it('does not fall back from a missing explicitly selected Rscript', () => {
  const directory = fixture();
  write(
    directory,
    'scripts/oracle-fixtures.json',
    JSON.stringify({
      version: 1,
      profiles: {rscript: {runtime: 'rscript'}},
      generators: {
        'fixture.R': {kind: 'fixture', profile: 'rscript', outputs: ['oracles/fixtures/value.json']}
      }
    })
  );
  write(directory, 'modules/analysis/test/oracles/fixture.R', '');
  write(directory, 'modules/analysis/test/oracles/fixtures/value.json', '{}');
  write(
    directory,
    'bin/Rscript',
    '#!/bin/sh\nif [ "$1" = "--version" ]; then exit 0; fi\nprintf \'{}\' > modules/analysis/test/oracles/fixtures/value.json\n'
  );
  chmodSync(join(directory, 'bin/Rscript'), 0o755);
  const result = spawnSync(
    process.execPath,
    [
      join(directory, 'scripts/run-oracle-fixture.mjs'),
      'fixture.R',
      '--rscript',
      join(directory, 'missing-Rscript')
    ],
    {
      cwd: directory,
      encoding: 'utf8',
      env: {...process.env, PATH: `${join(directory, 'bin')}:${process.env['PATH']}`}
    }
  );
  expect(result.status, result.stdout + result.stderr).toBe(2);
});

it('serializes saddle command provenance identically across temporary roots', () => {
  const script = join(ROOT, 'modules/analysis/test/oracles/parity-wp5-deep-visibility.py');
  // Exercise make_minimal itself with deterministic numerical/tool stand-ins, so serializing
  // its old real command paths fails without installing or claiming GDAL/GRASS execution.
  const result = spawnSync(
    '/usr/bin/python3',
    [
      '-c',
      `
import contextlib, json, runpy, sys, tempfile, types
from pathlib import Path
import numpy
namespace = runpy.run_path(sys.argv[1], run_name='oracle_test')
make = namespace['make_minimal']
scope = make.__globals__
class Dataset:
    def __enter__(self): return self
    def __exit__(self, *arguments): pass
    def write(self, *arguments): pass
    def read(self, *arguments, **keywords): return numpy.ma.array(numpy.zeros((9, 9)))
sys.modules['rasterio'] = types.SimpleNamespace(open=lambda *arguments, **keywords: Dataset())
sys.modules['rasterio.transform'] = types.SimpleNamespace(from_origin=lambda *arguments: None)
def run(arguments, **keywords):
    if '--gdal-grid' in arguments: Path(arguments[-1]).write_text(json.dumps([[0]*81, [0]*81]))
    elif '--tmp-project' in arguments:
        assert Path(arguments[-1]).parent.as_posix() in Path(arguments[-1]).read_text()
scope['subprocess'] = types.SimpleNamespace(run=run)
scope['REFERENCE_PYTHON'] = '/reference/python'
scope['GRASS'] = '/reference/grass'
results = []
for name in ['first-root', 'different-root']:
    with tempfile.TemporaryDirectory() as parent:
        folder = Path(parent) / name
        folder.mkdir()
        @contextlib.contextmanager
        def temporary(**keywords): yield str(folder)
        scope['tempfile'] = types.SimpleNamespace(TemporaryDirectory=temporary)
        results.append(json.dumps(make(), sort_keys=True))
print(json.dumps(results))
`,
      script
    ],
    {encoding: 'utf8'}
  );
  expect(result.status, result.stderr).toBe(0);
  const [first, second] = JSON.parse(result.stdout);
  expect(first).toBe(second);
  expect(JSON.parse(first).commands[0][1]).toBe('input=dem.tif');
});
