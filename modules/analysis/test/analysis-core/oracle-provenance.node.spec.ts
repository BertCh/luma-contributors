// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {afterEach, expect, it} from 'vitest';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const ORACLES = 'modules/analysis/test/oracles';
const OWNER = 'parity-wp3-topology-modes.py';
const manifest = JSON.parse(readFileSync(join(ROOT, 'scripts/oracle-fixtures.json'), 'utf8'));
const directories: string[] = [];

function readFixture(file: string) {
  return JSON.parse(readFileSync(join(ROOT, ORACLES, 'fixtures', file), 'utf8'));
}

afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

it.each([
  'parity-wp4-contours.json',
  'parity-wp4-stitched.json'
])('resolves every recorded input path and digest in %s', file => {
  const fixture = readFixture(file);
  const sources = fixture.sources ?? [fixture.source];
  expect(sources.length).toBeGreaterThan(0);
  const inputs = manifest.generators[file.replace(/\.json$/, '.py')].inputs as string[];
  expect(sources.map((source: {path: string}) => source.path)).toEqual(
    expect.arrayContaining(inputs.map(input => `modules/analysis/test/${input}`))
  );
  for (const source of sources) {
    const path = join(ROOT, source.path);
    expect(existsSync(path), source.path).toBe(true);
    expect(createHash('sha256').update(readFileSync(path)).digest('hex'), source.path).toBe(
      source.sha256
    );
  }
});

it.each(
  manifest.generators[OWNER].outputs as string[]
)('records the manifest owner in every showcase record of %s', output => {
  const owners = Object.entries(manifest.generators)
    .filter(([, entry]) => (entry as {outputs?: string[]}).outputs?.includes(output))
    .map(([name]) => name);
  expect(owners).toEqual([OWNER]);
  const fixture = JSON.parse(readFileSync(join(ROOT, 'modules/analysis/test', output), 'utf8'));
  const records = Array.isArray(fixture) ? fixture : [fixture];
  expect(records.length).toBeGreaterThan(0);
  for (const record of records) expect(record.generator).toBe(OWNER);
});

it('the topology entry script stamps all helper-produced showcases with its own generator', () => {
  const directory = mkdtempSync(join(tmpdir(), 'oracle-provenance-'));
  directories.push(directory);
  const scripts = join(directory, ORACLES);
  mkdirSync(join(scripts, 'fixtures'), {recursive: true});
  writeFileSync(join(scripts, OWNER), readFileSync(join(ROOT, ORACLES, OWNER)));
  for (const [name, value] of Object.entries({
    'parity-wp3-graph-communities-undirected.json': {nodeCount: 2, edges: [[0, 1, 1]]},
    'parity-wp3-network-analysis-synthetic-directed-input.json': {
      nodeCount: 2,
      offsets: [0, 1, 1],
      neighbors: [1]
    },
    'parity-wp3-network-analysis-synthetic-directed-outgoing.json': {
      generator: 'parity-wp3-network-analysis.py'
    }
  }))
    writeFileSync(join(scripts, 'fixtures', name), JSON.stringify(value));
  const result = spawnSync(
    '/usr/bin/python3',
    [
      '-I',
      '-c',
      `
import json, runpy, sys, types
from pathlib import Path
entry = Path(sys.argv[1])
class Graph:
    def add_nodes_from(self, nodes): self.nodes = list(nodes)
    def add_weighted_edges_from(self, edges): pass
    def add_edges_from(self, edges): pass
    def __iter__(self): return iter(self.nodes)
    def subgraph(self, nodes): return self
    def has_edge(self, source, target): return True
sys.modules['numpy'] = types.SimpleNamespace(float32=float)
sys.modules['networkx'] = types.SimpleNamespace(MultiGraph=Graph, MultiDiGraph=Graph,
    single_source_shortest_path_length=lambda graph, start, cutoff: {0: 0, 1: 1})
namespace = runpy.run_path(str(entry))
fixtures = entry.parent / 'fixtures'
def noding(output, name, coordinates, tolerance, source):
    (output / ('parity-wp3-noding-' + name + '.json')).write_text(json.dumps(dict(generator='parity-wp3-noding.py', payload=[1, 2])))
def communities(name, fixture):
    fixture['generator'] = 'parity-wp3-graph-communities.py'
    (fixtures / ('parity-wp3-graph-communities-' + name + '.json')).write_text(json.dumps(fixture))
helpers = {'noding': types.SimpleNamespace(generate=noding), 'graph-communities': types.SimpleNamespace(
    evaluate=lambda graph, labels, resolution: dict(score=0.0), write=communities)}
namespace['main'].__globals__['loadModule'] = lambda name: helpers[name]
namespace['main']()
`,
      join(scripts, OWNER)
    ],
    {cwd: directory, encoding: 'utf8'}
  );
  expect(result.status, result.stdout + result.stderr).toBe(0);
  for (const output of manifest.generators[OWNER].outputs) {
    const fixture = JSON.parse(
      readFileSync(join(directory, 'modules/analysis/test', output), 'utf8')
    );
    const records = Array.isArray(fixture) ? fixture : [fixture];
    for (const record of records) expect(record.generator, output).toBe(OWNER);
    if (output.includes('neighborhood')) expect(records).toHaveLength(24);
    if (output.includes('noding')) expect(fixture.payload).toEqual([1, 2]);
  }
});
