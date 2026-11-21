// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, it} from 'vitest';
import typescript from 'typescript';
import {
  buildOracleCensus,
  renderOracleCensus,
  updateOracleCensus
} from '../../../../scripts/oracle-census.mjs';

function createInventory() {
  const root = mkdtempSync(join(tmpdir(), 'oracle-census-'));
  const write = (relative: string, value: string) => {
    const target = join(root, relative);
    mkdirSync(dirname(target), {recursive: true});
    writeFileSync(target, value);
  };
  for (const program of ['first.py', 'second.py', 'unknown.py', 'inventory.py'])
    write(`modules/analysis/test/oracles/${program}`, '# miniature generator\n');
  // Three fixtures; two owners share one existing output, one declared output is absent.
  write('modules/analysis/test/oracles/fixtures/shared.json', '{}');
  write(
    'modules/analysis/test/oracles/fixtures/provenance-only.json',
    '{"generator":"unknown.py"}'
  );
  write('modules/analysis/test/oracles/fixtures/extra.json', '{}');
  const manifest = {
    version: 1,
    profiles: {python: {runtime: 'python'}},
    generators: {
      'first.py': {
        kind: 'fixture',
        profile: 'python',
        outputs: ['oracles/fixtures/shared.json', 'oracles/fixtures/missing.json']
      },
      'second.py': {kind: 'fixture', profile: 'python', outputs: ['oracles/fixtures/shared.json']},
      'unknown.py': {
        kind: 'unconfigured',
        profile: 'python',
        reason: 'unmapped',
        outputs: ['oracles/fixtures/provenance-only.json']
      },
      'inventory.py': {kind: 'inventory', profile: 'python', reason: 'source inventory'}
    }
  };
  return {root, write, manifest};
}

it('renders separate disk, declaration and producer populations from a miniature inventory (TOOL-107)', () => {
  const inventory = createInventory();
  try {
    const census = buildOracleCensus({
      repositoryRoot: inventory.root,
      manifest: inventory.manifest
    });
    expect(census.counts).toMatchObject({
      fixtures: 3,
      programs: 4,
      producers: 2,
      unconfigured: 1,
      inventory: 1,
      declaredOutputs: 3,
      configuredOutputs: 2,
      existingOutputs: 2,
      existingConfiguredOutputs: 1,
      missingOutputs: 1,
      fixturesWithoutProducer: 2
    });
    expect(census.missingOutputs).toEqual(['oracles/fixtures/missing.json']);
    const rendered = renderOracleCensus(census);
    for (const line of [
      '- Fixture JSON files on disk: **3**.',
      '- Generator programs on disk: **4**.',
      '- Manifest programs: **4** (**2** fixture producers, **1** unconfigured, **1** inventory).',
      '- Distinct declared output paths: **3**; existing: **2**; missing: **1**.',
      '- Configured output paths: **2**; existing: **1**; missing: **1**.',
      '- Fixture JSON files with no configured producer: **2**.',
      '- Run evidence: **0** identified runs; no outputs are certified as regenerated and compared.'
    ])
      expect(rendered).toContain(line);
    const markdown = 'Introduction\n\n## Counts\n\nOld totals.\n\n## Reproducing\nCommands\n';
    const updated = updateOracleCensus(markdown, rendered);
    expect(updated).not.toContain('Old totals');
    expect(updated).toContain('## Reproducing\nCommands');
    expect(updateOracleCensus(updated, rendered)).toBe(updated);
    expect(() =>
      buildOracleCensus({
        repositoryRoot: inventory.root,
        manifest: {...inventory.manifest, version: 2}
      })
    ).toThrow();
  } finally {
    rmSync(inventory.root, {recursive: true, force: true});
  }
});

it('keeps the reader-facing census synchronized with disk and the validated manifest (TOOL-107)', () => {
  const repositoryRoot = fileURLToPath(new URL('../../../../', import.meta.url));
  const manifest = JSON.parse(
    readFileSync(join(repositoryRoot, 'scripts/oracle-fixtures.json'), 'utf8')
  );
  const markdown = readFileSync(
    join(repositoryRoot, 'modules/analysis/test/oracles/GENERATORS.md'),
    'utf8'
  );
  const rendered = renderOracleCensus(buildOracleCensus({repositoryRoot, manifest}));
  expect(markdown, 'Run node scripts/oracle-census.mjs --write after inventory changes').toBe(
    updateOracleCensus(markdown, rendered)
  );
});

it('attributes hashes only through explicit identity, declared ownership or exact source paths', () => {
  const inventory = createInventory();
  // SHA-256('abc'), a published SHA-256 test vector; no production hash helper sets expectations.
  const digest = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';
  try {
    inventory.write('modules/analysis/test/oracles/first.py', 'abc');
    inventory.write('modules/analysis/test/oracles/second.py', 'abc');
    const writeFixture = (name: string, data: Record<string, unknown>) =>
      inventory.write(`modules/analysis/test/oracles/fixtures/${name}.json`, JSON.stringify(data));
    writeFixture('explicit', {generator: 'test/oracles/first.py', generatorSha256: digest});
    writeFixture('source', {
      source: 'modules/analysis/test/oracles/second.py',
      generatorSha256: digest
    });
    writeFixture('missing', {generatorSha256: digest}); // The first producer declares this exact path.
    writeFixture('mismatch', {generator: 'first.py', generatorSha256: '0'.repeat(64)});
    writeFixture('first', {generatorSha256: digest}); // A basename guess must not resolve this.
    writeFixture('shared', {generatorSha256: digest}); // Two declared owners cannot resolve this.
    writeFixture('conflict', {
      generator: 'first.py',
      source: 'modules/analysis/test/oracles/second.py',
      generatorSha256: digest
    });
    inventory.write('inputs/reference.txt', 'abc');
    writeFixture('input-verified', {source: 'inputs/reference.txt', sourceSha256: digest});
    writeFixture('input-mismatch', {source: 'inputs/reference.txt', sourceSha256: '0'.repeat(64)});
    writeFixture('input-unattributed', {source: '/historical/reference.txt', sourceSha256: digest});
    writeFixture('source-map', {
      sourceSha256: {
        'inputs/reference.txt': digest,
        'modules/analysis/test/oracles/first.py': '0'.repeat(64)
      }
    });
    inventory.write('modules/analysis/test/oracles/unknown.py', 'abc');
    writeFixture('provenance-only', {generatorSha256: digest});
    const census = buildOracleCensus({
      repositoryRoot: inventory.root,
      manifest: inventory.manifest
    });
    expect(census.generatorHashes.counts).toEqual({
      total: 8,
      verified: 4,
      mismatching: 1,
      unattributed: 3
    });
    expect(census.sourceHashes.counts).toEqual({
      total: 5,
      verified: 2,
      mismatching: 2,
      unattributed: 1
    });
    expect(
      census.generatorHashes.records.find(record => record.fixture.endsWith('/missing.json'))
    ).toMatchObject({
      status: 'verified',
      generator: 'first.py',
      attribution: 'declared output ownership'
    });
    const rendered = renderOracleCensus(census);
    expect(rendered).toContain(
      'Generator hashes: **8** recorded; verified: **4**; mismatching: **1**; unattributed: **3**.'
    );
    expect(rendered).toContain(
      'Recorded source path hashes: **5**; verified: **2**; mismatching: **2**; unattributed: **1**.'
    );
  } finally {
    rmSync(inventory.root, {recursive: true, force: true});
  }
});

it('counts only identified, byte-bound run evidence and keeps partial runs uncompared', async () => {
  const {createOracleRunEvidence} = await import('../../../../scripts/run-oracle-fixture.mjs');
  const inventory = createInventory();
  try {
    inventory.write('modules/analysis/test/oracles/first.py', 'abc');
    inventory.write('modules/analysis/test/oracles/fixtures/shared.json', '"abc"');
    inventory.write('modules/analysis/test/oracles/fixtures/missing.json', '"abc"');
    const base = {
      startedAt: '2026-11-19T00:00:00.000Z',
      completedAt: '2026-11-19T00:00:01.000Z',
      generator: 'first.py',
      generatorBytes: Buffer.from('abc'),
      declaredOutputs: ['oracles/fixtures/shared.json', 'oracles/fixtures/missing.json']
    };
    const output = (path: string, regenerated: string) => ({
      path,
      committedBytes: Buffer.from('"abc"'),
      regeneratedBytes: Buffer.from(regenerated)
    });
    const missing = createOracleRunEvidence({...base, runId: 'missing', outputs: []});
    const partial = createOracleRunEvidence({
      ...base,
      runId: 'partial',
      outputs: [output(base.declaredOutputs[0], '"abc"')]
    });
    const matching = createOracleRunEvidence({
      ...base,
      runId: 'matching',
      outputs: base.declaredOutputs.map(path => output(path, '"abc"'))
    });
    const differing = createOracleRunEvidence({
      ...base,
      runId: 'differing',
      outputs: [output(base.declaredOutputs[0], '"abc"'), output(base.declaredOutputs[1], '"def"')]
    });
    const build = (evidence: unknown[]) =>
      buildOracleCensus({repositoryRoot: inventory.root, manifest: inventory.manifest, evidence});
    expect(build([missing]).regeneration.counts).toMatchObject({
      runs: 1,
      missing: 1,
      partial: 0,
      matching: 0,
      differing: 0,
      comparedOutputs: 0
    });
    expect(build([partial]).regeneration.counts).toMatchObject({
      runs: 1,
      missing: 0,
      partial: 1,
      matching: 0,
      differing: 0,
      comparedOutputs: 0
    });
    expect(build([matching]).regeneration.counts).toMatchObject({
      runs: 1,
      matching: 1,
      matchingOutputs: 2,
      differingOutputs: 0,
      comparedOutputs: 2
    });
    const census = build([differing]);
    expect(census.regeneration.counts).toMatchObject({
      runs: 1,
      differing: 1,
      matchingOutputs: 1,
      differingOutputs: 1,
      comparedOutputs: 2
    });
    expect(renderOracleCensus(census)).toContain(
      '- Run `differing` (first.py; 2026-11-19T00:00:01.000Z): differing.'
    );
    expect(renderOracleCensus(census)).toContain(
      'Distinct outputs regenerated and compared: **2**; matching: **1**; differing: **1**; existing configured outputs without comparison evidence: **0**.'
    );
    expect(() => build([{...matching, runId: ''}])).toThrow(/run evidence/);
    expect(() => build([matching, matching])).toThrow(/duplicate/);
    expect(() =>
      build([
        {...matching, outputs: matching.outputs.map(record => ({...record, regenerated: false}))}
      ])
    ).toThrow(/run evidence/);
    expect(build([matching, differing]).regeneration.counts).toMatchObject({
      runs: 2,
      comparedOutputs: 2,
      matchingOutputs: 1,
      differingOutputs: 1
    });
    expect(() =>
      build([{...matching, declaredOutputs: [base.declaredOutputs[0], base.declaredOutputs[0]]}])
    ).toThrow(/run evidence/);
    expect(() => build([{...partial, outputs: [matching.outputs[0]]}])).toThrow(/run evidence/);
    inventory.write('modules/analysis/test/oracles/fixtures/shared.json', '"unrelated"');
    expect(build([matching]).regeneration.counts).toMatchObject({stale: 1, comparedOutputs: 0});
    inventory.write('modules/analysis/test/oracles/fixtures/shared.json', '"abc"');
    inventory.write('modules/analysis/test/oracles/first.py', 'changed');
    expect(build([matching]).regeneration.counts).toMatchObject({stale: 1, comparedOutputs: 0});
  } finally {
    rmSync(inventory.root, {recursive: true, force: true});
  }
});

it.each([
  'missing rows',
  'missing comparison'
])('rejects %s in a fully regenerated receipt (B5)', async malformed => {
  const {createOracleRunEvidence} = await import('../../../../scripts/run-oracle-fixture.mjs');
  const inventory = createInventory();
  try {
    const declaredOutputs = ['oracles/fixtures/shared.json', 'oracles/fixtures/missing.json'];
    const receipt = createOracleRunEvidence({
      runId: 'inconsistent',
      startedAt: '2026-11-19T00:00:00.000Z',
      completedAt: '2026-11-19T00:00:01.000Z',
      generator: 'first.py',
      generatorBytes: Buffer.from('# miniature generator\n'),
      declaredOutputs,
      outputs: declaredOutputs.map(path => ({
        path,
        committedBytes: Buffer.from('{}'),
        regeneratedBytes: Buffer.from('{}')
      }))
    });
    const build = (evidence: unknown[]) =>
      buildOracleCensus({repositoryRoot: inventory.root, manifest: inventory.manifest, evidence});
    const outputs =
      malformed === 'missing rows'
        ? []
        : receipt.outputs.map((output, index) =>
            index === 0 ? output : {...output, comparison: 'not-compared'}
          );
    expect(() => build([{...receipt, outputs}])).toThrow(/run evidence/);
  } finally {
    rmSync(inventory.root, {recursive: true, force: true});
  }
});

it('rejects reversed, duplicate or missing census boundaries (B7)', () => {
  for (const markdown of [
    '## Reproducing\nCommands\n## Counts\nStale\n',
    '## Counts\nFirst\n## Counts\nSecond\n## Reproducing\nCommands\n',
    '## Counts\nStale\n## Reproducing\nCommands\n## Reproducing\nMore\n',
    '## Counts\nStale\n'
  ])
    expect(() => updateOracleCensus(markdown, 'Fresh')).toThrow(/Counts.*Reproducing/);
});

it('resolves exact recorded command paths through the documented graph-to-analysis move (B2)', () => {
  const inventory = createInventory();
  const digest = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';
  try {
    inventory.write('modules/analysis/test/oracles/parity-wp5-round8-composites.py', 'abc');
    inventory.write('modules/analysis/test/oracles/parity-wp5-round8-curvature.py', 'abc');
    for (const [name, script] of [
      ['moved', 'modules/graph/test/oracles/parity-wp5-round8-composites.py'],
      ['current', 'modules/analysis/test/oracles/parity-wp5-round8-curvature.py'],
      ['unrelated', 'unrelated/test/oracles/parity-wp5-round8-composites.py']
    ])
      inventory.write(
        `modules/analysis/test/oracles/fixtures/${name}.json`,
        JSON.stringify({
          command: ['/historical/venv/bin/python', '-I', script],
          generatorSha256: digest
        })
      );
    const census = buildOracleCensus({
      repositoryRoot: inventory.root,
      manifest: inventory.manifest
    });
    expect(census.generatorHashes.counts).toEqual({
      total: 3,
      verified: 2,
      mismatching: 0,
      unattributed: 1
    });
    expect(
      census.generatorHashes.records.find(record => record.fixture.endsWith('/moved.json'))
    ).toMatchObject({
      status: 'verified',
      generator: 'parity-wp5-round8-composites.py',
      attribution: 'recorded generator command'
    });
  } finally {
    rmSync(inventory.root, {recursive: true, force: true});
  }
});

it('renders source-list and documented fixture-relative hash outcomes (B3)', () => {
  const inventory = createInventory();
  const digest = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';
  try {
    inventory.write('inputs/reference.txt', 'abc');
    inventory.write(
      'modules/analysis/test/oracles/fixtures/parity-wp1-emerging-lengths.json',
      '"abc"'
    );
    inventory.write(
      'modules/analysis/test/oracles/fixtures/parity-wp1-trend-reference.json',
      JSON.stringify({
        generator: 'parity-wp1-trend-reference.py',
        sourceSha256: {'parity-wp1-emerging-lengths.json': '0'.repeat(64)}
      })
    );
    inventory.write(
      'modules/analysis/test/oracles/fixtures/list.json',
      JSON.stringify({
        sources: [
          {path: 'inputs/reference.txt', sha256: digest},
          {path: 'inputs/reference.txt', sha256: '0'.repeat(64)},
          {path: 'inputs/missing.txt', sha256: digest}
        ]
      })
    );
    const census = buildOracleCensus({
      repositoryRoot: inventory.root,
      manifest: inventory.manifest
    });
    expect(census.sourceHashes.counts).toEqual({
      total: 4,
      verified: 1,
      mismatching: 2,
      unattributed: 1
    });
    expect(
      census.sourceHashes.records.find(
        record => record.recordedPath === 'parity-wp1-emerging-lengths.json'
      )
    ).toMatchObject({
      source: 'modules/analysis/test/oracles/fixtures/parity-wp1-emerging-lengths.json',
      status: 'mismatching'
    });
    expect(renderOracleCensus(census)).toContain(
      'Recorded source path hashes: **4**; verified: **1**; mismatching: **2**; unattributed: **1**.'
    );
  } finally {
    rmSync(inventory.root, {recursive: true, force: true});
  }
});

it('marks unresolved and unmatched mapping conclusions as historical (S3)', () => {
  const markdown = readFileSync(new URL('./GENERATORS.md', import.meta.url), 'utf8');
  expect(markdown).toContain('## Historical fixtures with no discoverable generator');
  expect(markdown).toContain('## Historical generators that write no matched fixture');
});

it('emits a concrete buildOracleCensus boundary for strict TypeScript consumers (B6)', () => {
  // Declaration-only transform, without typechecking a project or running a compiler program.
  const fileName = 'scripts/oracle-census.mjs';
  const result = typescript.transpileDeclaration(
    readFileSync(new URL('../../../../scripts/oracle-census.mjs', import.meta.url), 'utf8'),
    {fileName, compilerOptions: {allowJs: true}}
  );
  const declaration = typescript.createSourceFile(
    'boundary.d.ts',
    result.outputText,
    typescript.ScriptTarget.Latest,
    true
  );
  const boundary = declaration.statements.find(
    statement =>
      typescript.isFunctionDeclaration(statement) && statement.name?.text === 'buildOracleCensus'
  ) as typescript.FunctionDeclaration;
  expect(boundary.type?.getText(declaration)).toBe('OracleCensus');
});
