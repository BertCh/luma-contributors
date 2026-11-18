// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  unlinkSync,
  writeFileSync
} from 'node:fs';
import {createHash} from 'node:crypto';
import {tmpdir} from 'node:os';
import {join, relative} from 'node:path';
import {describe, expect, it, vi} from 'vitest';
import * as recordingRegistry from '../../src/computation-lenses/registry-implementation';
import {collectPendingRecordFailures, PENDING_LENS_RECORDS} from './pending-lens-records';
// Registers scenario descriptors (example paths, elements for pack guards) and their domain packs.
import '../../src/computation-lenses/scenarios/index';
import {
  fromRecordedLensFile,
  getLensScenario,
  type DecodedLensFile
} from '../../src/computation-lenses/index';
import {getLensExamplePath} from '../../src/computation-lenses/lens-example';
import {buildDecodedLensTwin} from '../../src/computation-views/index';
import {
  parseLensDirective,
  validateLensTwin,
  type LensTwin
} from '../../src/computation-views/twin/static-twin';
import {
  DOCS_ROOT,
  REPOSITORY_ROOT,
  listDocsMarkdownFiles,
  scanLensDirectives,
  twinKey
} from './lens-directives';

// Regenerate with UPDATE_LENS_TWINS=1 npx vitest run --project node
// modules/lenses/test/computation-docs/lens-twins.node.spec.ts.
// Only the capture run writes source records; this spec writes derived twins when opted in.
const RECORDS_DIRECTORY = join(DOCS_ROOT, 'lens-records');
const TWINS_DIRECTORY = join(RECORDS_DIRECTORY, 'twins');
const LENSES_SOURCE_DIRECTORY = join(REPOSITORY_ROOT, 'modules/lenses/src/computation-lenses');

/** Source text of a scenario's `.example.ts` (CARD-DESIGN §1: printed under the result figure). */
function readLensExampleSource(scenarioId: string): string | undefined {
  const examplePath = join(
    LENSES_SOURCE_DIRECTORY,
    getLensScenario(scenarioId)?.example?.path ?? getLensExamplePath(scenarioId)
  );
  return existsSync(examplePath) ? readFileSync(examplePath, 'utf8') : undefined;
}

/** Catalog thumbnails exist for plain `rung=result` directives (CARD-DESIGN §7). */
function getThumbnailTwinFile(attributes: Record<string, string>): string | undefined {
  return attributes['rung'] === 'result' &&
    attributes['datum'] === undefined &&
    attributes['iteration'] === undefined &&
    attributes['view'] === undefined
    ? `${twinKey(attributes)}~thumbnail.json`
    : undefined;
}

function serializeTwin(
  recorded: DecodedLensFile,
  attributes: Record<string, string>,
  size: 'full' | 'thumbnail' = 'full'
): string {
  const directive = {
    ...attributes,
    scenario: attributes['scenario'],
    rung: attributes['rung'],
    // Author captions belong to hosts and must not affect a shared key's generated twin.
    caption: undefined
  };
  const parsed = parseLensDirective(directive);
  const envelope = {
    schema: 'luma.lens-twin/1',
    scenario: parsed.scenarioId,
    rung: parsed.rung,
    ...(attributes['datum'] === undefined ? {} : {datum: Number(attributes['datum'])}),
    ...(attributes['iteration'] === undefined ? {} : {iteration: Number(attributes['iteration'])}),
    ...(attributes['view'] === undefined ? {} : {view: attributes['view']})
  };
  const example =
    size === 'full' && parsed.rung === 'result'
      ? readLensExampleSource(parsed.scenarioId)
      : undefined;
  const twin = buildDecodedLensTwin(recorded, directive, {
    size,
    ...(example === undefined ? {} : {example})
  });
  return `${JSON.stringify({...envelope, ...twin}, null, 2)}\n`;
}

function listTwinFiles(directory = TWINS_DIRECTORY): string[] {
  return existsSync(directory)
    ? readdirSync(directory, {withFileTypes: true})
        .filter(file => file.isFile() && file.name.endsWith('.json'))
        .map(file => file.name)
        .sort()
    : [];
}

function collectTwinDrift(
  expected: ReadonlyMap<string, string>,
  existing: ReadonlyMap<string, string>,
  referenced: ReadonlySet<string>
): string[] {
  const failures: string[] = [];
  for (const [file, text] of expected) {
    if (!existing.has(file)) failures.push(`missing twin: ${file}`);
    else if (existing.get(file) !== text) failures.push(`stale twin: ${file}`);
  }
  for (const file of existing.keys()) {
    if (!referenced.has(file)) failures.push(`orphan twin: ${file}`);
    else if (!expected.has(file)) failures.push(`missing source record: ${file}`);
  }
  return failures.sort();
}

function collectTwinContractFailures(twin: LensTwin): string[] {
  const required = {
    caption: twin?.caption,
    'keyNumber.label': twin?.keyNumber?.label,
    'keyNumber.value': twin?.keyNumber?.value,
    description: twin?.description,
    'provenance.adapter': twin?.provenance?.adapter,
    'provenance.date': twin?.provenance?.date,
    'provenance.runId': twin?.provenance?.runId
  };
  const failures = Object.entries(required)
    .filter(([, value]) => typeof value !== 'string' || value.trim().length === 0)
    .map(([name]) => `${name} must be a non-empty string`);
  try {
    failures.push(...validateLensTwin(twin));
  } catch (error) {
    failures.push(`validateLensTwin: ${String(error)}`);
  }
  return failures;
}

function materializeCanonicalTwins(
  files: readonly string[],
  options: {
    update?: boolean;
    recordsDirectory?: string;
    twinsDirectory?: string;
    pending?: Readonly<Record<string, string>>;
    writeTwin?: (path: string, data: Uint8Array) => void;
    removeTwin?: (path: string) => void;
  } = {}
): void {
  const recordsDirectory = options.recordsDirectory ?? RECORDS_DIRECTORY;
  const twinsDirectory = options.twinsDirectory ?? TWINS_DIRECTORY;
  const writeTwin = options.writeTwin ?? writeFileSync;
  const removeTwin = options.removeTwin ?? unlinkSync;
  expect(files.length).toBeGreaterThan(0);
  // Retain only filenames and hashes across scenarios, never all serialized or decoded records.
  const expected = new Map<string, string>();
  const referenced = new Set<string>();
  const failures: string[] = [];
  const groups = new Map<string, ReturnType<typeof scanLensDirectives>>();
  for (const file of files) {
    try {
      for (const directive of scanLensDirectives(file)) {
        const scenarioId = directive.attributes['scenario'];
        const group = groups.get(scenarioId) ?? [];
        group.push(directive);
        groups.set(scenarioId, group);
      }
    } catch (error) {
      failures.push(String(error));
    }
  }
  failures.push(
    ...collectPendingRecordFailures(
      new Set(groups.keys()),
      options.pending ?? PENDING_LENS_RECORDS,
      scenarioId => existsSync(join(recordsDirectory, `${scenarioId}.json`))
    )
  );
  if (failures.length) throw new Error(failures.join('\n'));
  const update = options.update ?? process.env['UPDATE_LENS_TWINS'] === '1';
  const stagingDirectory = update ? mkdtempSync(join(tmpdir(), 'lens-twins-')) : undefined;
  const existingFiles = listTwinFiles(twinsDirectory);
  function checkTwin(file: string, text: string): void {
    const hash = createHash('sha256').update(text).digest('hex');
    if (expected.has(file)) {
      if (expected.get(file) !== hash)
        failures.push(`${file}: directives with the same key produced different twins`);
      return;
    }
    expected.set(file, hash);
    if (stagingDirectory) writeFileSync(join(stagingDirectory, file), text);
  }
  try {
    for (const [scenarioId, directives] of groups) {
      for (const {attributes} of directives) {
        referenced.add(`${twinKey(attributes)}.json`);
        const thumbnail = getThumbnailTwinFile(attributes);
        if (thumbnail) referenced.add(thumbnail);
      }
      const recordFile = join(recordsDirectory, `${scenarioId}.json`);
      if (!existsSync(recordFile)) continue;
      try {
        // Decode once, render this scenario's variants, then release its capture arrays.
        const recorded = fromRecordedLensFile(JSON.parse(readFileSync(recordFile, 'utf8')));
        for (const {attributes} of directives) {
          checkTwin(`${twinKey(attributes)}.json`, serializeTwin(recorded, attributes));
          const thumbnail = getThumbnailTwinFile(attributes);
          if (thumbnail) checkTwin(thumbnail, serializeTwin(recorded, attributes, 'thumbnail'));
        }
      } catch (error) {
        const directive = directives[0];
        failures.push(
          `${relative(REPOSITORY_ROOT, directive.file)}:${directive.line}: ${String(error)}`
        );
      }
    }
    // Include orphan twins: losing their capture must fail even an opt-in regeneration.
    const existing = new Map<string, string>();
    for (const file of existingFiles) {
      const text = readFileSync(join(twinsDirectory, file), 'utf8');
      existing.set(file, createHash('sha256').update(text).digest('hex'));
      const twin = JSON.parse(text);
      if (!existsSync(join(recordsDirectory, `${twin.scenario}.json`)))
        failures.push(`missing source record: ${file}`);
    }
    // Never update or prune after a failed scan or build.
    expect(failures).toEqual([]);
    if (stagingDirectory) {
      if (expected.size > 0) mkdirSync(twinsDirectory, {recursive: true});
      for (const file of expected.keys())
        writeTwin(join(twinsDirectory, file), readFileSync(join(stagingDirectory, file)));
      for (const file of existingFiles)
        if (!referenced.has(file)) removeTwin(join(twinsDirectory, file));
    } else {
      expect(
        collectTwinDrift(expected, existing, referenced),
        'Run with UPDATE_LENS_TWINS=1 to regenerate docs/lens-records/twins'
      ).toEqual([]);
    }
  } finally {
    if (stagingDirectory) rmSync(stagingDirectory, {recursive: true, force: true});
  }
}

describe('documentation lens twins', () => {
  it('materializes canonical twins and detects missing, stale and orphan files', () => {
    materializeCanonicalTwins(listDocsMarkdownFiles());
  });

  it('every existing twin satisfies the accessible documentation contract and size budget', () => {
    const failures: string[] = [];
    for (const file of listTwinFiles()) {
      try {
        const text = readFileSync(join(TWINS_DIRECTORY, file), 'utf8');
        const size = Buffer.byteLength(text);
        const budget = file.endsWith('~thumbnail.json') ? 48 * 1024 : 256 * 1024;
        if (size > budget) failures.push(`${file}: ${size} bytes exceeds ${budget} bytes`);
        const twin: LensTwin = JSON.parse(text);
        failures.push(...collectTwinContractFailures(twin).map(failure => `${file}: ${failure}`));
      } catch (error) {
        failures.push(`${file}: ${String(error)}`);
      }
    }
    expect(failures).toEqual([]);
  });
});

describe('twin helpers', () => {
  it('a one-directive scenario typo cannot write or prune existing twins', () => {
    const directory = mkdtempSync(join(tmpdir(), 'lens-typo-'));
    const recordsDirectory = join(directory, 'records');
    const twinsDirectory = join(directory, 'twins');
    mkdirSync(recordsDirectory);
    mkdirSync(twinsDirectory);
    const documentFile = join(directory, 'page.md');
    writeFileSync(documentFile, '::lens{scenario=gpu-dataframe.class-breaxs rung=result}');
    writeFileSync(
      join(recordsDirectory, 'gpu-dataframe.class-breaks.json'),
      readFileSync(join(RECORDS_DIRECTORY, 'gpu-dataframe.class-breaks.json'))
    );
    writeFileSync(
      join(twinsDirectory, 'gpu-dataframe.class-breaks~result.json'),
      readFileSync(join(TWINS_DIRECTORY, 'gpu-dataframe.class-breaks~result.json'))
    );
    const writeTwin = vi.fn();
    const removeTwin = vi.fn();
    let error: string | undefined;
    try {
      try {
        materializeCanonicalTwins([documentFile], {
          update: true,
          recordsDirectory,
          twinsDirectory,
          pending: {},
          writeTwin,
          removeTwin
        });
      } catch (caught) {
        error = String(caught);
      }
      expect({
        error,
        writes: writeTwin.mock.calls.length,
        deletions: removeTwin.mock.calls.length
      }).toEqual({
        error: expect.stringContaining(
          'gpu-dataframe.class-breaxs: missing record without a documented pending entry'
        ),
        writes: 0,
        deletions: 0
      });
      // The guard must still allow regeneration when the same directive names the real capture.
      writeFileSync(documentFile, '::lens{scenario=gpu-dataframe.class-breaks rung=result}');
      materializeCanonicalTwins([documentFile], {
        update: true,
        recordsDirectory,
        twinsDirectory,
        pending: {},
        writeTwin,
        removeTwin
      });
      expect(writeTwin).toHaveBeenCalledTimes(2);
      expect(removeTwin).not.toHaveBeenCalled();
    } finally {
      rmSync(directory, {recursive: true, force: true});
    }
  });
  const recorded: DecodedLensFile = {
    schema: 'luma.lens-record/1',
    scenarioId: 'test.example',
    recordedAt: '2026-10-14T00:00:00.000Z',
    adapterLabel: 'Fixture adapter',
    record: {
      schema: 'luma.computation/1',
      id: 'test.example',
      runId: 'fixture-run',
      createdAt: '2026-10-14T00:00:00.000Z',
      adapter: {backend: 'webgpu'},
      domains: ['test'],
      graphs: [],
      axes: {axes: [], keySpaces: [], relations: []},
      refs: {formulas: [], sources: []},
      decisions: [],
      choices: []
    }
  };

  it('materializes a catalog thumbnail only for plain result directives', () => {
    expect(getThumbnailTwinFile({scenario: 'test.example', rung: 'result'})).toBe(
      'test.example~result~thumbnail.json'
    );
    expect(getThumbnailTwinFile({scenario: 'test.example', rung: 'datum'})).toBeUndefined();
    expect(
      getThumbnailTwinFile({scenario: 'test.example', rung: 'result', datum: '3'})
    ).toBeUndefined();
    const restored = fromRecordedLensFile(JSON.parse(JSON.stringify(recorded)));
    const thumbnail = JSON.parse(
      serializeTwin(restored, {scenario: 'test.example', rung: 'result'}, 'thumbnail')
    );
    expect(thumbnail).toMatchObject({rung: 'result', size: 'thumbnail'});
    expect(thumbnail.receipt).toBeUndefined();
    expect(thumbnail.example).toBeUndefined();
  });

  it('keeps zero pins in canonical keys and omits absent pins', () => {
    expect(twinKey({scenario: 'test.example', rung: 'table'})).toBe('test.example~table');
    expect(twinKey({scenario: 'test.example', rung: 'table', datum: '0', iteration: '00'})).toBe(
      'test.example~table~d0~i0'
    );
    expect(twinKey({scenario: 'test.example', rung: 'datum', view: 'neighbourhood-view'})).toBe(
      'test.example~datum~vneighbourhood-view'
    );
  });

  it('builds deterministic twins with numeric pins and host-independent captions', () => {
    const attributes = {scenario: 'test.example', rung: 'table', datum: '0', iteration: '0'};
    const restored = fromRecordedLensFile(JSON.parse(JSON.stringify(recorded)));
    const serialized = serializeTwin(restored, {...attributes, caption: 'First host caption'});
    expect(serialized).toBe(
      serializeTwin(restored, {...attributes, caption: 'Second host caption'})
    );
    const twin = JSON.parse(serialized);
    expect(twin).toMatchObject({
      schema: 'luma.lens-twin/1',
      scenario: 'test.example',
      rung: 'table',
      datum: 0,
      iteration: 0
    });
    expect(collectTwinContractFailures(twin)).toEqual([]);
    expect(serialized).toBe(`${JSON.stringify(twin, null, 2)}\n`);
    const withoutPins = JSON.parse(
      serializeTwin(restored, {scenario: 'test.example', rung: 'table'})
    );
    expect(withoutPins).not.toHaveProperty('datum');
    expect(withoutPins).not.toHaveProperty('iteration');
  });

  it('reuses one decoded record for full and thumbnail twins', () => {
    const restored = fromRecordedLensFile(JSON.parse(JSON.stringify(recorded)));
    const decode = vi.spyOn(recordingRegistry, 'fromRecordedLensFile');
    try {
      serializeTwin(restored, {scenario: 'test.example', rung: 'table'});
      serializeTwin(restored, {scenario: 'test.example', rung: 'table'}, 'thumbnail');
      expect(decode).not.toHaveBeenCalled();
    } finally {
      decode.mockRestore();
    }
  });

  it('rejects a checked-in twin whose source record has become pending', () => {
    expect(
      collectTwinDrift(
        new Map([
          ['missing.json', 'new'],
          ['stale.json', 'new']
        ]),
        new Map([
          ['stale.json', 'old'],
          ['orphan.json', 'old'],
          ['pending.json', 'existing']
        ]),
        new Set(['missing.json', 'stale.json', 'pending.json'])
      )
    ).toEqual([
      'missing source record: pending.json',
      'missing twin: missing.json',
      'orphan twin: orphan.json',
      'stale twin: stale.json'
    ]);
  });

  it('rejects empty required contract fields', () => {
    const twin: LensTwin = JSON.parse(
      serializeTwin(recorded, {scenario: 'test.example', rung: 'table'})
    );
    twin.keyNumber.label = ' ';
    twin.description = '';
    twin.provenance.adapter = '';
    twin.provenance.date = '';
    expect(collectTwinContractFailures(twin)).toEqual([
      'keyNumber.label must be a non-empty string',
      'description must be a non-empty string',
      'provenance.adapter must be a non-empty string',
      'provenance.date must be a non-empty string'
    ]);
  });
});
