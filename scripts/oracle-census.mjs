// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fs from 'node:fs';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {validateOracleManifest} from './run-oracle-fixture.mjs';

const TEST_DIRECTORY = 'modules/analysis/test';
const ORACLE_DIRECTORY = `${TEST_DIRECTORY}/oracles`;
const PROGRAM_EXTENSION = /\.(?:py|R|mjs|cpp)$/;

/**
 * @typedef {'verified'|'mismatching'|'unattributed'} HashStatus
 * @typedef {{fixture: string, status: HashStatus, generator?: string, attribution?: string, field?: string, recordedPath?: unknown, source?: string, recordedSha256?: unknown, currentSha256?: string}} HashRecord
 * @typedef {{total: number, verified: number, mismatching: number, unattributed: number}} HashCounts
 * @typedef {{records: HashRecord[], counts: HashCounts}} HashSummary
 * @typedef {{kind: 'fixture'|'unconfigured'|'inventory', profile: string, outputs?: string[], inputs?: string[], reason?: string}} OracleDeclaration
 * @typedef {{version: number, profiles: Record<string, {runtime: string}>, generators: Record<string, OracleDeclaration>}} OracleManifest
 * @typedef {{fixtures: number, programs: number, manifestPrograms: number, producers: number, unconfigured: number, inventory: number, declaredOutputs: number, existingOutputs: number, missingOutputs: number, configuredOutputs: number, existingConfiguredOutputs: number, missingConfiguredOutputs: number, fixturesWithoutProducer: number}} CensusCounts
 * @typedef {{runId: string, generator: string, completedAt: string, status: 'missing'|'partial'|'matching'|'differing'|'stale'}} RunRecord
 * @typedef {{runs: number, missing: number, partial: number, matching: number, differing: number, stale: number, comparedOutputs: number, matchingOutputs: number, differingOutputs: number, outputsWithoutEvidence: number}} RunCounts
 * @typedef {{counts: CensusCounts, missingOutputs: string[], generatorHashes: HashSummary, sourceHashes: HashSummary, regeneration: {runs: RunRecord[], counts: RunCounts}} OracleCensus
 * @typedef {{repositoryRoot: string, manifest: unknown, evidence?: unknown[]}} CensusInput
 */

// Exact R100 moves recorded by 58dd86d85d95e8c81bcc27a88f9bd7777fda0a7b.
// This is a documented identity mapping, not a basename search through historical paths.
const RECORDED_GENERATOR_MOVES = new Map([
  [
    'modules/graph/test/oracles/parity-wp5-round8-composites.py',
    `${ORACLE_DIRECTORY}/parity-wp5-round8-composites.py`
  ],
  [
    'modules/graph/test/oracles/parity-wp5-round8-curvature.py',
    `${ORACLE_DIRECTORY}/parity-wp5-round8-curvature.py`
  ]
]);

const hashBytes = bytes => createHash('sha256').update(bytes).digest('hex');

// Resolve recorded identities, never a basename extracted from an arbitrary path.
function resolveRecordedPath(repositoryRoot, recorded, generator = false) {
  if (typeof recorded !== 'string' || recorded.includes('\\')) return undefined;
  let relative = generator ? (RECORDED_GENERATOR_MOVES.get(recorded) ?? recorded) : recorded;
  if (generator && !relative.includes('/')) relative = `${ORACLE_DIRECTORY}/${relative}`;
  else if (generator && relative.startsWith('test/oracles/'))
    relative = `modules/analysis/${relative}`;
  const target = path.resolve(repositoryRoot, relative);
  const root = fs.realpathSync(repositoryRoot);
  if (
    !target.startsWith(`${path.resolve(repositoryRoot)}${path.sep}`) ||
    !fs.existsSync(target) ||
    !fs.statSync(target).isFile() ||
    !fs.realpathSync(target).startsWith(`${root}${path.sep}`)
  )
    return undefined;
  return path.relative(repositoryRoot, target).split(path.sep).join('/');
}

/** @param {HashRecord[]} records @returns {HashSummary} */
function summarizeHashes(records) {
  return {
    records,
    counts: {
      total: records.length,
      verified: records.filter(record => record.status === 'verified').length,
      mismatching: records.filter(record => record.status === 'mismatching').length,
      unattributed: records.filter(record => record.status === 'unattributed').length
    }
  };
}

function inspectProvenance(repositoryRoot, fixtures, programs, owners) {
  const generatorRecords = [];
  const sourceRecords = [];
  for (const fixture of fixtures) {
    const data = JSON.parse(
      fs.readFileSync(path.join(repositoryRoot, TEST_DIRECTORY, fixture), 'utf8')
    );
    if (!data || typeof data !== 'object' || Array.isArray(data)) continue;
    if (Object.hasOwn(data, 'generatorSha256')) {
      const candidates = new Map();
      for (const [recorded, attribution] of [
        [data.generator, 'explicit generator identity'],
        [data.generatorPath, 'recorded generator path'],
        [data.source, 'recorded source path'],
        ...(Array.isArray(data.command)
          ? data.command
              .slice(1)
              .filter(
                argument => typeof argument === 'string' && /\.(?:py|R|mjs|cpp)$/.test(argument)
              )
              .map(argument => [argument, 'recorded generator command'])
          : [])
      ]) {
        const resolved = resolveRecordedPath(repositoryRoot, recorded, true);
        const program = resolved?.startsWith(`${ORACLE_DIRECTORY}/`)
          ? resolved.slice(ORACLE_DIRECTORY.length + 1)
          : undefined;
        if (programs.includes(program)) candidates.set(program, attribution);
      }
      for (const owner of owners.get(fixture) ?? []) {
        if (
          programs.includes(owner) &&
          resolveRecordedPath(repositoryRoot, owner, true) &&
          !candidates.has(owner)
        )
          candidates.set(owner, 'declared output ownership');
      }
      const [generator, attribution] = candidates.size === 1 ? [...candidates][0] : [];
      generatorRecords.push({
        fixture,
        generator,
        attribution,
        status: generator
          ? hashBytes(fs.readFileSync(path.join(repositoryRoot, ORACLE_DIRECTORY, generator))) ===
            data.generatorSha256
            ? 'verified'
            : 'mismatching'
          : 'unattributed'
      });
    }
    const recordSource = (field, recorded, recordedDigest, fixtureRelative = false) => {
      const sourcePath =
        fixtureRelative && typeof recorded === 'string'
          ? `${TEST_DIRECTORY}/oracles/fixtures/${recorded}`
          : recorded;
      const resolved = resolveRecordedPath(repositoryRoot, sourcePath);
      const currentDigest = resolved
        ? hashBytes(fs.readFileSync(path.join(repositoryRoot, resolved)))
        : undefined;
      sourceRecords.push({
        fixture,
        field,
        recordedPath: recorded,
        source: resolved,
        recordedSha256: recordedDigest,
        currentSha256: currentDigest,
        status: resolved
          ? currentDigest === recordedDigest
            ? 'verified'
            : 'mismatching'
          : 'unattributed'
      });
    };
    // Corpus sources[] explicitly identifies both the repository-relative path and its digest.
    for (const [index, entry] of (Array.isArray(data.sources) ? data.sources : []).entries()) {
      if (
        entry &&
        typeof entry === 'object' &&
        Object.hasOwn(entry, 'path') &&
        Object.hasOwn(entry, 'sha256')
      )
        recordSource(`sources[${index}]`, entry.path, entry.sha256);
    }
    for (const [field, digest] of Object.entries(data)) {
      if (!field.endsWith('Sha256') || field === 'generatorSha256') continue;
      const hashMap = digest && typeof digest === 'object' && !Array.isArray(digest);
      const recordedHashes = hashMap
        ? Object.entries(digest)
        : [[data[field.slice(0, -6)], digest]];
      // parity-wp1-trend-reference.py:91 records path.name after reading DIRECTORY / 'fixtures/...'.
      const fixtureRelative =
        hashMap &&
        field === 'sourceSha256' &&
        fixture === 'oracles/fixtures/parity-wp1-trend-reference.json' &&
        data.generator === 'parity-wp1-trend-reference.py';
      for (const [recorded, recordedDigest] of recordedHashes)
        recordSource(field, recorded, recordedDigest, fixtureRelative);
    }
  }
  return {
    generatorHashes: summarizeHashes(generatorRecords),
    sourceHashes: summarizeHashes(sourceRecords)
  };
}

function inspectRunEvidence(repositoryRoot, manifest, existingOutputs, evidence) {
  const runs = [];
  const identifiers = new Set();
  const outputResults = new Map();
  const digestPattern = /^[a-f0-9]{64}$/;
  const invalid = () => {
    throw new Error('invalid oracle run evidence');
  };
  for (const receipt of evidence) {
    if (
      !receipt ||
      receipt.version !== 1 ||
      typeof receipt.runId !== 'string' ||
      !receipt.runId.trim() ||
      !Number.isFinite(Date.parse(receipt.startedAt)) ||
      !Number.isFinite(Date.parse(receipt.completedAt)) ||
      Date.parse(receipt.completedAt) < Date.parse(receipt.startedAt) ||
      !digestPattern.test(receipt.generatorSha256)
    )
      invalid();
    if (identifiers.has(receipt.runId))
      throw new Error(`duplicate oracle run evidence ${receipt.runId}`);
    identifiers.add(receipt.runId);
    const declaration = manifest.generators[receipt.generator];
    if (
      declaration?.kind !== 'fixture' ||
      !Array.isArray(receipt.declaredOutputs) ||
      JSON.stringify([...receipt.declaredOutputs].sort()) !==
        JSON.stringify([...declaration.outputs].sort()) ||
      !Array.isArray(receipt.outputs)
    )
      invalid();
    const seen = new Set();
    for (const output of receipt.outputs) {
      if (
        !output ||
        !declaration.outputs.includes(output.path) ||
        seen.has(output.path) ||
        typeof output.regenerated !== 'boolean' ||
        !['not-compared', 'matching', 'differing'].includes(output.comparison) ||
        (output.regenerated && !digestPattern.test(output.regeneratedSha256)) ||
        (!output.regenerated && output.regeneratedSha256 !== undefined) ||
        (output.committedSha256 !== undefined && !digestPattern.test(output.committedSha256))
      )
        invalid();
      seen.add(output.path);
      if (
        output.comparison !== 'not-compared' &&
        (!output.regenerated ||
          !digestPattern.test(output.committedSha256) ||
          (output.comparison === 'matching') !==
            (output.committedSha256 === output.regeneratedSha256))
      )
        invalid();
    }
    if (seen.size !== declaration.outputs.length) invalid();
    const regeneratedCount = receipt.outputs.filter(output => output.regenerated).length;
    const complete = regeneratedCount === declaration.outputs.length;
    if (
      (!complete && receipt.outputs.some(output => output.comparison !== 'not-compared')) ||
      (complete &&
        receipt.outputs.some(
          output => output.committedSha256 !== undefined && output.comparison === 'not-compared'
        ))
    )
      invalid();
    const generatorPath = resolveRecordedPath(repositoryRoot, receipt.generator, true);
    let stale =
      !generatorPath ||
      hashBytes(fs.readFileSync(path.join(repositoryRoot, generatorPath))) !==
        receipt.generatorSha256;
    for (const output of receipt.outputs) {
      if (output.comparison === 'not-compared') continue;
      const target = resolveRecordedPath(repositoryRoot, `${TEST_DIRECTORY}/${output.path}`);
      const currentHash = target
        ? hashBytes(fs.readFileSync(path.join(repositoryRoot, target)))
        : undefined;
      // --write may have accepted differing bytes after comparison; both snapshots are recorded.
      if (currentHash !== output.committedSha256 && currentHash !== output.regeneratedSha256)
        stale = true;
    }
    const status = stale
      ? 'stale'
      : regeneratedCount === 0
        ? 'missing'
        : !complete
          ? 'partial'
          : receipt.outputs.some(output => output.comparison === 'differing')
            ? 'differing'
            : receipt.outputs.every(output => output.comparison === 'matching')
              ? 'matching'
              : 'partial';
    runs.push({
      runId: receipt.runId,
      generator: receipt.generator,
      completedAt: receipt.completedAt,
      status
    });
    if (stale) continue;
    for (const output of receipt.outputs) {
      if (output.comparison === 'not-compared' || !existingOutputs.includes(output.path)) continue;
      // Across identified runs, a differing comparison remains visible even if another run matched.
      if (outputResults.get(output.path) !== 'differing')
        outputResults.set(output.path, output.comparison);
    }
  }
  return {
    runs,
    counts: {
      runs: runs.length,
      missing: runs.filter(run => run.status === 'missing').length,
      partial: runs.filter(run => run.status === 'partial').length,
      matching: runs.filter(run => run.status === 'matching').length,
      differing: runs.filter(run => run.status === 'differing').length,
      stale: runs.filter(run => run.status === 'stale').length,
      comparedOutputs: outputResults.size,
      matchingOutputs: [...outputResults.values()].filter(value => value === 'matching').length,
      differingOutputs: [...outputResults.values()].filter(value => value === 'differing').length,
      outputsWithoutEvidence: existingOutputs.length - outputResults.size
    }
  };
}

/**
 * Census populations are sets of paths, never sums of producer output counts.
 * @param {CensusInput} input
 * @returns {OracleCensus}
 */
export function buildOracleCensus({repositoryRoot, manifest, evidence = []}) {
  validateOracleManifest(manifest);
  const validatedManifest = /** @type {OracleManifest} */ (manifest);
  const oracleRoot = path.join(repositoryRoot, ORACLE_DIRECTORY);
  const programs = fs
    .readdirSync(oracleRoot)
    .filter(
      file => PROGRAM_EXTENSION.test(file) && fs.statSync(path.join(oracleRoot, file)).isFile()
    )
    .sort();
  const fixtures = fs
    .readdirSync(path.join(oracleRoot, 'fixtures'))
    .filter(
      file =>
        file.endsWith('.json') && fs.statSync(path.join(oracleRoot, 'fixtures', file)).isFile()
    )
    .sort()
    .map(file => `oracles/fixtures/${file}`);
  const owners = new Map();
  const configuredOwners = new Map();
  const declarations = Object.entries(validatedManifest.generators);
  for (const [generator, declaration] of declarations) {
    for (const output of declaration.outputs ?? []) {
      if (!owners.has(output)) owners.set(output, []);
      owners.get(output).push(generator);
      if (declaration.kind === 'fixture') {
        if (!configuredOwners.has(output)) configuredOwners.set(output, []);
        configuredOwners.get(output).push(generator);
      }
    }
  }
  const declaredOutputs = [...owners.keys()].sort();
  const existingOutputs = declaredOutputs.filter(output => {
    return resolveRecordedPath(repositoryRoot, `${TEST_DIRECTORY}/${output}`) !== undefined;
  });
  const configuredOutputs = [...configuredOwners.keys()].sort();
  const existingConfiguredOutputs = configuredOutputs.filter(output =>
    existingOutputs.includes(output)
  );
  const missingOutputs = declaredOutputs.filter(output => !existingOutputs.includes(output));
  return {
    counts: {
      fixtures: fixtures.length,
      programs: programs.length,
      manifestPrograms: declarations.length,
      producers: declarations.filter(([, declaration]) => declaration.kind === 'fixture').length,
      unconfigured: declarations.filter(([, declaration]) => declaration.kind === 'unconfigured')
        .length,
      inventory: declarations.filter(([, declaration]) => declaration.kind === 'inventory').length,
      declaredOutputs: declaredOutputs.length,
      existingOutputs: existingOutputs.length,
      missingOutputs: missingOutputs.length,
      configuredOutputs: configuredOutputs.length,
      existingConfiguredOutputs: existingConfiguredOutputs.length,
      missingConfiguredOutputs: configuredOutputs.length - existingConfiguredOutputs.length,
      fixturesWithoutProducer: fixtures.filter(file => !configuredOwners.has(file)).length
    },
    missingOutputs,
    regeneration: inspectRunEvidence(
      repositoryRoot,
      validatedManifest,
      existingConfiguredOutputs,
      evidence
    ),
    ...inspectProvenance(repositoryRoot, fixtures, programs, owners)
  };
}

/** @param {OracleCensus} census @returns {string} */
export function renderOracleCensus(census) {
  const count = census.counts;
  const generators = census.generatorHashes.counts;
  const sources = census.sourceHashes.counts;
  const regeneration = census.regeneration.counts;
  return [
    '<!-- oracle-census:start; generated by scripts/oracle-census.mjs -->',
    `- Fixture JSON files on disk: **${count.fixtures}**.`,
    `- Generator programs on disk: **${count.programs}**.`,
    `- Manifest programs: **${count.manifestPrograms}** (**${count.producers}** fixture producers, **${count.unconfigured}** unconfigured, **${count.inventory}** inventory).`,
    `- Distinct declared output paths: **${count.declaredOutputs}**; existing: **${count.existingOutputs}**; missing: **${count.missingOutputs}**.`,
    `- Configured output paths: **${count.configuredOutputs}**; existing: **${count.existingConfiguredOutputs}**; missing: **${count.missingConfiguredOutputs}**.`,
    `- Fixture JSON files with no configured producer: **${count.fixturesWithoutProducer}**.`,
    `- Generator hashes: **${generators.total}** recorded; verified: **${generators.verified}**; mismatching: **${generators.mismatching}**; unattributed: **${generators.unattributed}**.`,
    `- Recorded source path hashes: **${sources.total}**; verified: **${sources.verified}**; mismatching: **${sources.mismatching}**; unattributed: **${sources.unattributed}**.`,
    ...(regeneration.runs === 0
      ? [
          '- Run evidence: **0** identified runs; no outputs are certified as regenerated and compared.'
        ]
      : [
          `- Run evidence: **${regeneration.runs}** identified runs; missing: **${regeneration.missing}**; partial: **${regeneration.partial}**; matching: **${regeneration.matching}**; differing: **${regeneration.differing}**; stale against current bytes: **${regeneration.stale}**.`,
          ...census.regeneration.runs.map(
            run => `- Run \`${run.runId}\` (${run.generator}; ${run.completedAt}): ${run.status}.`
          )
        ]),
    `- Distinct outputs regenerated and compared: **${regeneration.comparedOutputs}**; matching: **${regeneration.matchingOutputs}**; differing: **${regeneration.differingOutputs}**; existing configured outputs without comparison evidence: **${regeneration.outputsWithoutEvidence}**.`,
    '<!-- oracle-census:end -->'
  ].join('\n');
}

/** @param {string} markdown @param {string} rendered @returns {string} */
export function updateOracleCensus(markdown, rendered) {
  const counts = [...markdown.matchAll(/^## Counts\n/gm)];
  const reproducing = [...markdown.matchAll(/^## Reproducing\n/gm)];
  if (counts.length !== 1 || reproducing.length !== 1 || counts[0].index >= reproducing[0].index)
    throw new Error(
      'expected exactly one Counts followed by exactly one Reproducing in GENERATORS.md'
    );
  const start = counts[0].index + counts[0][0].length;
  return `${markdown.slice(0, start)}\n${rendered}\n\nThe literal/inferred/unresolved split (542/204/2) is historical: it described 748 fixtures and 242 scripts.\n\n${markdown.slice(reproducing[0].index)}`;
}

function main() {
  const argumentsList = process.argv.slice(2);
  const evidence = [];
  const flags = new Set();
  for (let index = 0; index < argumentsList.length; index++) {
    const argument = argumentsList[index];
    if (argument === '--evidence' && argumentsList[index + 1])
      evidence.push(JSON.parse(fs.readFileSync(argumentsList[++index], 'utf8')));
    else if (['--write', '--check', '--json'].includes(argument)) flags.add(argument);
    else
      throw new Error(
        'usage: node scripts/oracle-census.mjs [--write | --check | --json] [--evidence <receipt>]...'
      );
  }
  if (flags.size > 1 || (evidence.length && (flags.has('--write') || flags.has('--check'))))
    throw new Error(
      'run evidence is report-only; the checked prose uses the reproducible census without local receipts'
    );
  const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const manifest = JSON.parse(
    fs.readFileSync(path.join(repositoryRoot, 'scripts/oracle-fixtures.json'), 'utf8')
  );
  const census = buildOracleCensus({repositoryRoot, manifest, evidence});
  if (flags.has('--json')) {
    console.log(JSON.stringify(census, null, 2));
    return;
  }
  const rendered = renderOracleCensus(census);
  const target = path.join(repositoryRoot, ORACLE_DIRECTORY, 'GENERATORS.md');
  const markdown = fs.readFileSync(target, 'utf8');
  const updated = updateOracleCensus(markdown, rendered);
  if (flags.has('--write')) fs.writeFileSync(target, updated);
  else if (flags.has('--check')) {
    if (markdown !== updated)
      throw new Error('oracle census drift; run node scripts/oracle-census.mjs --write');
  } else console.log(rendered);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    console.error(`oracle-census: ${error.message}`);
    process.exitCode = 1;
  }
}
