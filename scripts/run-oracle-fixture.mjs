// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// PLAT-06. Regenerates one oracle fixture family and diffs it against the committed fixtures.
//
//   yarn oracle-fixture <generator> [--python <path>] [--rscript <path>] [--write] [--keep] [--evidence <new-path>]
//   yarn oracle-fixture --list
//
// <generator> is a file in modules/analysis/test/oracles, with or without its extension
// (`statistics-markov`, `interpolation-gstat.R`). The generator runs in a sandbox: a copy-on-write
// clone of modules/analysis/test inside a temporary directory, with every other repository
// directory symlinked, so normal fixture writes are staged. oracle-fixtures.json declares the
// interpreter profile and exact outputs; all must be recreated before any comparison/copy-back.
// Nothing is copied back unless --write is given. This is not an OS security sandbox.
//
// Exit codes: 0 identical or written, 1 differences (without --write), 2 environment problem
// (missing interpreter, Python package or R), 3 generator failure, 64 usage error.
import {spawnSync} from 'node:child_process';
import {createHash, randomUUID} from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const REPOSITORY_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_DIRECTORY = 'modules/analysis/test';
const ORACLE_DIRECTORY = `${TEST_DIRECTORY}/oracles`;
const GENERATOR_EXTENSIONS = new Set(['.py', '.R', '.mjs']);
const MAXIMUM_REPORTED_DIFFERENCES = 10;

/** Finds the generator script for a name such as `statistics-markov` or `interpolation-gstat.R`. */
export function resolveGenerator(name, oracleDirectory) {
  const candidates = fs
    .readdirSync(oracleDirectory)
    .filter(file => fs.statSync(path.join(oracleDirectory, file)).isFile())
    .filter(file => file === name || path.parse(file).name === name);
  const supported = candidates.filter(file => GENERATOR_EXTENSIONS.has(path.extname(file)));
  if (supported.length === 1) return {file: supported[0]};
  if (supported.length > 1) {
    return {error: `ambiguous generator ${name}: ${supported.join(', ')}; give the extension`};
  }
  if (candidates.length > 0) {
    return {
      error: `${candidates.join(', ')} is not runnable here (supported: ${[...GENERATOR_EXTENSIONS].join(' ')}); .cpp generators need CGAL and a compiler`
    };
  }
  return {error: `no generator named ${name} in ${ORACLE_DIRECTORY}`};
}

/** Reads the historical prose inventory; runtime declarations live in oracle-fixtures.json. */
export function parseExpectedFixtures(generatorsMarkdown) {
  const expected = new Map();
  let current = null;
  for (const line of generatorsMarkdown.split('\n')) {
    const heading = line.match(/^### `([^`]+)`/);
    if (heading) {
      current = heading[1];
      if (!expected.has(current)) expected.set(current, []);
    } else if (line.startsWith('## ')) {
      current = null;
    } else if (current) {
      for (const match of line.matchAll(/`([\w.-]+\.json)`/g)) {
        expected.get(current).push(match[1]);
      }
    }
  }
  return expected;
}

/** Flattens parsed JSON to a map from a path such as `a.b[3]` to a leaf value. */
export function flattenJson(value, prefix = '', leaves = new Map()) {
  if (Array.isArray(value)) {
    if (value.length === 0) leaves.set(`${prefix}[]`, '[]');
    value.forEach((item, index) => flattenJson(item, `${prefix}[${index}]`, leaves));
  } else if (value !== null && typeof value === 'object') {
    const keys = Object.keys(value);
    if (keys.length === 0) leaves.set(`${prefix}{}`, '{}');
    for (const key of keys) flattenJson(value[key], prefix ? `${prefix}.${key}` : key, leaves);
  } else {
    leaves.set(prefix, value);
  }
  return leaves;
}

/** Compares two parsed JSON documents leaf by leaf. */
export function diffJson(committed, regenerated) {
  const before = flattenJson(committed);
  const after = flattenJson(regenerated);
  const changed = [];
  const added = [];
  const removed = [];
  let maximumAbsoluteDifference = 0;
  let changedNumberCount = 0;
  for (const [key, value] of before) {
    if (!after.has(key)) {
      removed.push(key);
    } else if (!Object.is(after.get(key), value)) {
      changed.push({path: key, committed: value, regenerated: after.get(key)});
      if (typeof value === 'number' && typeof after.get(key) === 'number') {
        changedNumberCount++;
        maximumAbsoluteDifference = Math.max(
          maximumAbsoluteDifference,
          Math.abs(value - after.get(key))
        );
      }
    }
  }
  for (const key of after.keys()) if (!before.has(key)) added.push(key);
  const isMetadata = key => /^(versions|generatorSha256|python)\b/.test(key);
  return {
    changed,
    added,
    removed,
    changedNumberCount,
    maximumAbsoluteDifference,
    metadataOnly:
      changed.length + added.length + removed.length > 0 &&
      [...changed.map(item => item.path), ...added, ...removed].every(isMetadata)
  };
}

/**
 * @typedef {{path: string, regenerated: boolean, comparison: 'not-compared'|'matching'|'differing', committedSha256?: string, regeneratedSha256?: string}} OracleRunOutput
 * @typedef {{version: 1, runId: string, startedAt: string, completedAt: string, generator: string, generatorSha256: string, declaredOutputs: string[], outputs: OracleRunOutput[]}} OracleRunEvidence
 * @typedef {{path: string, committedBytes?: import('node:buffer').Buffer, regeneratedBytes?: import('node:buffer').Buffer}} OracleOutputBytes
 * @typedef {{runId: string, startedAt: string, completedAt: string, generator: string, generatorBytes: import('node:buffer').Buffer, declaredOutputs: string[], outputs: OracleOutputBytes[]}} OracleRunInput
 */

/**
 * A run receipt describes bytes actually recreated; partial runs certify no comparisons.
 * @param {OracleRunInput} input
 * @returns {OracleRunEvidence}
 */
export function createOracleRunEvidence({
  runId,
  startedAt,
  completedAt,
  generator,
  generatorBytes,
  declaredOutputs,
  outputs
}) {
  const hashBytes = bytes => createHash('sha256').update(bytes).digest('hex');
  const byPath = new Map(outputs.map(output => [output.path, output]));
  const complete = declaredOutputs.every(
    output => byPath.get(output)?.regeneratedBytes !== undefined
  );
  return {
    version: 1,
    runId,
    startedAt,
    completedAt,
    generator,
    generatorSha256: hashBytes(generatorBytes),
    declaredOutputs: [...declaredOutputs],
    outputs: declaredOutputs.map(output => {
      const bytes = byPath.get(output);
      const regenerated = bytes?.regeneratedBytes !== undefined;
      const compared = complete && bytes?.committedBytes !== undefined;
      return {
        path: output,
        regenerated,
        comparison: !compared
          ? 'not-compared'
          : bytes.committedBytes.equals(bytes.regeneratedBytes)
            ? 'matching'
            : 'differing',
        committedSha256:
          bytes?.committedBytes === undefined ? undefined : hashBytes(bytes.committedBytes),
        regeneratedSha256: regenerated ? hashBytes(bytes.regeneratedBytes) : undefined
      };
    })
  };
}

/** Classifies interpreter stderr into an environment problem, or undefined for a generator failure. */
export function classifyFailure(extension, standardError) {
  if (extension === '.py') {
    const missing = standardError.match(
      /(?:ModuleNotFoundError|ImportError): No module named '([^']+)'/
    );
    if (missing) {
      return `missing Python package: ${missing[1].split('.')[0]} (pins: ${ORACLE_DIRECTORY}/requirements.txt; install it into a virtual environment and pass --python <venv>/bin/python)`;
    }
    const importFailure = standardError.match(/ImportError: (.+)/);
    if (importFailure) return `Python import failure: ${importFailure[1]}`;
    const metadata = standardError.match(
      /PackageNotFoundError: No package metadata was found for (\S+)/
    );
    if (metadata) return `missing Python package: ${metadata[1]} (version lookup failed)`;
  }
  if (extension === '.R') {
    const missing = standardError.match(/there is no package called .([^'’‘’]+)./);
    if (missing) return `missing R package: ${missing[1]}`;
  }
  if (extension === '.mjs') {
    const missing = standardError.match(/Cannot find (?:package|module) '([^']+)'/);
    if (missing) return `missing Node module: ${missing[1]}`;
  }
  return undefined;
}

/** Builds the sandbox root: a clone of the test tree, symlinks to everything else. */
function createSandbox(sandboxRoot) {
  const mirror = relativeDirectory => {
    const source = path.join(REPOSITORY_ROOT, relativeDirectory);
    const target = path.join(sandboxRoot, relativeDirectory);
    fs.mkdirSync(target, {recursive: true});
    for (const entry of fs.readdirSync(source)) {
      const childRelative = relativeDirectory === '.' ? entry : path.join(relativeDirectory, entry);
      if (childRelative === TEST_DIRECTORY) continue;
      if (TEST_DIRECTORY.startsWith(`${childRelative}/`)) mirror(childRelative);
      else fs.symlinkSync(path.join(source, entry), path.join(target, entry));
    }
  };
  mirror('.');
  // COPYFILE_FICLONE shares blocks on APFS, btrfs and XFS and falls back to a plain copy.
  fs.cpSync(path.join(REPOSITORY_ROOT, TEST_DIRECTORY), path.join(sandboxRoot, TEST_DIRECTORY), {
    recursive: true,
    mode: fs.constants.COPYFILE_FICLONE,
    preserveTimestamps: true
  });
}

function parseArguments(argv) {
  const options = {write: false, keep: false, list: false, generator: undefined};
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === '--write') options.write = true;
    else if (argument === '--keep') options.keep = true;
    else if (argument === '--list') options.list = true;
    else if (argument === '--evidence') {
      options.evidence = argv[++index];
      if (!options.evidence || options.evidence.startsWith('--'))
        return {error: '--evidence needs a new receipt path'};
    } else if (argument === '--python') options.python = argv[++index];
    else if (argument === '--rscript') options.rscript = argv[++index];
    else if (argument.startsWith('--')) return {error: `unknown option ${argument}`};
    else if (options.generator === undefined) options.generator = argument;
    else return {error: `unexpected argument ${argument}`};
  }
  if (!options.list && !options.generator) return {error: 'give a generator name, or --list'};
  return options;
}

function findExecutable(candidates) {
  for (const candidate of candidates.filter(Boolean)) {
    const probe = spawnSync(candidate, ['--version'], {encoding: 'utf8'});
    if (!probe.error && probe.status === 0) return candidate;
  }
  return undefined;
}

function describeValue(value) {
  const text = JSON.stringify(value);
  return text && text.length > 60 ? `${text.slice(0, 57)}...` : text;
}

/** Validate declarations without executing any generator. Paths are relative to the test tree. */
export function validateOracleManifest(manifest) {
  if (manifest.version !== 1 || !manifest.generators || !manifest.profiles)
    throw new Error('expected version 1, profiles and generators');
  const validatePath = relative => {
    if (
      typeof relative !== 'string' ||
      !relative ||
      relative.includes('\\') ||
      path.isAbsolute(relative) ||
      relative.split('/').some(part => part === '..' || part === '.' || !part)
    )
      throw new Error(`invalid test-relative fixture path ${relative}`);
  };
  for (const [file, declaration] of Object.entries(manifest.generators)) {
    if (
      path.basename(file) !== file ||
      !['fixture', 'inventory', 'unconfigured'].includes(declaration.kind)
    )
      throw new Error(`invalid generator declaration ${file}`);
    const profile = manifest.profiles[declaration.profile];
    if (!profile || !['python', 'rscript', 'node', 'unsupported'].includes(profile.runtime))
      throw new Error(`${file}: invalid interpreter profile`);
    if (declaration.kind !== 'fixture') {
      if (typeof declaration.reason !== 'string' || !declaration.reason.trim())
        throw new Error(`${file}: a reason is required for ${declaration.kind}`);
      if (declaration.outputs !== undefined) {
        if (
          !Array.isArray(declaration.outputs) ||
          new Set(declaration.outputs).size !== declaration.outputs.length
        )
          throw new Error(`${file}: declared outputs must be a unique path array`);
        for (const output of declaration.outputs) validatePath(output);
      }
      continue;
    }
    if (
      profile.runtime === 'unsupported' ||
      !Array.isArray(declaration.outputs) ||
      !declaration.outputs.length
    )
      throw new Error(`${file}: fixture outputs and a runnable profile are required`);
    if (declaration.inputs !== undefined && !Array.isArray(declaration.inputs))
      throw new Error(`${file}: inputs must be an array`);
    for (const relative of [...declaration.outputs, ...(declaration.inputs ?? [])])
      validatePath(relative);
    if (
      new Set(declaration.outputs).size !== declaration.outputs.length ||
      declaration.outputs.some(output => declaration.inputs?.includes(output))
    )
      throw new Error(`${file}: outputs must be unique and separate from inputs`);
    for (const [option, environment] of Object.entries(declaration.toolArguments ?? {})) {
      if (
        !/^--[a-z][a-z-]*$/.test(option) ||
        typeof environment !== 'string' ||
        !/^[A-Z][A-Z_0-9]*$/.test(environment)
      )
        throw new Error(`${file}: invalid tool argument declaration`);
    }
  }
}

function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.error) {
    console.error(`oracle-fixture: ${options.error}`);
    return 64;
  }
  const oracleDirectory = path.join(REPOSITORY_ROOT, ORACLE_DIRECTORY);
  let manifest;
  try {
    manifest = JSON.parse(
      fs.readFileSync(path.join(REPOSITORY_ROOT, 'scripts/oracle-fixtures.json'), 'utf8')
    );
    validateOracleManifest(manifest);
  } catch (error) {
    console.error(`oracle-fixture: invalid oracle-fixtures.json: ${error.message}`);
    return 64;
  }
  if (options.list) {
    for (const [file, declaration] of Object.entries(manifest.generators).sort(([left], [right]) =>
      left.localeCompare(right)
    )) {
      if (declaration.kind === 'fixture')
        console.log(`${file} [${declaration.profile}]  ->  ${declaration.outputs.join(', ')}`);
    }
    return 0;
  }

  const resolved = resolveGenerator(options.generator, oracleDirectory);
  if (resolved.error) {
    console.error(`oracle-fixture: ${resolved.error}`);
    return 64;
  }
  const declaration = manifest.generators[resolved.file];
  if (declaration?.kind !== 'fixture') {
    console.error(
      `oracle-fixture: ${resolved.file} has no declared fixture outputs for execution (${declaration?.kind ?? 'unconfigured'}): ${declaration?.reason ?? 'add a reviewed declaration to scripts/oracle-fixtures.json'}`
    );
    return 64;
  }
  const profile = manifest.profiles[declaration.profile];
  if (
    !profile ||
    !['python', 'rscript', 'node'].includes(profile.runtime) ||
    !Array.isArray(declaration.outputs) ||
    !declaration.outputs.length
  ) {
    console.error(
      `oracle-fixture: ${resolved.file}: invalid profile or empty outputs in oracle-fixtures.json`
    );
    return 64;
  }
  const expectedPaths = [...new Set(declaration.outputs)];
  for (const relative of [...expectedPaths, ...(declaration.inputs ?? [])]) {
    if (
      typeof relative !== 'string' ||
      !relative ||
      relative.includes('\\') ||
      path.isAbsolute(relative) ||
      relative.split('/').some(part => part === '..' || part === '.' || !part)
    ) {
      console.error(`oracle-fixture: invalid test-relative fixture path ${relative}`);
      return 64;
    }
    const committedPath = path.join(REPOSITORY_ROOT, TEST_DIRECTORY, relative);
    const testRoot = fs.realpathSync(path.join(REPOSITORY_ROOT, TEST_DIRECTORY));
    if (
      !fs.existsSync(committedPath) ||
      !fs.statSync(committedPath).isFile() ||
      !fs.realpathSync(committedPath).startsWith(`${testRoot}${path.sep}`)
    ) {
      console.error(
        `oracle-fixture: expected fixture ${path.basename(relative)} is missing or escapes the committed test tree: ${relative}`
      );
      return 3;
    }
  }
  const extension = path.extname(resolved.file);
  let command;
  let commandArguments;
  const generatorPath = `${ORACLE_DIRECTORY}/${resolved.file}`;
  if (profile.runtime === 'python') {
    const requested = options.python ?? process.env.ORACLE_PYTHON;
    command = findExecutable(
      requested !== undefined ? [requested] : (profile.executables ?? ['python3.14', 'python3'])
    );
    if (!command) {
      console.error(
        'oracle-fixture: no Python interpreter found; pass --python <path> or set ORACLE_PYTHON'
      );
      return 2;
    }
    if (profile.pythonVersion) {
      const version = spawnSync(
        command,
        [
          '-I',
          '-c',
          'import sys; print(str(sys.version_info.major) + "." + str(sys.version_info.minor))'
        ],
        {encoding: 'utf8'}
      );
      if (version.status !== 0 || version.stdout.trim() !== profile.pythonVersion) {
        console.error(
          `oracle-fixture: ${declaration.profile} requires Python ${profile.pythonVersion}; ${command} reported ${version.stdout?.trim() || 'no version'}`
        );
        return 2;
      }
    }
    commandArguments = ['-I', generatorPath];
  } else if (profile.runtime === 'rscript') {
    const requested = options.rscript ?? process.env.ORACLE_RSCRIPT;
    command = findExecutable(requested !== undefined ? [requested] : ['Rscript']);
    if (!command) {
      console.error(
        `oracle-fixture: R is not installed (Rscript not found), so ${resolved.file} cannot run; pass --rscript <path> or set ORACLE_RSCRIPT. The committed fixture is unchecked.`
      );
      return 2;
    }
    commandArguments = [generatorPath];
  } else {
    command = process.execPath;
    commandArguments = [generatorPath];
  }
  if (
    (profile.runtime === 'python' && extension !== '.py') ||
    (profile.runtime === 'rscript' && extension !== '.R') ||
    (profile.runtime === 'node' && extension !== '.mjs')
  ) {
    console.error(`oracle-fixture: ${resolved.file}: runtime does not match generator extension`);
    return 64;
  }
  for (const [option, environment] of Object.entries(declaration.toolArguments ?? {})) {
    const value = process.env[environment];
    if (!value) {
      console.error(
        `oracle-fixture: ${resolved.file} needs ${environment} for ${option}; profile ${declaration.profile} (${profile.description ?? ''})`
      );
      return 2;
    }
    commandArguments.push(option, value);
  }

  const startedAt = new Date().toISOString();
  const runId = randomUUID();
  const generatorBytes = fs.readFileSync(path.join(REPOSITORY_ROOT, generatorPath));
  const recordEvidence = outputs => {
    if (!options.evidence) return;
    const receipt = createOracleRunEvidence({
      runId,
      startedAt,
      completedAt: new Date().toISOString(),
      generator: resolved.file,
      generatorBytes,
      declaredOutputs: expectedPaths,
      outputs
    });
    // Never overwrite an existing file, including a fixture or an earlier run's evidence.
    fs.writeFileSync(options.evidence, `${JSON.stringify(receipt, null, 2)}\n`, {flag: 'wx'});
    console.log(`oracle-fixture: run ${runId} evidence written to ${options.evidence}`);
  };
  const sandboxRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'oracle-fixture-'));
  try {
    createSandbox(sandboxRoot);
    const sandboxTest = path.join(sandboxRoot, TEST_DIRECTORY);
    // Remove staged copies first: an untouched old fixture is never proof of regeneration.
    for (const relative of expectedPaths) fs.unlinkSync(path.join(sandboxTest, relative));
    console.log(
      `oracle-fixture: ${[command, ...commandArguments].join(' ')} (sandbox ${sandboxRoot})`
    );
    const run = spawnSync(command, commandArguments, {
      cwd: sandboxRoot,
      encoding: 'utf8',
      maxBuffer: 256 * 1024 * 1024
    });
    if (run.error) {
      console.error(`oracle-fixture: could not start ${command}: ${run.error.message}`);
      return 2;
    }
    if (run.status !== 0) {
      const problem = classifyFailure(extension, run.stderr ?? '');
      if (problem) {
        console.error(
          `oracle-fixture: ${problem}\noracle-fixture: ${resolved.file} did not run; the committed fixtures are unchecked.`
        );
        return 2;
      }
      console.error(
        `${run.stderr}\noracle-fixture: ${resolved.file} failed with status ${run.status}`
      );
      return 3;
    }

    const written = expectedPaths.map(relative => path.join(sandboxTest, relative));
    const sandboxTestRoot = fs.realpathSync(sandboxTest);
    const missing = written.filter(
      file =>
        !fs.existsSync(file) ||
        !fs.statSync(file).isFile() ||
        !fs.realpathSync(file).startsWith(`${sandboxTestRoot}${path.sep}`)
    );
    if (missing.length) {
      for (const file of missing)
        console.error(
          `NOT WRITTEN ${path.relative(sandboxTest, file)} (required by oracle-fixtures.json for ${resolved.file})`
        );
      console.error('oracle-fixture: incomplete generation; no fixtures compared or copied back');
      if (options.evidence)
        recordEvidence(
          written.map((file, index) => ({
            path: expectedPaths[index],
            committedBytes: fs.readFileSync(
              path.join(REPOSITORY_ROOT, TEST_DIRECTORY, expectedPaths[index])
            ),
            regeneratedBytes: missing.includes(file) ? undefined : fs.readFileSync(file)
          }))
        );
      return 3;
    }
    const evidenceOutputs = [];
    let differenceCount = 0;
    for (const file of written) {
      const relative = path.relative(sandboxTest, file);
      const committedPath = path.join(REPOSITORY_ROOT, TEST_DIRECTORY, relative);
      const label = `${TEST_DIRECTORY}/${relative}`;
      if (!fs.existsSync(committedPath)) {
        console.log(`NEW        ${label} (no committed file)`);
        differenceCount++;
        continue;
      }
      const committedBytes = fs.readFileSync(committedPath);
      const regeneratedBytes = fs.readFileSync(file);
      if (options.evidence)
        evidenceOutputs.push({path: relative, committedBytes, regeneratedBytes});
      if (committedBytes.equals(regeneratedBytes)) {
        console.log(`IDENTICAL  ${label}`);
        continue;
      }
      differenceCount++;
      if (extension === '.R' || path.extname(file) !== '.json') {
        console.log(
          `DIFFERS    ${label} (${committedBytes.length} -> ${regeneratedBytes.length} bytes; not JSON)`
        );
        continue;
      }
      const diff = diffJson(JSON.parse(committedBytes), JSON.parse(regeneratedBytes));
      console.log(
        `DIFFERS    ${label}: ${diff.changed.length} changed (${diff.changedNumberCount} numeric, max abs ${diff.maximumAbsoluteDifference}), ${diff.added.length} added, ${diff.removed.length} removed${diff.metadataOnly ? ' (metadata only: versions or hashes)' : ''}`
      );
      for (const item of diff.changed.slice(0, MAXIMUM_REPORTED_DIFFERENCES)) {
        console.log(
          `             ${item.path}: ${describeValue(item.committed)} -> ${describeValue(item.regenerated)}`
        );
      }
      for (const key of diff.added.slice(0, 3)) console.log(`             + ${key}`);
      for (const key of diff.removed.slice(0, 3)) console.log(`             - ${key}`);
    }
    recordEvidence(evidenceOutputs);
    if (options.write && differenceCount > 0) {
      for (const file of written) {
        const relative = path.relative(sandboxTest, file);
        const target = path.join(REPOSITORY_ROOT, TEST_DIRECTORY, relative);
        if (fs.existsSync(target) && fs.readFileSync(target).equals(fs.readFileSync(file)))
          continue;
        fs.mkdirSync(path.dirname(target), {recursive: true});
        fs.copyFileSync(file, target);
        console.log(`WROTE      ${TEST_DIRECTORY}/${relative}`);
      }
      return 0;
    }
    if (differenceCount > 0) {
      console.log(
        'oracle-fixture: differences found; nothing was written (use --write to accept them).'
      );
      return 1;
    }
    return 0;
  } finally {
    if (options.keep) console.log(`oracle-fixture: sandbox kept at ${sandboxRoot}`);
    else fs.rmSync(sandboxRoot, {recursive: true, force: true});
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = main();
}
