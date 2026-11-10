// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// PLAT-06. Regenerates one oracle fixture family and diffs it against the committed fixtures.
//
//   yarn oracle-fixture <generator> [--python <path>] [--rscript <path>] [--write] [--keep]
//   yarn oracle-fixture --list
//
// <generator> is a file in modules/analysis/test/oracles, with or without its extension
// (`statistics-markov`, `interpolation-gstat.R`). The generator runs in a sandbox: a copy-on-write
// clone of modules/analysis/test inside a temporary directory, with every other repository
// directory symlinked, so the committed fixtures are never touched while it runs. Files it wrote
// are compared with the committed ones and nothing is copied back unless --write is given.
//
// Exit codes: 0 identical or written, 1 differences (without --write), 2 environment problem
// (missing interpreter, Python package or R), 3 generator failure, 64 usage error.
import {spawnSync} from 'node:child_process';
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

/** Maps generator file to the fixture names GENERATORS.md lists for it (informational). */
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
        maximumAbsoluteDifference = Math.max(maximumAbsoluteDifference, Math.abs(value - after.get(key)));
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

/** Classifies interpreter stderr into an environment problem, or undefined for a generator failure. */
export function classifyFailure(extension, standardError) {
  if (extension === '.py') {
    const missing = standardError.match(/(?:ModuleNotFoundError|ImportError): No module named '([^']+)'/);
    if (missing) {
      return `missing Python package: ${missing[1].split('.')[0]} (pins: ${ORACLE_DIRECTORY}/requirements.txt; install it into a virtual environment and pass --python <venv>/bin/python)`;
    }
    const importFailure = standardError.match(/ImportError: (.+)/);
    if (importFailure) return `Python import failure: ${importFailure[1]}`;
    const metadata = standardError.match(/PackageNotFoundError: No package metadata was found for (\S+)/);
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

function listFiles(directory) {
  const files = [];
  const walk = current => {
    for (const entry of fs.readdirSync(current, {withFileTypes: true})) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile()) files.push(full);
    }
  };
  walk(directory);
  return files;
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
    else if (argument === '--python') options.python = argv[++index];
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

function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.error) {
    console.error(`oracle-fixture: ${options.error}`);
    return 64;
  }
  const oracleDirectory = path.join(REPOSITORY_ROOT, ORACLE_DIRECTORY);
  const expectedByGenerator = parseExpectedFixtures(
    fs.readFileSync(path.join(oracleDirectory, 'GENERATORS.md'), 'utf8')
  );
  if (options.list) {
    for (const file of fs.readdirSync(oracleDirectory).sort()) {
      if (!GENERATOR_EXTENSIONS.has(path.extname(file))) continue;
      const fixtures = expectedByGenerator.get(file) ?? [];
      console.log(`${file}${fixtures.length ? `  ->  ${fixtures.join(', ')}` : ''}`);
    }
    return 0;
  }

  const resolved = resolveGenerator(options.generator, oracleDirectory);
  if (resolved.error) {
    console.error(`oracle-fixture: ${resolved.error}`);
    return 64;
  }
  const extension = path.extname(resolved.file);
  let command;
  let commandArguments;
  const generatorPath = `${ORACLE_DIRECTORY}/${resolved.file}`;
  if (extension === '.py') {
    command = findExecutable([options.python, process.env.ORACLE_PYTHON, 'python3.14', 'python3']);
    if (!command) {
      console.error('oracle-fixture: no Python interpreter found; pass --python <path> or set ORACLE_PYTHON');
      return 2;
    }
    commandArguments = ['-I', generatorPath];
  } else if (extension === '.R') {
    command = findExecutable([options.rscript, process.env.ORACLE_RSCRIPT, 'Rscript']);
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

  const sandboxRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'oracle-fixture-'));
  const startTime = Date.now() - 2000;
  try {
    createSandbox(sandboxRoot);
    console.log(`oracle-fixture: ${[command, ...commandArguments].join(' ')} (sandbox ${sandboxRoot})`);
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
        console.error(`oracle-fixture: ${problem}\noracle-fixture: ${resolved.file} did not run; the committed fixtures are unchecked.`);
        return 2;
      }
      console.error(`${run.stderr}\noracle-fixture: ${resolved.file} failed with status ${run.status}`);
      return 3;
    }

    const sandboxTest = path.join(sandboxRoot, TEST_DIRECTORY);
    const written = listFiles(sandboxTest)
      .filter(file => fs.statSync(file).mtimeMs >= startTime)
      .filter(file => !file.includes('__pycache__') && path.basename(file) !== resolved.file);
    const expected = expectedByGenerator.get(resolved.file) ?? [];
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
      if (committedBytes.equals(regeneratedBytes)) {
        console.log(`IDENTICAL  ${label}`);
        continue;
      }
      differenceCount++;
      if (extension === '.R' || path.extname(file) !== '.json') {
        console.log(`DIFFERS    ${label} (${committedBytes.length} -> ${regeneratedBytes.length} bytes; not JSON)`);
        continue;
      }
      const diff = diffJson(JSON.parse(committedBytes), JSON.parse(regeneratedBytes));
      console.log(
        `DIFFERS    ${label}: ${diff.changed.length} changed (${diff.changedNumberCount} numeric, max abs ${diff.maximumAbsoluteDifference}), ${diff.added.length} added, ${diff.removed.length} removed${diff.metadataOnly ? ' (metadata only: versions or hashes)' : ''}`
      );
      for (const item of diff.changed.slice(0, MAXIMUM_REPORTED_DIFFERENCES)) {
        console.log(`             ${item.path}: ${describeValue(item.committed)} -> ${describeValue(item.regenerated)}`);
      }
      for (const key of diff.added.slice(0, 3)) console.log(`             + ${key}`);
      for (const key of diff.removed.slice(0, 3)) console.log(`             - ${key}`);
    }
    if (written.length === 0) console.log('oracle-fixture: the generator wrote no file');
    for (const name of expected) {
      if (!written.some(file => path.basename(file) === name)) {
        console.log(`NOT WRITTEN ${name} (listed in GENERATORS.md for ${resolved.file})`);
      }
    }

    if (options.write && differenceCount > 0) {
      for (const file of written) {
        const relative = path.relative(sandboxTest, file);
        const target = path.join(REPOSITORY_ROOT, TEST_DIRECTORY, relative);
        if (fs.existsSync(target) && fs.readFileSync(target).equals(fs.readFileSync(file))) continue;
        fs.mkdirSync(path.dirname(target), {recursive: true});
        fs.copyFileSync(file, target);
        console.log(`WROTE      ${TEST_DIRECTORY}/${relative}`);
      }
      return 0;
    }
    if (differenceCount > 0) {
      console.log('oracle-fixture: differences found; nothing was written (use --write to accept them).');
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
