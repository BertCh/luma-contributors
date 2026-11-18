// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {execFileSync} from 'node:child_process';
import {createHash, randomUUID} from 'node:crypto';
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  realpathSync,
  renameSync,
  rmSync,
  symlinkSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {join, relative, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {buildPackages} from './build.mjs';
import {writeFileAtomic} from './build-record.mjs';
import {getChildNodeOptions} from './verification-policy.mjs';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const MODULES = ['analysis', 'lenses', 'kepler'];
const ROOT_FILES = ['package.json', 'tsconfig.json', 'tsconfig.build.json'];

function includePackageFile(moduleDirectory, file) {
  const path = relative(moduleDirectory, file);
  return (
    !['dist', 'test', 'node_modules', '.git'].includes(path.split(sep)[0]) &&
    !file.endsWith('.tsbuildinfo')
  );
}

function fingerprintSources(rootDirectory) {
  const hash = createHash('sha256');
  function visit(file) {
    hash.update(relative(rootDirectory, file));
    const entries = readdirSync(file, {withFileTypes: true}).sort((left, right) =>
      left.name.localeCompare(right.name)
    );
    for (const entry of entries) {
      const path = join(file, entry.name);
      const moduleDirectory = join(
        rootDirectory,
        'modules',
        relative(join(rootDirectory, 'modules'), path).split(sep)[0]
      );
      if (!includePackageFile(moduleDirectory, path)) continue;
      hash.update(relative(rootDirectory, path));
      if (entry.isDirectory()) visit(path);
      else if (entry.isFile()) hash.update(readFileSync(path));
      else throw new Error(`release snapshots require ordinary files/directories: ${path}`);
    }
  }
  for (const file of ROOT_FILES) hash.update(readFileSync(join(rootDirectory, file)));
  for (const moduleName of MODULES) visit(join(rootDirectory, 'modules', moduleName));
  return hash.digest('hex');
}

function fingerprintOutputs(rootDirectory) {
  return Object.fromEntries(
    MODULES.map(moduleName => {
      const directory = join(rootDirectory, 'modules', moduleName, 'dist');
      const hash = createHash('sha256');
      function visit(parent) {
        for (const entry of readdirSync(parent, {withFileTypes: true}).sort((left, right) =>
          left.name.localeCompare(right.name)
        )) {
          const file = join(parent, entry.name);
          const path = relative(directory, file);
          if (entry.isDirectory()) {
            hash.update(JSON.stringify([path, 'directory']));
            visit(file);
          } else if (entry.isFile()) {
            const contents = readFileSync(file);
            hash.update(JSON.stringify([path, contents.byteLength]));
            hash.update(contents);
          } else throw new Error(`verified output must be an ordinary file/directory: ${file}`);
        }
      }
      visit(directory);
      return [moduleName, hash.digest('hex')];
    })
  );
}

/** Install all three verified output trees, rolling back ordinary filesystem errors. */
export function promotePackages(
  rootDirectory,
  releaseDirectory,
  move = renameSync,
  commitRecords = () => {}
) {
  const previousDirectory = join(releaseDirectory, 'previous');
  mkdirSync(previousDirectory, {recursive: true});
  const journal = {status: 'promoting', packages: []};
  const journalFile = join(releaseDirectory, 'promotion.json');
  const save = () => writeFileAtomic(journalFile, `${JSON.stringify(journal, null, 2)}\n`);
  try {
    for (const moduleName of MODULES) {
      const destination = join(rootDirectory, 'modules', moduleName, 'dist');
      const staged = join(releaseDirectory, 'workspace/modules', moduleName, 'dist');
      const previous = join(previousDirectory, moduleName);
      const cache = join(rootDirectory, 'modules', moduleName, 'tsconfig.tsbuildinfo');
      const previousCache = join(previousDirectory, `${moduleName}.tsbuildinfo`);
      const item = {
        destination,
        staged,
        previous,
        cache,
        previousCache,
        cacheBackedUp: false,
        backedUp: false,
        installed: false
      };
      journal.packages.push(item);
      save();
      if (existsSync(cache)) {
        move(cache, previousCache);
        item.cacheBackedUp = true;
        save();
      }
      if (existsSync(destination)) {
        move(destination, previous);
        item.backedUp = true;
        save();
      }
      move(staged, destination);
      item.installed = true;
      save();
    }
    journal.status = 'promoted';
    save();
    // Publication is the final fallible operation; rollback stays available until it succeeds.
    commitRecords();
  } catch (error) {
    const failures = [error];
    for (const item of [...journal.packages].reverse()) {
      try {
        if (item.installed) move(item.destination, item.staged);
        if (item.backedUp) move(item.previous, item.destination);
        if (item.cacheBackedUp) move(item.previousCache, item.cache);
      } catch (rollbackError) {
        failures.push(rollbackError);
      }
    }
    journal.status = failures.length === 1 ? 'rolled-back' : 'recovery-required';
    save();
    throw new AggregateError(
      failures,
      `release promotion failed: ${error.message}; ${journal.status}; journal: ${journalFile}`
    );
  }
}

async function buildRelease(promote) {
  const environment = {...process.env, NODE_OPTIONS: getChildNodeOptions()};
  const metadataDirectory = join(ROOT, '.typecheck');
  mkdirSync(metadataDirectory, {recursive: true});
  const lock = join(metadataDirectory, 'release-build.lock');
  try {
    mkdirSync(lock);
  } catch {
    throw new Error(
      `release build lock exists: ${lock}; inspect its owner before removing a stale lock`
    );
  }
  writeFileSync(
    join(lock, 'owner.json'),
    JSON.stringify({pid: process.pid, startedAt: new Date().toISOString()})
  );
  let releaseDirectory;
  try {
    const originalFingerprint = fingerprintSources(ROOT);
    const parentDirectory = join(metadataDirectory, 'release-builds');
    mkdirSync(parentDirectory, {recursive: true});
    releaseDirectory = mkdtempSync(join(parentDirectory, 'build-'));
    const workspace = join(releaseDirectory, 'workspace');
    mkdirSync(workspace);
    for (const file of ROOT_FILES)
      cpSync(join(ROOT, file), join(workspace, file), {preserveTimestamps: true});
    for (const moduleName of MODULES) {
      const source = join(ROOT, 'modules', moduleName);
      cpSync(source, join(workspace, 'modules', moduleName), {
        recursive: true,
        preserveTimestamps: true,
        filter: file => includePackageFile(source, file)
      });
      const scopeDirectory = join(workspace, 'node_modules/@luma-contributors');
      mkdirSync(scopeDirectory, {recursive: true});
      symlinkSync(join(workspace, 'modules', moduleName), join(scopeDirectory, moduleName), 'dir');
    }
    mkdirSync(join(workspace, 'scripts'));
    for (const file of [
      'verify-package-exports.mjs',
      'typescript-project-inputs.mjs',
      'build-record.mjs'
    ])
      cpSync(join(ROOT, 'scripts', file), join(workspace, 'scripts', file));
    if (fingerprintSources(workspace) !== originalFingerprint)
      throw new Error('source/config/package inputs changed while taking the release snapshot');
    await buildPackages(workspace, join(ROOT, 'node_modules/.bin/tspc'), environment);
    const verifiedOutputs = fingerprintOutputs(workspace);
    execFileSync(process.execPath, [join(workspace, 'scripts/verify-package-exports.mjs')], {
      cwd: workspace,
      env: environment,
      stdio: 'inherit'
    });
    const temporaryDirectory = mkdtempSync(join(tmpdir(), 'release-packed-'));
    try {
      const output = execFileSync(
        process.execPath,
        [
          join(ROOT, 'scripts/check-packed-consumer.mjs'),
          '--package-directory',
          join(workspace, 'modules/analysis'),
          '--keep-artifact'
        ],
        {
          cwd: workspace,
          encoding: 'utf8',
          env: {...environment, TMPDIR: temporaryDirectory},
          stdio: ['ignore', 'pipe', 'inherit']
        }
      );
      process.stdout.write(output);
      const tarball = output.match(/^checked tarball: (.+)$/m)?.[1];
      if (!tarball || !existsSync(tarball))
        throw new Error('packed gate did not return its checked tarball');
      cpSync(tarball, join(releaseDirectory, 'analysis.tgz'));
    } finally {
      rmSync(temporaryDirectory, {recursive: true, force: true});
    }
    if (fingerprintSources(ROOT) !== originalFingerprint)
      throw new Error(
        'source/config/package inputs changed during release build; previous outputs preserved'
      );
    if (fingerprintSources(workspace) !== originalFingerprint)
      throw new Error(
        'staged source/config/package inputs changed during artifact verification; previous outputs preserved'
      );
    const record = {
      directory: releaseDirectory,
      status: promote ? 'promoted' : 'staged',
      sourceFingerprint: originalFingerprint,
      outputDigests: verifiedOutputs,
      tarball: join(releaseDirectory, 'analysis.tgz')
    };
    const buildFile = join(releaseDirectory, 'build.json');
    const latestFile = join(metadataDirectory, 'release-build-latest.json');
    const buildTemporary = `${buildFile}.${randomUUID()}.tmp`;
    const latestTemporary = `${latestFile}.${randomUUID()}.tmp`;
    try {
      const contents = `${JSON.stringify(record, null, 2)}\n`;
      writeFileSync(buildTemporary, contents, {flag: 'wx'});
      writeFileSync(latestTemporary, contents, {flag: 'wx'});
      const commitRecords = () => {
        const installedOutputs = fingerprintOutputs(promote ? ROOT : workspace);
        if (JSON.stringify(installedOutputs) !== JSON.stringify(verifiedOutputs))
          throw new Error('verified output trees changed after export verification');
        renameSync(buildTemporary, buildFile);
        try {
          renameSync(latestTemporary, latestFile);
        } catch (error) {
          rmSync(buildFile, {force: true});
          throw error;
        }
      };
      if (promote) promotePackages(ROOT, releaseDirectory, renameSync, commitRecords);
      else commitRecords();
    } finally {
      rmSync(buildTemporary, {force: true});
      rmSync(latestTemporary, {force: true});
    }
    process.stdout.write(
      `release build ${record.status} at: ${releaseDirectory}\nBuild/package gates passed; full Node/GPU qualification is separate.\n`
    );
  } catch (error) {
    if (releaseDirectory)
      process.stderr.write(`failed release stage retained for diagnosis: ${releaseDirectory}\n`);
    throw error;
  } finally {
    rmSync(lock, {recursive: true, force: true});
  }
}

if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const arguments_ = process.argv.slice(2);
    if (arguments_.length > 1 || arguments_.some(argument => argument !== '--promote'))
      throw new Error('Usage: yarn build:release [--promote]');
    await buildRelease(arguments_.includes('--promote'));
  } catch (error) {
    process.stderr.write(`release build failed: ${error.message}\n`);
    process.exitCode = 1;
  }
}
