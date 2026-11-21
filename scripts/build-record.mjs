// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {execFileSync} from 'node:child_process';
import {createHash, randomUUID} from 'node:crypto';
import {
  existsSync,
  readdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  writeFileSync
} from 'node:fs';
import {rename, rm, writeFile} from 'node:fs/promises';
import {createRequire, isBuiltin} from 'node:module';
import {basename, dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typeScript from 'typescript';
import {listTypeScriptFreshnessInputs} from './typescript-project-inputs.mjs';

const RECORD_FILE = 'dist/.build-record.json';
const hashFile = file => createHash('sha256').update(readFileSync(file)).digest('hex');
export const BUILD_TOOL_PREFIX = '@tool:';
export const DEPENDENCY_INPUT_PREFIX = '@dependency:';
// Invocation context stays in memory; packed records contain hashes and logical identifiers only.
const invocationContexts = new WeakMap();

function findAncestorFile(directory, file) {
  for (let current = directory; ; current = dirname(current)) {
    const candidate = join(current, file);
    if (existsSync(candidate)) return candidate;
    if (dirname(current) === current) return undefined;
  }
}

function packageInformation(file) {
  const metadataFile = findAncestorFile(dirname(realpathSync(file)), 'package.json');
  const directory = dirname(metadataFile);
  const metadata = JSON.parse(readFileSync(metadataFile, 'utf8'));
  return {directory, metadata};
}

/** Ask the installed API to select its executable, intercepting spawn before any service starts. */
export function getEsbuildExecutable(
  esbuildEntry = createRequire(import.meta.url).resolve('esbuild')
) {
  const entry = esbuildEntry.startsWith('file:') ? fileURLToPath(esbuildEntry) : esbuildEntry;
  const selection = JSON.parse(
    execFileSync(
      process.execPath,
      [
        '-e',
        `
const childProcess = require('node:child_process');
childProcess.spawn = (command, arguments_) => {
  process.stdout.write(JSON.stringify({command, arguments_}));
  process.exit(0);
};
require(process.argv[1]).transform('', {}).catch(error => { console.error(error); process.exit(1); });
`,
        entry
      ],
      {encoding: 'utf8', env: process.env}
    )
  );
  return resolve(
    /^(?:node|node.exe)$/.test(basename(selection.command))
      ? selection.arguments_[0]
      : selection.command
  );
}

function captureBuildTools(moduleDirectory, project, compilerBinary) {
  const inputs = {};
  // Release snapshots copy only verification scripts; their enclosing checkout supplies the engine.
  const engine = findAncestorFile(dirname(fileURLToPath(import.meta.url)), 'scripts/build.mjs');
  if (engine)
    for (const name of [
      'build.mjs',
      'build-commonjs.mjs',
      'build-record.mjs',
      'typescript-project-inputs.mjs'
    ]) {
      const file = join(dirname(engine), name);
      if (existsSync(file)) inputs[`${BUILD_TOOL_PREFIX}${name}`] = hashFile(file);
    }
  const compiler = compilerBinary ?? findAncestorFile(moduleDirectory, 'node_modules/.bin/tspc');
  if (compiler) inputs[`${BUILD_TOOL_PREFIX}compiler-entry`] = hashFile(compiler);
  const require = createRequire(join(moduleDirectory, 'package.json'));
  const toolRequire = createRequire(engine ?? import.meta.url);
  const packages = new Set();
  function addPackage(entry) {
    const {directory, metadata} = packageInformation(entry);
    if (packages.has(directory)) return;
    packages.add(directory);
    const files = [];
    function visit(current) {
      for (const item of readdirSync(current, {withFileTypes: true})) {
        if (item.name === 'node_modules' || item.name.startsWith('.')) continue;
        const file = join(current, item.name);
        if (item.isDirectory()) visit(file);
        else if (/\.(?:[cm]?js|json|node|wasm|d\.[cm]?ts)$/.test(file) || !item.name.includes('.'))
          files.push([relative(directory, file), hashFile(file)]);
      }
    }
    visit(directory);
    const key = `${BUILD_TOOL_PREFIX}package:${metadata.name ?? basename(directory)}@${metadata.version ?? 'unknown'}`;
    const content = createHash('sha256')
      .update(JSON.stringify(files.sort(([left], [right]) => left.localeCompare(right))))
      .digest('hex');
    // Distinct installs of the same package version remain visible without retaining their paths.
    inputs[key] = [...(inputs[key] ?? []), content];
    const packageRequire = createRequire(join(directory, 'package.json'));
    for (const name of Object.keys(metadata.dependencies ?? {})) {
      try {
        addPackage(packageRequire.resolve(name));
      } catch (error) {
        if (error.code !== 'MODULE_NOT_FOUND' && error.code !== 'ERR_PACKAGE_PATH_NOT_EXPORTED')
          throw error;
      }
    }
  }
  for (const name of ['typescript', 'esbuild', 'ts-patch']) {
    try {
      addPackage(toolRequire.resolve(name));
    } catch (error) {
      if (error.code !== 'MODULE_NOT_FOUND') throw error;
    }
  }
  inputs[`${BUILD_TOOL_PREFIX}esbuild-executable`] = hashFile(
    getEsbuildExecutable(toolRequire.resolve('esbuild'))
  );
  for (const plugin of project.options.plugins ?? []) {
    if (!plugin.transform) continue;
    const entry = require.resolve(plugin.transform, {
      paths: [dirname(project.options.configFilePath ?? join(moduleDirectory, 'tsconfig.json'))]
    });
    const consumerPackage = packageInformation(join(moduleDirectory, 'package.json')).directory;
    const pluginPackage = packageInformation(entry).directory;
    // A local plugin is code, not the consuming package tree (which contains generated dist).
    if (
      pluginPackage === consumerPackage ||
      plugin.transform.startsWith('.') ||
      plugin.transform.startsWith('/')
    ) {
      const visited = new Set();
      function visitRuntime(file) {
        file = realpathSync(file);
        if (visited.has(file)) return;
        visited.add(file);
        const key = `${BUILD_TOOL_PREFIX}plugin:${plugin.transform.startsWith('/') ? 'local' : plugin.transform}/${relative(dirname(entry), file)}`;
        inputs[key] = hashFile(file);
        if (!/\.[cm]?[jt]s$/.test(file)) return;
        const runtimeRequire = createRequire(file);
        const references = typeScript.preProcessFile(readFileSync(file, 'utf8'), true, true);
        for (const imported of references.importedFiles) {
          if (isBuiltin(imported.fileName)) continue;
          const resolved = runtimeRequire.resolve(imported.fileName);
          if (packageInformation(resolved).directory === pluginPackage) visitRuntime(resolved);
          else addPackage(resolved);
        }
      }
      visitRuntime(entry);
      inputs[
        `${BUILD_TOOL_PREFIX}plugin-metadata:${plugin.transform.startsWith('/') ? 'local' : plugin.transform}`
      ] = hashFile(join(pluginPackage, 'package.json'));
    } else addPackage(entry);
  }
  return Object.fromEntries(
    Object.entries(inputs)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, value]) => [
        key,
        Array.isArray(value)
          ? createHash('sha256')
              .update(JSON.stringify([...new Set(value)].sort()))
              .digest('hex')
          : value
      ])
  );
}

function dependencyIdentity(file, moduleDirectory) {
  const workspacePath = relative(resolve(moduleDirectory, '../..'), realpathSync(file));
  if (!workspacePath.startsWith('..') && !workspacePath.split('/').includes('node_modules'))
    return `${DEPENDENCY_INPUT_PREFIX}workspace/${workspacePath}`;
  const {directory, metadata} = packageInformation(file);
  return `${DEPENDENCY_INPUT_PREFIX}${metadata.name ?? basename(directory)}@${metadata.version ?? 'unknown'}/${relative(directory, realpathSync(file)).split('\\').join('/')}`;
}

/** @param {string | undefined} [compilerBinary] */
export function captureBuildInputs(moduleDirectory, project, compilerBinary = undefined) {
  const files = [
    ...project.files,
    ...(project.configFiles ?? []),
    ...[
      'package.json',
      'tsconfig.json',
      '../../package.json',
      '../../tsconfig.json',
      '../../tsconfig.build.json'
    ].map(file => resolve(moduleDirectory, file))
  ];
  const inputs = {...captureBuildTools(moduleDirectory, project, compilerBinary)};
  const dependencyFiles = new Set(listTypeScriptFreshnessInputs(project));
  const workspaceDirectory = resolve(moduleDirectory, '../..');
  for (const file of [...new Set(files)].filter(file => existsSync(file)).sort()) {
    const workspacePath = relative(workspaceDirectory, realpathSync(file));
    if (workspacePath.startsWith('..') || workspacePath.split('/').includes('node_modules'))
      dependencyFiles.add(file);
    else inputs[relative(moduleDirectory, file)] = hashFile(file);
  }
  const dependencies = new Map();
  for (const file of dependencyFiles) {
    const key = dependencyIdentity(file, moduleDirectory);
    dependencies.set(key, [...(dependencies.get(key) ?? []), hashFile(file)]);
  }
  for (const [key, hashes] of dependencies)
    inputs[key] = createHash('sha256')
      .update(JSON.stringify([...new Set(hashes)].sort()))
      .digest('hex');
  const snapshot = Object.fromEntries(
    Object.entries(inputs).sort(([left], [right]) => left.localeCompare(right))
  );
  invocationContexts.set(snapshot, {compilerBinary});
  return snapshot;
}

export function writeFileAtomic(file, contents) {
  const temporary = `${file}.${randomUUID()}.tmp`;
  try {
    writeFileSync(temporary, contents, {flag: 'wx'});
    renameSync(temporary, file);
  } finally {
    rmSync(temporary, {force: true});
  }
}

export async function writeFileAtomicAsync(file, contents) {
  const temporary = `${file}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporary, contents, {flag: 'wx'});
    await rename(temporary, file);
  } finally {
    await rm(temporary, {force: true});
  }
}

export function checkBuildRecord(moduleDirectory, project, inputSnapshot = undefined) {
  const file = join(moduleDirectory, RECORD_FILE);
  if (!existsSync(file))
    return {present: false, valid: false, outputsValid: false, outputFiles: []};
  try {
    const record = JSON.parse(readFileSync(file, 'utf8'));
    if (record.version !== 1 || !record.inputs || !record.outputs)
      throw new Error('invalid record');
    const outputFiles = Object.keys(record.outputs);
    const outputsValid =
      outputFiles.length > 0 &&
      outputFiles.every(output => {
        if (!output.startsWith('dist/') || output.split('/').includes('..')) return false;
        return (
          existsSync(join(moduleDirectory, output)) &&
          hashFile(join(moduleDirectory, output)) === record.outputs[output]
        );
      });
    const currentInputs = inputSnapshot ?? captureBuildInputs(moduleDirectory, project);
    const changedInputs = [
      ...new Set([...Object.keys(currentInputs), ...Object.keys(record.inputs)])
    ].filter(file => currentInputs[file] !== record.inputs[file]);
    const inputsValid = changedInputs.length === 0;
    return {
      present: true,
      valid: inputsValid && outputsValid,
      outputsValid,
      outputFiles,
      changedInputs,
      outputHashes: record.outputs
    };
  } catch {
    return {present: true, valid: false, outputsValid: false, outputFiles: []};
  }
}

export function validateBuildInputs(moduleDirectory, project, compiledInputs) {
  const currentInputs = captureBuildInputs(
    moduleDirectory,
    project,
    invocationContexts.get(compiledInputs)?.compilerBinary
  );
  if (!compiledInputs || JSON.stringify(currentInputs) !== JSON.stringify(compiledInputs)) {
    const changed = [
      ...new Set([...Object.keys(currentInputs), ...Object.keys(compiledInputs ?? {})])
    ].filter(key => currentInputs[key] !== compiledInputs?.[key]);
    throw new Error(
      `${moduleDirectory}: inputs changed during build; refusing to publish a build record (${changed.slice(0, 8).join(', ')}${changed.length > 8 ? ', …' : ''})`
    );
  }
}

/** Publish only after the compiler and every CommonJS transform completed successfully. */
export function writeBuildRecord(moduleDirectory, project, expectedOutputs, compiledInputs) {
  validateBuildInputs(moduleDirectory, project, compiledInputs);
  const outputs = Object.fromEntries(
    [...expectedOutputs]
      .sort()
      .map(file => [`dist/${file}`, hashFile(join(moduleDirectory, 'dist', file))])
  );
  writeFileAtomic(
    join(moduleDirectory, RECORD_FILE),
    `${JSON.stringify({version: 1, inputs: compiledInputs, outputs}, null, 2)}\n`
  );
}
