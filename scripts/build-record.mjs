// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash, randomUUID} from 'node:crypto';
import {existsSync, readFileSync, renameSync, rmSync, writeFileSync} from 'node:fs';
import {rename, rm, writeFile} from 'node:fs/promises';
import {join, relative, resolve} from 'node:path';

const RECORD_FILE = 'dist/.build-record.json';
const hashFile = file => createHash('sha256').update(readFileSync(file)).digest('hex');

export function captureBuildInputs(moduleDirectory, project) {
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
  return Object.fromEntries(
    [...new Set(files)]
      .filter(file => existsSync(file))
      .sort()
      .map(file => [relative(moduleDirectory, file), hashFile(file)])
  );
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
  const currentInputs = captureBuildInputs(moduleDirectory, project);
  if (!compiledInputs || JSON.stringify(currentInputs) !== JSON.stringify(compiledInputs))
    throw new Error(
      `${moduleDirectory}: inputs changed during build; refusing to publish a build record`
    );
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
