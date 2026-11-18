// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Fast workspace artifact check. Build first; packed-consumer verification is a separate gate.
import {existsSync, readdirSync, readFileSync, statSync} from 'node:fs';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {getTypeScriptProjectInputs} from './typescript-project-inputs.mjs';
import {checkBuildRecord} from './build-record.mjs';

const require = createRequire(import.meta.url);
// Entry points imported only for their registration side effects.
const SIDE_EFFECT_ENTRIES = new Set([
  '@luma-contributors/lenses/computation-lenses/scenarios/index'
]);
const failures = [];
let checked = 0;

function listFiles(directory) {
  if (!existsSync(directory)) return [];
  return readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    const file = path.join(directory, entry.name);
    return entry.isDirectory() ? listFiles(file) : [file];
  });
}

function expandEntries(key, target, directory) {
  if (!key.includes('*')) return [[key, target]];
  const conditions = typeof target === 'string' ? {default: target} : target;
  const replacements = new Set();
  for (const pattern of Object.values(conditions)) {
    if (typeof pattern !== 'string' || !pattern.includes('*')) continue;
    const [prefix, suffix] = pattern.split('*');
    const searchDirectory = prefix.endsWith('/')
      ? path.resolve(directory, prefix)
      : path.dirname(path.resolve(directory, prefix));
    for (const file of listFiles(searchDirectory)) {
      const relative = `./${path.relative(directory, file).split(path.sep).join('/')}`;
      if (relative.startsWith(prefix) && relative.endsWith(suffix)) {
        replacements.add(relative.slice(prefix.length, suffix ? -suffix.length : undefined));
      }
    }
  }
  if (!replacements.size) throw new Error(`wildcard ${key} has no built targets`);
  return [...replacements]
    .sort()
    .map(replacement => [
      key.replaceAll('*', replacement),
      typeof target === 'string'
        ? target.replaceAll('*', replacement)
        : Object.fromEntries(
            Object.entries(target).map(([condition, value]) => [
              condition,
              typeof value === 'string' ? value.replaceAll('*', replacement) : value
            ])
          )
    ]);
}

for (const moduleName of ['analysis', 'lenses', 'kepler']) {
  const directory = fileURLToPath(new URL(`../modules/${moduleName}/`, import.meta.url));
  const packageJson = JSON.parse(readFileSync(path.join(directory, 'package.json'), 'utf8'));
  const project = getTypeScriptProjectInputs(path.join(directory, 'tsconfig.json'), [
    path.join(directory, 'src')
  ]);
  const sources = project.files.filter(file => !/\.d\.[cm]?ts$/.test(file));
  const buildRecord = checkBuildRecord(directory, project);
  if (buildRecord.present && !buildRecord.valid)
    failures.push(`${packageJson.name}: stale build record; inputs or emitted content changed`);
  if (!sources.length) failures.push(`${packageJson.name}: no current source files found`);
  const metadataTime = Math.max(
    0,
    ...project.configFiles.map(file => statSync(file).mtimeMs),
    ...['package.json', 'tsconfig.json', '../../tsconfig.json', '../../tsconfig.build.json'].map(
      file =>
        existsSync(path.join(directory, file)) ? statSync(path.join(directory, file)).mtimeMs : 0
    )
  );
  for (const source of sources) {
    const sourceRelative = path.relative(path.join(directory, 'src'), source);
    const relative = sourceRelative.replace(/\.[^.]+$/, '');
    const extensions = source.endsWith('.json')
      ? ['.json']
      : /\.[cm][jt]s$/.test(source)
        ? source.endsWith('mts') || source.endsWith('mjs')
          ? ['.mjs', '.d.mts']
          : ['.cjs', '.d.cts']
        : /\.[jt]sx$/.test(source) && [1, 3].includes(project.options.jsx)
          ? ['.jsx', '.d.ts']
          : ['.js', '.cjs', '.d.ts'];
    for (const extension of extensions) {
      const emitted = path.join(directory, 'dist', relative + extension);
      if (!existsSync(emitted) || !statSync(emitted).isFile())
        failures.push(`${packageJson.name}: missing build output dist/${relative}${extension}`);
      else if (
        buildRecord.present
          ? !buildRecord.outputFiles.includes(`dist/${relative}${extension}`)
          : statSync(emitted).mtimeMs < Math.max(statSync(source).mtimeMs, metadataTime)
      ) {
        failures.push(
          `${packageJson.name}: stale build output dist/${relative}${extension}; rebuild after current source/config`
        );
      }
    }
  }
  for (const [key, target] of Object.entries(packageJson.exports)) {
    try {
      for (const [entryKey, entryTarget] of expandEntries(key, target, directory)) {
        const specifier =
          entryKey === '.' ? packageJson.name : `${packageJson.name}/${entryKey.slice(2)}`;
        const conditions = typeof entryTarget === 'string' ? {default: entryTarget} : entryTarget;
        for (const value of Object.values(conditions)) {
          if (typeof value !== 'string')
            throw new Error(`${specifier}: unsupported export condition`);
          if (
            !value.startsWith('./') ||
            path.relative(directory, path.resolve(directory, value)).startsWith('..')
          )
            throw new Error(`${specifier}: invalid package export target ${value}`);
          if (
            !existsSync(path.resolve(directory, value)) ||
            !statSync(path.resolve(directory, value)).isFile()
          )
            throw new Error(`${specifier}: missing target ${value}`);
        }
        const importTarget = conditions.import ?? conditions.default;
        let names;
        if (
          importTarget &&
          (typeof entryTarget !== 'string' || /\.(mjs|cjs|js)$/.test(importTarget))
        ) {
          const moduleExports = await import(specifier);
          names = Object.keys(moduleExports).sort();
          if (!names.length && !SIDE_EFFECT_ENTRIES.has(specifier))
            throw new Error(`${specifier}: ESM entry has no exports`);
        }
        if (conditions.require) {
          const commonJSNames = Object.keys(require(specifier)).sort();
          if (!commonJSNames.length && !SIDE_EFFECT_ENTRIES.has(specifier))
            throw new Error(`${specifier}: CommonJS entry has no exports`);
          if (names && JSON.stringify(commonJSNames) !== JSON.stringify(names)) {
            throw new Error(
              `${specifier}: CommonJS names [${commonJSNames}] differ from ESM [${names}]`
            );
          }
        }
        if (typeof entryTarget === 'string' && !/\.(mjs|cjs|js)$/.test(entryTarget))
          import.meta.resolve(specifier);
        checked++;
      }
    } catch (error) {
      failures.push(`${packageJson.name} ${key}: ${String(error.message).split('\n')[0]}`);
    }
  }
}
process.stdout.write(`verified ${checked} entry points\n`);
if (!checked) failures.push('no entry points checked');
if (failures.length) {
  process.stdout.write(failures.join('\n') + '\n');
  process.exitCode = 1;
}
