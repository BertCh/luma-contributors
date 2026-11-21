// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Fast workspace artifact check. Build first; packed-consumer verification is a separate gate.
import {existsSync, readdirSync, readFileSync, realpathSync, statSync} from 'node:fs';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {getTypeScriptProjectInputs, getTypeScriptOutputPlan} from './typescript-project-inputs.mjs';
import {checkBuildRecord} from './build-record.mjs';

const require = createRequire(import.meta.url);
// Entry points imported only for their registration side effects.
const SIDE_EFFECT_ENTRIES = new Set([
  '@luma-contributors/lenses/computation-lenses/scenarios/index'
]);
const failures = [];
let checked = 0;

function getLoaderConditions(loader) {
  const conditions = new Set(['node', loader]);
  if (process.features.require_module) conditions.add('module-sync');
  let addons = true;
  const environmentOptions =
    (process.env.NODE_OPTIONS ?? '').match(/(?:[^\s"]+|"(?:\\.|[^"])*")+/g) ?? [];
  const arguments_ = [
    ...environmentOptions.map(value =>
      value.replace(/"((?:\\.|[^"])*)"/g, (_match, quoted) => quoted.replace(/\\(.)/g, '$1'))
    ),
    ...process.execArgv
  ];
  for (let index = 0; index < arguments_.length; index++) {
    const argument = arguments_[index];
    if (argument === '--no-addons') addons = false;
    else if (argument === '--addons') addons = true;
    else if (argument === '--conditions' || argument === '-C') conditions.add(arguments_[++index]);
    else if (argument.startsWith('--conditions='))
      conditions.add(argument.slice('--conditions='.length));
    else if (argument.startsWith('-C') && argument.length > 2) conditions.add(argument.slice(2));
  }
  if (addons) conditions.add('node-addons');
  return conditions;
}

function selectTarget(target, conditions) {
  if (typeof target === 'string' || target === null) return target;
  if (Array.isArray(target)) {
    for (const value of target) {
      const selected = selectTarget(value, conditions);
      if (selected !== undefined && selected !== null) return selected;
    }
    return null;
  }
  for (const [condition, value] of Object.entries(target)) {
    if (condition === 'default' || conditions.has(condition)) {
      const selected = selectTarget(value, conditions);
      if (selected !== undefined) return selected;
    }
  }
}

function listTargets(target) {
  if (typeof target === 'string') return [target];
  if (target === null) return [];
  if (typeof target !== 'object') throw new Error('unsupported export condition');
  return Object.values(target).flatMap(listTargets);
}

function replaceTargets(target, replacement) {
  if (typeof target === 'string') return target.replaceAll('*', replacement);
  if (target === null) return null;
  if (Array.isArray(target)) return target.map(value => replaceTargets(value, replacement));
  return Object.fromEntries(
    Object.entries(target).map(([condition, value]) => [
      condition,
      replaceTargets(value, replacement)
    ])
  );
}

function hasRequireCondition(target) {
  return (
    target !== null &&
    typeof target === 'object' &&
    (Object.hasOwn(target, 'require') || Object.values(target).some(hasRequireCondition))
  );
}

function verifyResolvedTarget(specifier, resolved, directory, target, condition) {
  const expected = realpathSync(path.resolve(directory, target));
  const selected = realpathSync(resolved);
  if (selected !== expected)
    throw new Error(
      `${specifier}: ${condition} target mismatch; expected ${expected}, resolved ${selected}`
    );
}

function listFiles(directory) {
  if (!existsSync(directory)) return [];
  return readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    const file = path.join(directory, entry.name);
    return entry.isDirectory() ? listFiles(file) : [file];
  });
}

function expandEntries(key, target, directory) {
  if (!key.includes('*')) return [[key, target]];
  const replacements = new Set();
  for (const pattern of listTargets(target)) {
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
    .map(replacement => [key.replaceAll('*', replacement), replaceTargets(target, replacement)]);
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
  for (const {source, compilerOutputs, transformedOutputs} of getTypeScriptOutputPlan(
    project,
    directory
  )) {
    for (const output of [...compilerOutputs, ...transformedOutputs]) {
      const emitted = path.join(directory, 'dist', output);
      if (!existsSync(emitted) || !statSync(emitted).isFile())
        failures.push(`${packageJson.name}: missing build output dist/${output}`);
      else if (
        buildRecord.present
          ? !buildRecord.outputFiles.includes(`dist/${output}`)
          : statSync(emitted).mtimeMs < Math.max(statSync(source).mtimeMs, metadataTime)
      ) {
        failures.push(
          `${packageJson.name}: stale build output dist/${output}; rebuild after current source/config`
        );
      }
    }
  }
  for (const [key, target] of Object.entries(packageJson.exports)) {
    try {
      for (const [entryKey, entryTarget] of expandEntries(key, target, directory)) {
        const specifier =
          entryKey === '.' ? packageJson.name : `${packageJson.name}/${entryKey.slice(2)}`;
        for (const value of listTargets(entryTarget)) {
          if (typeof value !== 'string')
            throw new Error(`${specifier}: unsupported export condition`);
          const relativeTarget = path.relative(directory, path.resolve(directory, value));
          if (
            !value.startsWith('./') ||
            relativeTarget === '..' ||
            relativeTarget.startsWith(`..${path.sep}`) ||
            path.isAbsolute(relativeTarget)
          )
            throw new Error(`${specifier}: invalid package export target ${value}`);
          if (
            !existsSync(path.resolve(directory, value)) ||
            !statSync(path.resolve(directory, value)).isFile()
          )
            throw new Error(`${specifier}: missing target ${value}`);
        }
        // Conditional export keys are tested in manifest order, including node before default.
        const importTarget = selectTarget(entryTarget, getLoaderConditions('import'));
        const requireTarget = hasRequireCondition(entryTarget)
          ? selectTarget(entryTarget, getLoaderConditions('require'))
          : undefined;
        // Preflight both native targets before either loader can run initialization.
        if (importTarget && /\.(mjs|cjs|js)$/.test(importTarget)) {
          verifyResolvedTarget(
            specifier,
            fileURLToPath(import.meta.resolve(specifier)),
            directory,
            importTarget,
            'ESM'
          );
        }
        if (requireTarget && /\.(mjs|cjs|js)$/.test(requireTarget)) {
          verifyResolvedTarget(
            specifier,
            require.resolve(specifier),
            directory,
            requireTarget,
            'CommonJS'
          );
        }
        let names;
        if (importTarget && /\.(mjs|cjs|js)$/.test(importTarget)) {
          const moduleExports = await import(specifier);
          names = Object.keys(moduleExports).sort();
          if (!names.length && !SIDE_EFFECT_ENTRIES.has(specifier))
            throw new Error(`${specifier}: ESM entry has no exports`);
        }
        if (requireTarget && /\.(mjs|cjs|js)$/.test(requireTarget)) {
          const commonJSNames = Object.keys(require(specifier)).sort();
          if (!commonJSNames.length && !SIDE_EFFECT_ENTRIES.has(specifier))
            throw new Error(`${specifier}: CommonJS entry has no exports`);
          if (names && JSON.stringify(commonJSNames) !== JSON.stringify(names)) {
            throw new Error(
              `${specifier}: CommonJS names [${commonJSNames}] differ from ESM [${names}]`
            );
          }
        }
        if (importTarget && !/\.(mjs|cjs|js)$/.test(importTarget)) import.meta.resolve(specifier);
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
