// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Usage: diff.mjs <old.json> <new.json> [<old node_modules/@luma.gl> <new node_modules/@luma.gl>]
//
// With two inventories from exports.mjs, reports added/removed entries and export names, as before.
// A name-only diff misses a changed signature or changed arithmetic under an unchanged name (the
// shared grid coordinate WGSL changed this way at 10.0.0-alpha.16), so when the two installed
// package roots are also given it appends two sections for every name exported by an entry in
// both versions:
//   CHANGED DECLARATIONS: the SHA-256 of the exported symbol's declaration text in the `.d.ts`
//     (all declarations of the aliased target, whitespace collapsed, comments excluded) differs.
//   CHANGED STRING VALUES: the SHA-256 of an exported string constant's runtime value differs, such
//     as WGSL source, including one level of string properties of exported plain objects (shader
//     module `source`, `vs`, `fs`), reported as `name.property`.
// Runtime values come from importing each entry's JavaScript target (`import`, `module`,
// `default`, `node` condition, in that order) in one child process per root, because luma.gl
// refuses to load two versions into one process; an entry that cannot be imported prints
// `UNLOADED ENTRY <root> <entry> <message>` and contributes no string values. Hashes are printed
// as their first 12 hex digits. Unchanged hashes print nothing.
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {existsSync, readFileSync} from 'node:fs';
import {join} from 'node:path';
import typescript from 'typescript';

const RESULT_MARKER = '\u0000diff-runtime-result\u0000';
const RUNTIME_SCRIPT = `
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
const hash = text => createHash('sha256').update(text).digest('hex').slice(0, 12);
const results = {};
for (const [entry, filePath] of JSON.parse(readFileSync(0, 'utf8'))) {
  try {
    const module = await import(pathToFileURL(filePath).href);
    const values = {};
    for (const [name, value] of Object.entries(module)) {
      if (typeof value === 'string') values[name] = hash(value);
      else if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype)
        for (const [property, propertyValue] of Object.entries(value))
          if (typeof propertyValue === 'string') values[name + '.' + property] = hash(propertyValue);
    }
    results[entry] = values;
  } catch (error) {
    results[entry] = String(error?.message ?? error).split('\\n')[0];
  }
}
process.stdout.write(${JSON.stringify(RESULT_MARKER)} + JSON.stringify(results));
process.exit(0);
`;

const [previousPath, currentPath, previousRoot, currentRoot] = process.argv.slice(2);
if (!previousPath || !currentPath || Boolean(previousRoot) !== Boolean(currentRoot))
  throw new Error(
    'Usage: diff.mjs <old.json> <new.json> [<old node_modules/@luma.gl> <new node_modules/@luma.gl>]'
  );
const previous = JSON.parse(readFileSync(previousPath, 'utf8'));
const current = JSON.parse(readFileSync(currentPath, 'utf8'));
for (const entry of new Set([...Object.keys(previous), ...Object.keys(current)])) {
  if (!Object.hasOwn(previous, entry)) {
    console.log('NEW ENTRY', entry, Object.keys(current[entry] || {}).length);
    continue;
  }
  if (!Object.hasOwn(current, entry)) {
    console.log('REMOVED ENTRY', entry);
    continue;
  }
  if (current[entry] === null || previous[entry] === null) {
    if (current[entry] !== previous[entry])
      console.log(current[entry] === null ? 'UNRESOLVED ENTRY' : 'RESOLVED ENTRY', entry);
    continue;
  }
  const added = Object.keys(current[entry]).filter(name => !(name in previous[entry]));
  const removed = Object.keys(previous[entry]).filter(name => !(name in current[entry]));
  if (added.length || removed.length)
    console.log(
      `${entry}\n  + ${added.map(name => `${name}(${current[entry][name].kind})`).join(', ')}\n  - ${removed.join(', ')}`
    );
}

if (previousRoot && currentRoot) {
  const sharedEntries = Object.keys(current).filter(
    entry => previous[entry] && current[entry] && Object.hasOwn(previous, entry)
  );
  const previousFingerprints = await getFingerprints(previousRoot, sharedEntries);
  const currentFingerprints = await getFingerprints(currentRoot, sharedEntries);
  const changedDeclarations = [];
  const changedValues = [];
  for (const entry of sharedEntries) {
    const before = previousFingerprints.get(entry);
    const after = currentFingerprints.get(entry);
    if (!before || !after) continue;
    for (const [name, hash] of after.declarations) {
      const previousHash = before.declarations.get(name);
      if (previousHash && previousHash !== hash)
        changedDeclarations.push(
          `  ${entry} ${name}(${current[entry][name]?.kind ?? 'type'}) ${previousHash} -> ${hash}`
        );
    }
    for (const [name, hash] of after.values) {
      const previousHash = before.values.get(name);
      if (previousHash && previousHash !== hash)
        changedValues.push(`  ${entry} ${name} ${previousHash} -> ${hash}`);
    }
  }
  console.log(`CHANGED DECLARATIONS ${changedDeclarations.length}`);
  for (const line of changedDeclarations) console.log(line);
  console.log(`CHANGED STRING VALUES ${changedValues.length}`);
  for (const line of changedValues) console.log(line);
}

function getHash(text) {
  return createHash('sha256').update(text).digest('hex').slice(0, 12);
}

function getEntryLocation(root, entry) {
  const [, packageName, ...rest] = entry.split('/');
  const packageDirectory = join(root, packageName);
  const manifestPath = join(packageDirectory, 'package.json');
  if (!existsSync(manifestPath)) return undefined;
  const exportMap = JSON.parse(readFileSync(manifestPath, 'utf8')).exports;
  const subpath = rest.length ? `./${rest.join('/')}` : '.';
  const target =
    typeof exportMap === 'string' ||
    (exportMap && !Object.keys(exportMap).some(key => key.startsWith('.')))
      ? subpath === '.'
        ? exportMap
        : undefined
      : exportMap?.[subpath];
  return target === undefined || target === null ? undefined : {packageDirectory, target};
}

// Same resolution as exports.mjs: explicit declarations, or declaration siblings of JS targets.
function resolveDeclarationPath(packageDirectory, target) {
  if (typeof target === 'string') {
    const candidates = /\.d\.(?:ts|mts|cts)$/.test(target)
      ? [target]
      : [
          target
            .replace(/\.mjs$/, '.d.mts')
            .replace(/\.cjs$/, '.d.cts')
            .replace(/\.js$/, '.d.ts'),
          target.replace(/\.(?:mjs|cjs)$/, '.d.ts')
        ];
    return candidates.find(
      candidate =>
        /\.d\.(?:ts|mts|cts)$/.test(candidate) && existsSync(join(packageDirectory, candidate))
    );
  }
  if (target && typeof target === 'object') {
    const candidates = Array.isArray(target)
      ? target
      : [target.types, target.import, target.default, ...Object.values(target)];
    for (const candidate of candidates) {
      const resolved = resolveDeclarationPath(packageDirectory, candidate);
      if (resolved) return resolved;
    }
  }
  return undefined;
}

function resolveRuntimePath(packageDirectory, target) {
  if (typeof target === 'string') {
    return /\.(?:m?js|cjs)$/.test(target) && existsSync(join(packageDirectory, target))
      ? target
      : undefined;
  }
  if (target && typeof target === 'object') {
    const candidates = Array.isArray(target)
      ? target
      : [target.import, target.module, target.default, target.node];
    for (const candidate of candidates) {
      const resolved = resolveRuntimePath(packageDirectory, candidate);
      if (resolved) return resolved;
    }
  }
  return undefined;
}

async function getFingerprints(root, entries) {
  const fingerprints = new Map();
  const declarationFiles = [];
  for (const entry of entries) {
    const location = getEntryLocation(root, entry);
    if (!location) continue;
    const declarationPath = resolveDeclarationPath(location.packageDirectory, location.target);
    fingerprints.set(entry, {declarations: new Map(), values: new Map(), location});
    if (declarationPath)
      declarationFiles.push([entry, join(location.packageDirectory, declarationPath)]);
  }
  const program = typescript.createProgram(
    declarationFiles.map(([, filePath]) => filePath),
    {
      skipLibCheck: true,
      noResolve: false,
      moduleResolution: typescript.ModuleResolutionKind.Bundler,
      types: []
    }
  );
  const checker = program.getTypeChecker();
  for (const [entry, filePath] of declarationFiles) {
    const source = program.getSourceFile(filePath);
    const symbol = source && checker.getSymbolAtLocation(source);
    if (!symbol) continue;
    const {declarations} = fingerprints.get(entry);
    for (const exported of checker.getExportsOfModule(symbol)) {
      let target = exported;
      if (target.flags & typescript.SymbolFlags.Alias) {
        try {
          target = checker.getAliasedSymbol(exported);
        } catch {
          // An unresolved alias keeps its own declaration text.
        }
      }
      const text = (target.declarations ?? [])
        .map(declaration => declaration.getText().replace(/\s+/g, ' '))
        .join('\n');
      declarations.set(exported.name, getHash(text));
    }
  }
  // One child process per root: luma.gl refuses to load two versions into one process.
  const runtimeTargets = [...fingerprints]
    .map(([entry, {location}]) => {
      const runtimePath = resolveRuntimePath(location.packageDirectory, location.target);
      return runtimePath && [entry, join(location.packageDirectory, runtimePath)];
    })
    .filter(Boolean);
  for (const [entry, result] of Object.entries(getRuntimeStringHashes(runtimeTargets))) {
    if (typeof result === 'string') {
      console.log('UNLOADED ENTRY', root, entry, result);
      continue;
    }
    const {values} = fingerprints.get(entry);
    for (const [name, hash] of Object.entries(result)) values.set(name, hash);
  }
  return fingerprints;
}

function getRuntimeStringHashes(runtimeTargets) {
  if (!runtimeTargets.length) return {};
  const result = spawnSync(process.execPath, ['--input-type=module', '-e', RUNTIME_SCRIPT], {
    input: JSON.stringify(runtimeTargets),
    encoding: 'utf8',
    maxBuffer: 256 * 1024 * 1024
  });
  const markerIndex = result.stdout?.lastIndexOf(RESULT_MARKER) ?? -1;
  if (markerIndex < 0)
    return Object.fromEntries(
      runtimeTargets.map(([entry]) => [
        entry,
        `runtime loader failed: ${result.stderr?.split('\n')[0]}`
      ])
    );
  return JSON.parse(result.stdout.slice(markerIndex + RESULT_MARKER.length));
}
