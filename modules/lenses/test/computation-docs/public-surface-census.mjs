// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readdirSync, readFileSync} from 'node:fs';
import {dirname, resolve, relative} from 'node:path';
import ts from 'typescript';

export const repositoryRoot = resolve(import.meta.dirname, '../../../..');

function readSource(path) {
  return ts.createSourceFile(path, readFileSync(path, 'utf8'), ts.ScriptTarget.Latest, true);
}

/** Source-only census: includes type exports without loading scenarios, data or GPU modules. */
export function collectExportNames(path, visited = new Set(), overrides = {}) {
  path = resolve(path);
  if (visited.has(path)) return [];
  visited.add(path);
  const names = new Set();
  const parsed = overrides[path] === undefined ? readSource(path) : ts.createSourceFile(path, overrides[path], ts.ScriptTarget.Latest, true);
  for (const statement of parsed.statements) {
    if (ts.isExportDeclaration(statement)) {
      if (statement.exportClause && ts.isNamedExports(statement.exportClause)) {
        for (const element of statement.exportClause.elements) names.add(element.name.text);
      } else if (statement.exportClause && ts.isNamespaceExport(statement.exportClause)) {
        names.add(statement.exportClause.name.text);
      } else if (statement.moduleSpecifier && ts.isStringLiteral(statement.moduleSpecifier)) {
        const target = resolve(dirname(path), statement.moduleSpecifier.text);
        const source = [target + '.ts', target + '/index.ts'].find(existsSync);
        if (!source) throw new Error(`Cannot resolve ${target}`);
        for (const name of collectExportNames(source, visited, overrides)) if (name !== 'default') names.add(name);
      }
    } else if (statement.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword)) {
      if (ts.isVariableStatement(statement)) {
        for (const declaration of statement.declarationList.declarations) {
          if (!ts.isIdentifier(declaration.name)) throw new Error(`Export binding pattern: ${path}`);
          names.add(declaration.name.text);
        }
      } else if (statement.name) names.add(statement.name.text);
      else if (statement.modifiers.some(modifier => modifier.kind === ts.SyntaxKind.DefaultKeyword)) names.add('default');
    }
  }
  return [...names].sort();
}

export function createPublicSurfaceSnapshot(manifest, overrides = {}) {
  manifest ??= JSON.parse(readFileSync(resolve(repositoryRoot, 'modules/lenses/package.json'), 'utf8'));
  return {exports: manifest.exports, entries: Object.fromEntries(Object.entries(listPublicEntries(manifest)).map(([entry, source]) => [entry, collectExportKinds(source, overrides)]))};
}

function collectExportKinds(path, overrides) {
  const text = overrides[path] ?? readFileSync(path, 'utf8');
  const source = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true);
  const values = new Set();
  const types = new Set();
  for (const statement of source.statements) {
    if (ts.isExportDeclaration(statement) && statement.exportClause && ts.isNamedExports(statement.exportClause)) {
      for (const element of statement.exportClause.elements) (statement.isTypeOnly || element.isTypeOnly ? types : values).add(element.name.text);
    } else if (statement.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword)) {
      if (ts.isVariableStatement(statement)) for (const declaration of statement.declarationList.declarations) values.add(declaration.name.text);
      else if (statement.name) (ts.isTypeAliasDeclaration(statement) || ts.isInterfaceDeclaration(statement) ? types : values).add(statement.name.text);
    }
  }
  return {values: [...values].sort(), types: [...types].sort()};
}

function listFiles(directory) {
  return readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    if (['node_modules', 'dist', '.git', '.typecheck', 'build'].includes(entry.name)) return [];
    const path = resolve(directory, entry.name);
    return entry.isDirectory() ? listFiles(path) : [path];
  });
}

/** Expand both export-map patterns using real source files; data assets are inventoried separately. */
export function listPublicEntries(manifest = JSON.parse(readFileSync(resolve(repositoryRoot, 'modules/lenses/package.json'), 'utf8'))) {
  const sourceRoot = resolve(repositoryRoot, 'modules/lenses/src');
  const entries = {};
  for (const [key, conditions] of Object.entries(manifest.exports)) {
    const target = typeof conditions === 'string' ? conditions : conditions.types ?? conditions.import;
    const source = target.replace('./dist/', '').replace(/\.d\.ts$|\.[cm]?js$/, '.ts');
    if (key.includes('*')) {
      for (const path of listFiles(sourceRoot)) {
        const name = relative(sourceRoot, path);
        const [prefix, suffix] = source.split('*');
        if (name.endsWith('.ts') && name.startsWith(prefix) && name.endsWith(suffix)) {
          entries[key.replace('*', name.slice(prefix.length, suffix ? -suffix.length : undefined))] = path;
        }
      }
    } else if (source.endsWith('.ts')) entries[key] = resolve(sourceRoot, source);
  }
  return entries;
}

export function createPublicSurfaceCensus() {
  const consumers = ['modules/kepler', 'website', 'examples', 'docs'].flatMap(directory =>
    listFiles(resolve(repositoryRoot, directory)).filter(path => /\.(?:[cm]?[jt]sx?|md|mdx)$/.test(path))
  ).map(path => ({path: relative(repositoryRoot, path), text: readFileSync(path, 'utf8')}));
  const documentation = [readFileSync(resolve(repositoryRoot, 'modules/lenses/README.md'), 'utf8'),
    ...consumers.filter(file => file.path.startsWith('docs/')).map(file => file.text)].join('\n');
  const propertyReferences = new Set(consumers.flatMap(file => [...file.text.matchAll(/\b(?:views|lenses)\s*\.\s*([A-Za-z_$][\w$]*)/g)].map(match => match[1])));
  const surface = Object.fromEntries(Object.entries(listPublicEntries()).map(([entry, source]) => {
    const names = collectExportNames(source);
    const specifier = '@luma-contributors/lenses' + (entry === '.' ? '' : entry.slice(1));
    const importingFiles = consumers.filter(file => file.text.includes(specifier + "'") || file.text.includes(specifier + '"'));
    // A dynamic module can escape its loading file (the showcase returns {views, lenses}).
    // Follow named property accesses throughout the consumer roots as well.
    // Conservative for namespace/dynamic imports and Markdown snippets: a referenced exported
    // identifier in a file importing the entry counts as used, including string-key accesses.
    const usesName = (text, name) => new RegExp(`\\b${name}\\b`).test(text);
    return [entry, {source: relative(repositoryRoot, source), names,
      consumerNames: names.filter(name => importingFiles.some(file => usesName(file.text, name)) ||
        (importingFiles.some(file => file.text.includes(`import('${specifier}')`)) &&
          propertyReferences.has(name))),
      consumers: importingFiles.map(file => file.path),
      documentedNames: names.filter(name => usesName(documentation, name))}];
  }));
  const manifest = JSON.parse(readFileSync(resolve(repositoryRoot, 'modules/lenses/package.json'), 'utf8'));
  for (const [entry, target] of Object.entries(manifest.exports)) {
    if (typeof target !== 'string' || !target.includes('*')) continue;
    const [prefix, suffix] = target.replace('./dist/', 'modules/lenses/src/').split('*');
    surface[entry] = {source: target, names: [], consumerNames: [], consumers: [], documentedNames: [],
      assets: listFiles(resolve(repositoryRoot, 'modules/lenses/src')).map(path => relative(repositoryRoot, path))
        .filter(path => path.startsWith(prefix) && path.endsWith(suffix)).sort()};
  }
  return surface;
}
