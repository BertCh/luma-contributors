// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fs from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import type {Plugin} from 'vite';
import {DATA_FILES, DATA_SOURCES, MODEL_INPUTS, SCENARIOS} from './demo-data-sources.ts';

type EntryData = {sources: string[]; scenario?: string; caption?: string};
type BuildOptions = {watchFile?: (file: string) => void; onMissing?: (message: string) => void};

const RENDER_FILES = new Set(['demos/kit.ts', 'runs/define-run.ts', 'runs/layers.ts']);
const INVENTORY_ID = 'virtual:demo-data';

/** Follow value imports, including `?raw` and `?url&inline` assets that have no TS value symbol. */
function getFixturePath(declaration: ts.Declaration, dataDirectory: string): string | undefined {
  if (!ts.isImportClause(declaration) && !ts.isImportSpecifier(declaration)) return undefined;
  const statement = ts.findAncestor(declaration, ts.isImportDeclaration);
  if (!statement || !ts.isStringLiteral(statement.moduleSpecifier)) return undefined;
  const specifier = statement.moduleSpecifier.text.split('?')[0];
  if (!specifier.startsWith('.') || !/\.(json|bin|b64)$/.test(specifier)) return undefined;
  const absolute = path.resolve(path.dirname(declaration.getSourceFile().fileName), specifier);
  if (absolute.startsWith(dataDirectory + path.sep)) {
    return path.relative(dataDirectory, absolute).replaceAll(path.sep, '/');
  }
  return undefined;
}

/**
 * A file-level dependency graph assigns every dataset in runs/inputs.ts to every run importing
 * that module. Follow the value declarations actually referenced from each scene's default export.
 * This includes build/draw inputs and contextual layers such as the Japanese coastline.
 */
export function buildDemoData(gpuDirectory: string, options: BuildOptions = {}) {
  const files = fs
    .readdirSync(gpuDirectory, {recursive: true, encoding: 'utf8'})
    .filter(
      file =>
        /^(runs|demos|data)[/\\]/.test(file) &&
        file.endsWith('.ts') &&
        !file.endsWith('.parameters.ts') &&
        !file.endsWith('.d.ts')
    )
    .map(file => path.join(gpuDirectory, file));
  const program = ts.createProgram(files, {
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    types: [],
    noLib: true,
    noResolve: true
  });
  const checker = program.getTypeChecker();
  const relativeFiles = new Map(
    files.map(file => [file, path.relative(gpuDirectory, file).replaceAll(path.sep, '/')])
  );
  const dataDirectory = path.join(gpuDirectory, 'data');
  const entries: Record<string, EntryData> = {};
  const unregistered = new Set<string>();
  // References to a local variable or an object property do not lead out of the expression
  // being walked. Only file-scope bindings can lead to another helper or an asset import.
  const fileBindings = new Map<ts.SourceFile, ReadonlySet<string>>();
  function isFileBinding(node: ts.Identifier): boolean {
    const file = node.getSourceFile();
    let bindings = fileBindings.get(file);
    if (!bindings) {
      const symbols = checker.getSymbolsInScope(file, ts.SymbolFlags.Value | ts.SymbolFlags.Alias);
      bindings = new Set(symbols.map(symbol => symbol.name));
      fileBindings.set(file, bindings);
    }
    return bindings.has(node.text);
  }

  function findSources(root: ts.Expression, id: string): EntryData {
    const sources = new Set<string>();
    const notes = new Set<string>();
    const visited = new Set<ts.Node>();
    function visit(node: ts.Node) {
      if (visited.has(node) || ts.isTypeNode(node)) return;
      visited.add(node);
      if (!ts.isIdentifier(node)) {
        ts.forEachChild(node, visit);
        return;
      }
      if (!isFileBinding(node)) return;
      if (ts.isPropertyAccessExpression(node.parent) && node.parent.name === node) return;
      if (ts.isPropertyAssignment(node.parent) && node.parent.name === node) return;
      const symbol = ts.isShorthandPropertyAssignment(node.parent)
        ? checker.getShorthandAssignmentValueSymbol(node.parent)
        : checker.getSymbolAtLocation(node);
      if (!symbol) return;
      const target =
        symbol.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(symbol) : symbol;
      for (const candidate of symbol.declarations ?? []) {
        const fixture = getFixturePath(candidate, dataDirectory);
        if (!fixture) continue;
        if (DATA_FILES[fixture]) {
          for (const source of DATA_FILES[fixture]) sources.add(source);
        } else {
          unregistered.add(fixture);
          notes.add('Source information pending.');
        }
      }
      const declaration = target.valueDeclaration;
      if (!declaration || visited.has(declaration)) return;
      const file = relativeFiles.get(declaration.getSourceFile().fileName);
      if (!file || RENDER_FILES.has(file)) return;
      const note = MODEL_INPUTS[`${file}#${target.name}`];
      if (note) notes.add(note);
      visit(declaration);
    }
    visit(root);
    const scenario =
      SCENARIOS[id] ??
      ([...notes].join(' ') ||
        (sources.size ? undefined : 'Constructed input for this operation.'));
    const argument = ts.isCallExpression(root) ? root.arguments[0] : undefined;
    const property =
      argument && ts.isObjectLiteralExpression(argument)
        ? argument.properties.find(
            member => ts.isPropertyAssignment(member) && member.name.getText() === 'caption'
          )
        : undefined;
    const caption =
      property && ts.isPropertyAssignment(property) && ts.isStringLiteralLike(property.initializer)
        ? property.initializer.text
        : undefined;
    return {sources: [...sources].sort(), scenario, caption};
  }

  for (const [file, relative] of relativeFiles) {
    if (!RENDER_FILES.has(relative)) options.watchFile?.(file);
    const run = /^runs\/entries\/[^/]+\/([^/]+)\.ts$/.exec(relative);
    const demo = /^demos\/([^/]+)\.ts$/.exec(relative);
    const id = run?.[1] ?? demo?.[1];
    if (!id) continue;
    const scene = program.getSourceFile(file)?.statements.find(ts.isExportAssignment);
    if (scene) entries[`${run ? 'runs' : 'demos'}/${id}`] = findSources(scene.expression, id);
  }
  if (unregistered.size) {
    const message = `Unregistered demo datasets: ${[...unregistered].sort().join(', ')}`;
    if (!options.onMissing) throw new Error(message);
    options.onMissing(message);
  }
  return {sources: DATA_SOURCES, entries};
}

export function demoDataPlugin(gpuDirectory: string): Plugin {
  let build = false;
  return {
    name: 'luma-demo-data',
    configResolved(config) {
      build = config.command === 'build';
    },
    resolveId(id) {
      return id === INVENTORY_ID ? `\0${id}` : null;
    },
    load(id) {
      if (id !== `\0${INVENTORY_ID}`) return null;
      const inventory = buildDemoData(gpuDirectory, {
        watchFile: file => this.addWatchFile(file),
        onMissing: build ? undefined : message => this.warn(message)
      });
      return `export default ${JSON.stringify(inventory)};`;
    }
  };
}
