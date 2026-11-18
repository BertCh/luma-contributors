// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Maintained version of the 2026-11-11 public-export inventory; dated outputs stay in research.
import typescript from 'typescript';
import {existsSync, readFileSync, writeFileSync} from 'node:fs';
import {join, relative} from 'node:path';

const [packageRoot, outputPath] = process.argv.slice(2);
if (!packageRoot || !outputPath)
  throw new Error('Usage: exports.mjs <node_modules/@luma.gl> <out.json>');
const inventory = {};
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
for (const packageName of ['core', 'shaders', 'graph', 'engine']) {
  const packageDirectory = join(packageRoot, packageName);
  const manifest = JSON.parse(readFileSync(join(packageDirectory, 'package.json'), 'utf8'));
  const entries = [];
  const exportMap = manifest.exports;
  const publicEntries =
    typeof exportMap === 'string' ||
    (exportMap && !Object.keys(exportMap).some(key => key.startsWith('.')))
      ? [['.', exportMap]]
      : Object.entries(exportMap || {});
  for (const [subpath, target] of publicEntries) {
    if (subpath.includes('*') || subpath.split('/').includes('_internal')) continue;
    if (target === null) continue;
    const entryName = `@luma.gl/${packageName}${subpath === '.' ? '' : subpath.slice(1)}`;
    const declarationPath = resolveDeclarationPath(packageDirectory, target);
    if (!declarationPath) {
      inventory[entryName] = null;
      continue;
    }
    const filePath = join(packageDirectory, declarationPath);
    entries.push([entryName, filePath]);
  }
  const program = typescript.createProgram(
    entries.map(([, filePath]) => filePath),
    {
      skipLibCheck: true,
      noResolve: false,
      moduleResolution: typescript.ModuleResolutionKind.Bundler,
      types: []
    }
  );
  const checker = program.getTypeChecker();
  for (const [entryName, filePath] of entries) {
    const source = program.getSourceFile(filePath);
    const symbol = source && checker.getSymbolAtLocation(source);
    if (!symbol) {
      inventory[entryName] = null;
      continue;
    }
    const exports = {};
    for (const exported of checker.getExportsOfModule(symbol)) {
      let target = exported;
      if (target.flags & typescript.SymbolFlags.Alias) {
        try {
          target = checker.getAliasedSymbol(exported);
        } catch {
          // Keep the unresolved alias as a type, matching the historical inventory.
        }
      }
      const flags = target.flags;
      const kind =
        flags & typescript.SymbolFlags.Class
          ? 'class'
          : flags & typescript.SymbolFlags.Function
            ? 'function'
            : flags & typescript.SymbolFlags.Variable
              ? 'const'
              : flags & typescript.SymbolFlags.Enum
                ? 'enum'
                : 'type';
      const declaration = target.declarations?.[0];
      exports[exported.name] = {
        kind,
        file: declaration ? relative(packageDirectory, declaration.getSourceFile().fileName) : ''
      };
    }
    inventory[entryName] = exports;
  }
}
writeFileSync(outputPath, JSON.stringify(inventory));
for (const [entryName, exports] of Object.entries(inventory))
  console.log(entryName, exports ? Object.keys(exports).length : 'unresolved');
