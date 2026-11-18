// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, relative, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {describe, expect, it} from 'vitest';
import {ANALYSIS_DOC_PAGES} from './doc-program';
import {DOCUMENTED_PAGE_HEADINGS} from './documented-page-headings';

// Rule 20 (A-4, A-13): shared contracts live below every analysis entry, the entries form a DAG,
// and each entry barrel is the union of its subdirectory indexes.

const SOURCE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../src');
const DOCS_ROOT = resolve(SOURCE_ROOT, '../../../docs/api');

/** Shared layers that every analysis entry may import. They import no analysis entry. */
const SHARED_LAYERS = ['analysis-core'] as const;

/**
 * Analysis entries from lowest to highest. An entry may import shared layers and entries listed
 * before it, never one listed after it.
 */
const ENTRY_ORDER = [
  'geospatial-analysis',
  'dataframe-analysis',
  'raster-analysis',
  'pointcloud-analysis',
  'gpu-terrain',
  'gpu-spatial-analysis',
  'gpu-network'
] as const;

/** Sibling source layers the analysis entries may import without being an entry themselves. */
const UPSTREAM_ENTRIES = new Set(['computation-record']);

/** Top-level source files each barrel re-exports explicitly; everything else comes from `export *`. */
const EXPLICIT_BARREL_SOURCES: Record<(typeof ENTRY_ORDER)[number], RegExp> = Object.fromEntries(
  ENTRY_ORDER.map(entry => [entry, /^\.\/[a-z0-9-]+$|^\.\.\/analysis-core\//])
) as Record<(typeof ENTRY_ORDER)[number], RegExp>;

/** Family pages come from the maintained documentation inventory, including their subpages. */
const DOCUMENTED_ENTRIES = ANALYSIS_DOC_PAGES.map(page => {
  const family = page.split('/')[0];
  const pageEntries: Record<string, string> = {
    'gpu-raster': 'raster-analysis',
    'gpu-dataframe-analysis.md': 'dataframe-analysis',
    'pointcloud-analysis.md': 'pointcloud-analysis',
    'geospatial.md': 'geospatial-analysis',
    'forest-inference.md': 'dataframe-analysis',
    'computation-record.md': 'computation-record',
    'operation-forms-and-cpu-twins.md': 'analysis-core',
    'variable-standardization.md': 'dataframe-analysis',
    'gpu-geometry-refinement.md': 'gpu-spatial-analysis',
    'gpu-spatial-analysis-cross-reference.md': 'gpu-spatial-analysis'
  };
  const markdown = readFileSync(join(DOCS_ROOT, page), 'utf8');
  const entry =
    ENTRY_ORDER.find(candidate => candidate === family) ??
    pageEntries[family] ??
    /@luma-contributors\/analysis\/([a-z-]+)/.exec(markdown)?.[1];
  const requiredHeadings = DOCUMENTED_PAGE_HEADINGS[page];
  if (!requiredHeadings) throw new Error(`missing heading requirements for ${page}`);
  if (!entry && requiredHeadings.length) throw new Error(`unclassified contributor page ${page}`);
  return {page, entry, requiredHeadings};
});

function getMissingHeadings(markdown: string, requiredHeadings: readonly string[]): string[] {
  const documented = getDocumentedNames(markdown);
  return requiredHeadings.filter(name => !documented.has(name));
}

function getDocumentedNames(markdown: string): Set<string> {
  const documented = new Set<string>();
  for (const heading of markdown.match(/^#{2,4} .*$/gm) ?? []) {
    for (const match of heading.matchAll(/`((?:GPU[A-Z]\w+)|(?:add\w+Recipe))`/g)) {
      documented.add(match[1]);
    }
  }
  return documented;
}

function getUnreachableNames(markdown: string, entryExports: ReadonlySet<string>): string[] {
  return [...getDocumentedNames(markdown)].filter(name => !entryExports.has(name));
}

it('detects a documented contributor missing from its entry', () => {
  const markdown = '### `GPUExample`\n\n#### `addExampleRecipe`';
  expect(getUnreachableNames(markdown, new Set(['GPUExample']))).toEqual(['addExampleRecipe']);
  expect(getUnreachableNames(markdown, new Set(['GPUExample', 'addExampleRecipe']))).toEqual([]);
});

it('maps the computation-record contributor page to its runtime barrel', () => {
  expect(DOCUMENTED_ENTRIES.find(({page}) => page === 'computation-record.md')?.entry).toBe(
    'computation-record'
  );
});

it('detects a contributor page stripped of its headings', () => {
  const markdown = readFileSync(
    join(DOCS_ROOT, 'gpu-spatial-analysis/sinkhorn-divergence.md'),
    'utf8'
  );
  const stripped = markdown.replace(/^#{2,4} .*$/gm, '');
  const {requiredHeadings} = DOCUMENTED_ENTRIES.find(
    ({page}) => page === 'gpu-spatial-analysis/sinkhorn-divergence.md'
  )!;
  expect(requiredHeadings).toContain('GPUSinkhornDivergence');
  expect(getMissingHeadings(markdown, requiredHeadings)).toEqual([]);
  expect(getMissingHeadings(stripped, requiredHeadings)).toContain('GPUSinkhornDivergence');
});

it.each([
  ['export class GPUExample {}', true],
  ['export type GPUExample = {x: number};', false],
  ['class GPUExample {} export type {GPUExample};', false]
])('requires a runtime contributor value: %s', (text, expected) => {
  const file = '/virtual/runtime-entry.ts';
  const options = {
    noLib: true,
    target: typescript.ScriptTarget.ES2022,
    module: typescript.ModuleKind.ESNext
  };
  const host = typescript.createCompilerHost(options);
  host.getSourceFile = name =>
    name === file ? typescript.createSourceFile(file, text, options.target, true) : undefined;
  const program = typescript.createProgram([file], options, host);
  const checker = program.getTypeChecker();
  const moduleSymbol = checker.getSymbolAtLocation(program.getSourceFile(file)!)!;
  const names = getRuntimeExportNames(moduleSymbol, checker);
  expect(names.has('GPUExample')).toBe(expected);
});

/** Follow runtime export syntax as well as value symbols: `export type` never emits a value. */
function getRuntimeExportNames(
  moduleSymbol: typescript.Symbol,
  checker: typescript.TypeChecker,
  visited = new Set<typescript.Symbol>()
): Set<string> {
  if (visited.has(moduleSymbol)) return new Set();
  const nextVisited = new Set(visited).add(moduleSymbol);
  const source = moduleSymbol.declarations?.find(typescript.isSourceFile);
  if (!source) return new Set();
  const names = new Set<string>();
  const isRuntimeSymbol = (symbol: typescript.Symbol | undefined): boolean => {
    if (!symbol) return false;
    if (symbol.flags & typescript.SymbolFlags.Alias) {
      for (const declaration of symbol.declarations ?? []) {
        if (typescript.isImportSpecifier(declaration)) {
          const clause = declaration.parent.parent;
          if (declaration.isTypeOnly || clause.isTypeOnly) return false;
          const imported = checker.getSymbolAtLocation(clause.parent.moduleSpecifier);
          return Boolean(
            imported &&
              getRuntimeExportNames(imported, checker, nextVisited).has(
                (declaration.propertyName ?? declaration.name).text
              )
          );
        }
      }
      return Boolean(checker.getAliasedSymbol(symbol).flags & typescript.SymbolFlags.Value);
    }
    return Boolean(symbol.flags & typescript.SymbolFlags.Value);
  };
  for (const statement of source.statements) {
    if (typescript.isExportDeclaration(statement)) {
      if (statement.isTypeOnly) continue;
      const target =
        statement.moduleSpecifier && checker.getSymbolAtLocation(statement.moduleSpecifier);
      const targetNames = target && getRuntimeExportNames(target, checker, nextVisited);
      if (!statement.exportClause) {
        for (const name of targetNames ?? []) if (name !== 'default') names.add(name);
      } else if (typescript.isNamedExports(statement.exportClause)) {
        for (const element of statement.exportClause.elements) {
          if (element.isTypeOnly) continue;
          if (
            statement.moduleSpecifier
              ? targetNames?.has((element.propertyName ?? element.name).text)
              : isRuntimeSymbol(checker.getExportSpecifierLocalTargetSymbol(element))
          )
            names.add(element.name.text);
        }
      } else names.add(statement.exportClause.name.text);
      continue;
    }
    if (
      !typescript.canHaveModifiers(statement) ||
      !typescript
        .getModifiers(statement)
        ?.some(modifier => modifier.kind === typescript.SyntaxKind.ExportKeyword)
    )
      continue;
    if (typescript.isVariableStatement(statement)) {
      for (const declaration of statement.declarationList.declarations) {
        if (
          typescript.isIdentifier(declaration.name) &&
          isRuntimeSymbol(checker.getSymbolAtLocation(declaration.name))
        )
          names.add(declaration.name.text);
      }
    } else if (
      (typescript.isClassDeclaration(statement) ||
        typescript.isFunctionDeclaration(statement) ||
        typescript.isEnumDeclaration(statement)) &&
      statement.name
    ) {
      if (isRuntimeSymbol(checker.getSymbolAtLocation(statement.name)))
        names.add(statement.name.text);
    }
  }
  return names;
}

it.each([
  ['export * from "./definitions";', true],
  ['export type * from "./definitions";', false],
  ['export type {GPUExample} from "./definitions";', false],
  ['export {type GPUExample} from "./definitions";', false]
])('follows runtime exports through a virtual barrel: %s', (middle, expected) => {
  const files = new Map([
    ['/virtual/entry.ts', 'export * from "./middle";'],
    ['/virtual/middle.ts', middle],
    ['/virtual/definitions.ts', 'export class GPUExample {}']
  ]);
  const options = {
    noLib: true,
    target: typescript.ScriptTarget.ES2022,
    module: typescript.ModuleKind.ESNext,
    moduleResolution: typescript.ModuleResolutionKind.Bundler
  };
  const host = typescript.createCompilerHost(options);
  host.fileExists = name => files.has(name);
  host.directoryExists = name => name === '/virtual';
  host.readFile = name => files.get(name);
  host.getSourceFile = name =>
    files.has(name)
      ? typescript.createSourceFile(name, files.get(name)!, options.target, true)
      : undefined;
  const program = typescript.createProgram(['/virtual/entry.ts'], options, host);
  const checker = program.getTypeChecker();
  const moduleSymbol = checker.getSymbolAtLocation(program.getSourceFile('/virtual/entry.ts')!)!;
  expect(getRuntimeExportNames(moduleSymbol, checker).has('GPUExample')).toBe(expected);
});

function listSourceFiles(directory: string): string[] {
  const files: string[] = [];
  for (const name of readdirSync(directory)) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) {
      files.push(...listSourceFiles(path));
    } else if (name.endsWith('.ts')) {
      files.push(path);
    }
  }
  return files;
}

function getRelativeImportLayers(file: string): Set<string> {
  const layers = new Set<string>();
  const source = readFileSync(file, 'utf8');
  for (const match of source.matchAll(/(?:from|import)\s*\(?\s*'(\.{1,2}\/[^']*)'/g)) {
    const target = relative(SOURCE_ROOT, resolve(dirname(file), match[1]));
    const layer = target.split(sep)[0];
    if (!layer.startsWith('..')) {
      layers.add(layer.replace(/\.ts$/, ''));
    }
  }
  return layers;
}

function getLayerImports(layer: string): Map<string, string[]> {
  const imports = new Map<string, string[]>();
  for (const file of listSourceFiles(join(SOURCE_ROOT, layer))) {
    for (const target of getRelativeImportLayers(file)) {
      if (target !== layer) {
        imports.set(target, [...(imports.get(target) || []), relative(SOURCE_ROOT, file)]);
      }
    }
  }
  return imports;
}

type EntryExportSnapshot = {
  exportsByFile: Map<string, Set<string>>;
  targetsByFile: Map<string, Map<string, typescript.Symbol>>;
};

/** Compare meanings and runtime availability, regardless of the re-export path's spelling. */
function getIncompleteFamilyExports(
  entryTargets: ReadonlyMap<string, typescript.Symbol>,
  entryRuntime: ReadonlySet<string>,
  familyTargets: ReadonlyMap<string, typescript.Symbol>,
  familyRuntime: ReadonlySet<string>
): string[] {
  return [...familyTargets]
    .filter(
      ([name, target]) =>
        entryTargets.get(name) !== target || (familyRuntime.has(name) && !entryRuntime.has(name))
    )
    .map(([name]) => name);
}

function collectEntryExports(
  program: typescript.Program,
  files: readonly string[],
  root: string
): EntryExportSnapshot {
  const checker = program.getTypeChecker();
  const exportsByFile = new Map<string, Set<string>>();
  const targetsByFile = new Map<string, Map<string, typescript.Symbol>>();
  for (const file of files) {
    const sourceFile = program.getSourceFile(file);
    const moduleSymbol = sourceFile && checker.getSymbolAtLocation(sourceFile);
    const exports = moduleSymbol ? checker.getExportsOfModule(moduleSymbol) : [];
    const key = relative(root, file);
    exportsByFile.set(key, new Set(exports.map(({name}) => name)));
    exportsByFile.set(
      `runtime:${key}`,
      moduleSymbol ? getRuntimeExportNames(moduleSymbol, checker) : new Set()
    );
    targetsByFile.set(
      key,
      new Map(
        exports.map(symbol => [
          symbol.name,
          symbol.flags & typescript.SymbolFlags.Alias ? checker.getAliasedSymbol(symbol) : symbol
        ])
      )
    );
  }
  return {exportsByFile, targetsByFile};
}

it.each([
  ['direct star', 'export * from "./family/index";', '', []],
  [
    'complete named compatibility bridge',
    'export * from "./bridge";',
    'export {GPUFirst, GPUSecond} from "./family/index"; export type {GPUFirstProps} from "./family/index";',
    []
  ],
  [
    'partial compatibility bridge',
    'export * from "./bridge";',
    'export {GPUFirst} from "./family/index"; export type {GPUFirstProps} from "./family/index";',
    ['GPUSecond']
  ],
  [
    'same name with a different constructor',
    'export * from "./bridge";',
    'export class GPUFirst {} export {GPUSecond} from "./family/index"; export type {GPUFirstProps} from "./family/index";',
    ['GPUFirst']
  ],
  [
    'constructor erased by a type-only bridge',
    'export * from "./bridge";',
    'export {GPUSecond} from "./family/index"; export type {GPUFirst, GPUFirstProps} from "./family/index";',
    ['GPUFirst']
  ]
] as const)('checks whole-family exports through %s', (_name, entry, bridge, expected) => {
  const root = '/virtual';
  const files = new Map([
    ['/virtual/index.ts', entry],
    ['/virtual/bridge.ts', bridge],
    [
      '/virtual/family/index.ts',
      'export class GPUFirst {} export class GPUSecond {} export type GPUFirstProps = {value: number};'
    ]
  ]);
  const options = {
    noLib: true,
    target: typescript.ScriptTarget.ES2022,
    module: typescript.ModuleKind.ESNext,
    moduleResolution: typescript.ModuleResolutionKind.Bundler
  };
  const host = typescript.createCompilerHost(options);
  host.fileExists = name => files.has(name);
  host.directoryExists = name => name === root || name === `${root}/family`;
  host.readFile = name => files.get(name);
  host.getSourceFile = name =>
    files.has(name)
      ? typescript.createSourceFile(name, files.get(name)!, options.target, true)
      : undefined;
  const program = typescript.createProgram([...files.keys()], options, host);
  const {exportsByFile, targetsByFile} = collectEntryExports(program, [...files.keys()], root);
  expect(
    getIncompleteFamilyExports(
      targetsByFile.get('index.ts')!,
      exportsByFile.get('runtime:index.ts')!,
      targetsByFile.get('family/index.ts')!,
      exportsByFile.get('runtime:family/index.ts')!
    )
  ).toEqual(expected);
});

function getEntryExports(): EntryExportSnapshot {
  const files: string[] = [];
  for (const entry of [...ENTRY_ORDER, ...SHARED_LAYERS, ...UPSTREAM_ENTRIES]) {
    files.push(join(SOURCE_ROOT, entry, 'index.ts'));
    for (const index of getSubdirectoryIndexes(entry)) {
      files.push(index);
    }
  }
  const program = typescript.createProgram(files, {
    noEmit: true,
    skipLibCheck: true,
    target: typescript.ScriptTarget.ES2022,
    module: typescript.ModuleKind.ESNext,
    moduleResolution: typescript.ModuleResolutionKind.Bundler
  });
  return collectEntryExports(program, files, SOURCE_ROOT);
}

function getSubdirectoryIndexes(entry: string): string[] {
  return readdirSync(join(SOURCE_ROOT, entry))
    .map(name => join(SOURCE_ROOT, entry, name, 'index.ts'))
    .filter(index => {
      try {
        return statSync(index).isFile();
      } catch {
        return false;
      }
    });
}

function expectFamilyFolderStructure(entryDirectory: string): void {
  const looseFiles = readdirSync(entryDirectory).filter(
    name => name.endsWith('.ts') && name !== 'index.ts'
  );
  expect(looseFiles).toEqual([]);
  const families = readdirSync(entryDirectory).filter(name =>
    statSync(join(entryDirectory, name)).isDirectory()
  );
  for (const family of families) {
    const index = join(entryDirectory, family, 'index.ts');
    expect(existsSync(index), `${family}/index.ts must exist`).toBe(true);
    expect(statSync(index).isFile(), `${family}/index.ts must be a file`).toBe(true);
    expect(readFileSync(join(entryDirectory, 'index.ts'), 'utf8')).toContain(
      `export * from './${family}/index';`
    );
  }
}

it('rejects a missing family index even when the root directly exports a resolvable leaf', () => {
  const entryDirectory = mkdtempSync(join(tmpdir(), 'root-files-missing-index-'));
  try {
    const familyDirectory = join(entryDirectory, 'viewport-bounds');
    mkdirSync(familyDirectory);
    const entryFile = join(entryDirectory, 'index.ts');
    writeFileSync(entryFile, "export * from './viewport-bounds/geographic-viewport-bounds';\n");
    writeFileSync(
      join(familyDirectory, 'geographic-viewport-bounds.ts'),
      'export function getGeographicViewportBounds() { return []; }\n'
    );
    const program = typescript.createProgram([entryFile], {
      noLib: true,
      module: typescript.ModuleKind.ESNext,
      moduleResolution: typescript.ModuleResolutionKind.Bundler
    });
    const checker = program.getTypeChecker();
    const moduleSymbol = checker.getSymbolAtLocation(program.getSourceFile(entryFile)!)!;
    expect(checker.getExportsOfModule(moduleSymbol).map(symbol => symbol.name)).toContain(
      'getGeographicViewportBounds'
    );
    expect(() => expectFamilyFolderStructure(entryDirectory)).toThrow('viewport-bounds/index.ts');
  } finally {
    rmSync(entryDirectory, {recursive: true, force: true});
  }
});

it('keeps the root-file gate guide aligned with the completed family migrations', () => {
  const guide = readFileSync(resolve(SOURCE_ROOT, '../../../dev-docs/script-gates.md'), 'utf8');
  const baseline = JSON.parse(
    readFileSync(
      resolve(SOURCE_ROOT, '../../../scripts/contributor-contract-baseline.json'),
      'utf8'
    )
  ) as {'root-file': Record<string, unknown>};
  for (const entry of ['pointcloud-analysis', 'geospatial-analysis']) {
    expect(readdirSync(join(SOURCE_ROOT, entry)).filter(name => name.endsWith('.ts'))).toEqual([
      'index.ts'
    ]);
    expect(
      Object.keys(baseline['root-file']).filter(path =>
        path.startsWith(`modules/analysis/src/${entry}/`)
      )
    ).toEqual([]);
  }
  expect(guide).not.toContain('Loose pointcloud/geospatial files remain recorded findings');
  for (const family of [
    'pointcloud-analysis/operations',
    'geospatial-analysis/datum-grid-transform',
    'geospatial-analysis/point-buffer',
    'geospatial-analysis/viewport-bounds'
  ]) {
    expect(statSync(join(SOURCE_ROOT, family)).isDirectory()).toBe(true);
    expect(guide, family).toContain(family);
  }
});

describe('analysis layering', () => {
  it('never imports @luma-contributors/lenses', () => {
    for (const file of listSourceFiles(SOURCE_ROOT)) {
      expect(readFileSync(file, 'utf8'), relative(SOURCE_ROOT, file)).not.toContain(
        '@luma-contributors/lenses'
      );
    }
  });

  it('keeps the shared layers below every analysis entry', () => {
    for (const layer of SHARED_LAYERS) {
      const imports = getLayerImports(layer);
      const allowed = [
        'computation-record',
        ...SHARED_LAYERS.slice(0, SHARED_LAYERS.indexOf(layer) + 1)
      ];
      for (const [target, files] of imports) {
        expect(allowed as readonly string[], `${layer} imports ${target} in ${files[0]}`).toContain(
          target
        );
      }
    }
  });

  it('orders the analysis entries as a DAG', () => {
    for (const [position, entry] of ENTRY_ORDER.entries()) {
      const allowed = new Set<string>([
        ...SHARED_LAYERS,
        ...UPSTREAM_ENTRIES,
        ...ENTRY_ORDER.slice(0, position)
      ]);
      for (const [target, files] of getLayerImports(entry)) {
        expect(allowed.has(target), `${entry} imports ${target} in ${files[0]}`).toBe(true);
      }
    }
  });
});

describe('analysis entry barrels', () => {
  const {exportsByFile, targetsByFile} = getEntryExports();

  it.each([
    'pointcloud-analysis',
    'geospatial-analysis'
  ])('keeps %s implementations in family folders', entry => {
    expectFamilyFolderStructure(join(SOURCE_ROOT, entry));
  });

  it('re-export every subdirectory index whole', () => {
    let checked = 0;
    for (const entry of ENTRY_ORDER) {
      const barrel = readFileSync(join(SOURCE_ROOT, entry, 'index.ts'), 'utf8');
      const entryFile = join(entry, 'index.ts');
      for (const index of getSubdirectoryIndexes(entry)) {
        checked++;
        const familyFile = relative(SOURCE_ROOT, index);
        expect(
          getIncompleteFamilyExports(
            targetsByFile.get(entryFile)!,
            exportsByFile.get(`runtime:${entryFile}`)!,
            targetsByFile.get(familyFile)!,
            exportsByFile.get(`runtime:${familyFile}`)!
          ),
          `${entry} must preserve the whole ${familyFile} public surface`
        ).toEqual([]);
      }
      for (const match of barrel.matchAll(/export\s+(?:type\s+)?\{[^}]*\}\s+from\s+'([^']*)'/g)) {
        expect(match[1], `${entry} explicit re-export`).toMatch(EXPLICIT_BARREL_SOURCES[entry]);
      }
    }
    expect(checked, 'subdirectory indexes checked').toBeGreaterThan(0);
  });

  it('exports the union of the subdirectory exports', () => {
    for (const entry of ENTRY_ORDER) {
      const entryExports = exportsByFile.get(join(entry, 'index.ts'))!;
      for (const index of getSubdirectoryIndexes(entry)) {
        const subdirectoryExports = exportsByFile.get(relative(SOURCE_ROOT, index))!;
        const missing = [...subdirectoryExports].filter(name => !entryExports.has(name));
        expect(missing, relative(SOURCE_ROOT, index)).toEqual([]);
      }
    }
  });

  it('reaches every documented contributor and recipe from its entry', () => {
    expect(DOCUMENTED_ENTRIES.length, 'documented entries').toBeGreaterThan(0);
    let documentedCount = 0;
    expect(DOCUMENTED_ENTRIES.length).toBe(ANALYSIS_DOC_PAGES.length);
    expect(Object.keys(DOCUMENTED_PAGE_HEADINGS).sort()).toEqual([...ANALYSIS_DOC_PAGES].sort());
    for (const {page, entry, requiredHeadings} of DOCUMENTED_ENTRIES) {
      const markdown = readFileSync(join(DOCS_ROOT, page), 'utf8');
      expect
        .soft(getMissingHeadings(markdown, requiredHeadings), `${page} lost required headings`)
        .toEqual([]);
      // Some family guides explicitly document contributors imported from another entry.
      const documentedEntries = new Set<string>(entry ? [entry] : []);
      for (const match of markdown.matchAll(/@luma-contributors\/analysis\/([a-z-]+)/g)) {
        documentedEntries.add(match[1]);
      }
      const entryExports = new Set(
        [...documentedEntries].flatMap(name => [
          ...(exportsByFile.get(`runtime:${join(name, 'index.ts')}`) ?? [])
        ])
      );
      documentedCount += getDocumentedNames(markdown).size;
      if (requiredHeadings.length) expect(entryExports.size, entry).toBeGreaterThan(0);
      const unreachable = getUnreachableNames(markdown, entryExports);
      expect
        .soft(unreachable, `${page} names missing from ${[...documentedEntries].join(', ')}`)
        .toEqual([]);
    }
    expect(documentedCount, 'documented contributor and recipe headings').toBeGreaterThan(0);
  });
});

// The extracted ingestion and profiling code must never acquire a dependency on its former host.
it('keeps reused analysis preparation independent of the Kepler host', () => {
  const families = ['dataframe-analysis/column-profile', 'gpu-spatial-analysis/spatial-weights'];
  for (const family of families) {
    for (const file of listSourceFiles(join(SOURCE_ROOT, family))) {
      const text = readFileSync(file, 'utf8');
      expect(text, relative(SOURCE_ROOT, file)).not.toMatch(
        /examples\/|@luma-contributors\/kepler|PreparedKepler/
      );
    }
  }
});
