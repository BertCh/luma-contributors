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
import {getDocSections} from './doc-heading-sections.test-utils';
import {ANALYSIS_DOC_PAGES} from './doc-program';
import {getAnalysisContributors} from './kernel-audit/analysis-contributors';
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

/** Verified grouped sections for contributors without their own heading, keyed by public entry. */
const GROUPED_CONTRIBUTOR_DOCS: Readonly<Record<string, string>> = {
  'raster-analysis/GPURasterEmbeddingProjection': 'gpu-raster/raster-embeddings.md#pca-to-rgb',
  'raster-analysis/GPURasterStandardizedIndexCalibration':
    'gpu-raster/raster-time-series.md#spi-and-spei',
  'raster-analysis/GPURasterStandardizedIndex': 'gpu-raster/raster-time-series.md#spi-and-spei',
  'gpu-spatial-analysis/GPURandomPointsOnLine':
    'gpu-spatial-analysis/operators-density.md#gpudotdensity-and-gpurandompointsinpolygon',
  'raster-analysis/GPURasterEmbeddingCovariance': 'gpu-raster/raster-embeddings.md#pca-to-rgb',
  'gpu-spatial-analysis/GPUAreaOfApplicability':
    'gpu-spatial-analysis/spatial-validation-advanced.md#area-of-applicability',
  'gpu-spatial-analysis/GPUNestedInterpolationValidation':
    'gpu-spatial-analysis/spatial-validation-advanced.md#nested-variogram-and-bandwidth-selection',
  'gpu-spatial-analysis/GPUPickRegionMask':
    'gpu-spatial-analysis/operators-statistics.md#gpuregionstatistics',
  'gpu-spatial-analysis/GPURegionMask':
    'gpu-spatial-analysis/operators-statistics.md#gpuregionstatistics',
  'gpu-spatial-analysis/GPURoseStatistic':
    'gpu-spatial-analysis/operators-statistics.md#additional-spatial-statistics-and-bounded-optimization',
  'gpu-spatial-analysis/GPUTrackSimilaritySearch':
    'gpu-spatial-analysis/track-similarity-measures.md#lower-bounds-and-pruned-search',
  'gpu-spatial-analysis/GPUTrajectoryKalman':
    'gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis',
  'gpu-spatial-analysis/GPUTrajectoryRestriction':
    'gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis',
  'gpu-spatial-analysis/GPUTransitionDiagnostics':
    'gpu-spatial-analysis/operators-dynamics.md#gputransitionmatrix',
  'gpu-spatial-analysis/addDasymetricRedistributionRecipe':
    'gpu-spatial-analysis/dasymetric-redistribution.md#dasymetric-redistribution',
  'gpu-spatial-analysis/addExcessRiskRecipe':
    'gpu-spatial-analysis/rate-mapping.md#rate-mapping-recipes',
  'gpu-spatial-analysis/addFleetDwellZoneEventsRecipe': 'gpu-spatial-analysis/recipes.md#recipes',
  'gpu-spatial-analysis/addSpatialRateRecipe':
    'gpu-spatial-analysis/rate-mapping.md#rate-mapping-recipes',
  'gpu-network/GPUAlternativeRoutes': 'gpu-network/alternative-routes.md#alternative-routes',
  'gpu-network/GPUMapMatchedEdgeVolumes':
    'gpu-network/map-matched-edge-volumes.md#edge-volumes-of-map-matched-tracks',
  'gpu-network/GPUNetworkFacilityOptimization':
    'gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis',
  'gpu-network/GPUNetworkFlow':
    'gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis',
  'gpu-network/GPUNetworkPointRouting':
    'gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis',
  'gpu-network/GPUNetworkRationalAccess':
    'gpu-network/network-movement-coverage.md#bounded-network-optimization-and-trajectory-analysis',
  'gpu-network/GPUNetworkStronglyConnectedComponents':
    'gpu-network/expansion.md#large-graph-strongly-connected-components',
  'gpu-network/GPUNetworkStructure':
    'gpu-network/expansion.md#directed-components-and-undirected-resilience',
  'gpu-network/addFacilityScenarioRecipe':
    'gpu-spatial-analysis/expansion-recipes.md#facility-scenario-evaluation',
  'gpu-network/addNetworkDisruptionRecipe':
    'gpu-spatial-analysis/expansion-recipes.md#network-disruption',
  'gpu-network/addTrackQualityRecipe':
    'gpu-network/expansion.md#track-quality-and-route-conformity',
  'dataframe-analysis/GPUSpaceTimeKFunction':
    'gpu-spatial-analysis/space-time-density-and-k-function.md#space-time-k-pair-correlation-local-statistics-gpuspacetimekfunction',
  'dataframe-analysis/GPUSpaceTimeDiggleTest':
    'gpu-spatial-analysis/space-time-density-and-k-function.md#diggle-test-of-space-time-interaction-gpuspacetimediggletest'
};

// These route references share a comparison section. They also have substantive named child
// headings now, so they are navigation pins rather than exceptions to canonical-heading coverage.
const PINNED_CONTRIBUTOR_REFERENCE_DOCS: Readonly<Record<string, string>> = {
  'gpu-spatial-analysis/addBuildingTessellationReferenceRecipe':
    'gpu-spatial-analysis/urban-morphology.md#addbuildingtessellationrecipe--ur01',
  'gpu-spatial-analysis/addLocalBuildingTessellationRecipe':
    'gpu-spatial-analysis/urban-morphology.md#addbuildingtessellationrecipe--ur01'
};

type ContributorDocPage = {page: string; entry: string | undefined; markdown: string};

/** Every identity resolves on its own entry's page, or to one explicit, verified grouped anchor. */
function getContributorDocIssues(
  contributors: ReadonlyMap<string, string>,
  pages: readonly ContributorDocPage[],
  grouped: Readonly<Record<string, string>>
): string[] {
  const identities = new Map(
    [...contributors].map(([key, entry]) => [
      `${entry}/${key.split('/').pop()!}`,
      key.split('/').pop()!
    ])
  );
  const issues: string[] = [];
  for (const key of Object.keys(grouped)) {
    if (!identities.has(key)) issues.push(`${key}: stale grouped mapping`);
  }
  for (const [identity, name] of identities) {
    const entry = identity.slice(0, identity.lastIndexOf('/'));
    const hasNamedHeading = pages.some(
      page =>
        page.entry === entry &&
        getDocSections(page.markdown).some(
          section =>
            !/ parameters$/i.test(section.heading) &&
            Boolean(section.body.trim()) &&
            section.heading.match(/\b(GPU[A-Z]\w+|add\w+Recipe)\b/g)?.includes(name)
        )
    );
    const destination = grouped[identity];
    if (!destination) {
      if (!hasNamedHeading)
        issues.push(`${identity}: missing canonical named heading or grouped destination`);
      continue;
    }
    if (hasNamedHeading)
      issues.push(`${identity}: grouped mapping is stale; canonical named heading exists`);
    const [path, anchor] = destination.split('#');
    const page = pages.find(candidate => candidate.page === path);
    const section =
      anchor &&
      page &&
      getDocSections(page.markdown).find(candidate => candidate.anchor === anchor);
    if (!section) {
      issues.push(`${identity}: missing grouped destination ${destination}`);
    } else if (
      !`${section.heading}\n${section.body}`
        .match(/\b(GPU[A-Z]\w+|add\w+Recipe)\b/g)
        ?.includes(name)
    ) {
      issues.push(`${identity}: ${destination} does not describe ${name}`);
    }
  }
  return issues;
}

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

/** Entry/Exports/Reference tables attribute only the exports on their own row. */
function getCatalogEntriesByContributor(markdown: string): Map<string, Set<string>> {
  const entries = new Map<string, Set<string>>();
  let inCatalog = false;
  for (const line of markdown.split('\n')) {
    if (/^\|\s*Entry\s*\|\s*Exports\s*\|\s*Reference\s*\|\s*$/.test(line)) {
      inCatalog = true;
      continue;
    }
    if (!line.startsWith('|')) {
      inCatalog = false;
      continue;
    }
    if (!inCatalog) continue;
    const row = /^\|\s*`([a-z-]+)`\s*\|([^|]+)\|/.exec(line);
    if (!row) continue;
    for (const name of row[2].match(/\b(GPU[A-Z]\w+|add\w+Recipe)\b/g) ?? []) {
      const namedEntries = entries.get(name) ?? new Set<string>();
      namedEntries.add(row[1]);
      entries.set(name, namedEntries);
    }
  }
  return entries;
}

function getUnreachableNames(
  markdown: string,
  entryExports: ReadonlySet<string>,
  exportsByEntry: ReadonlyMap<string, ReadonlySet<string>> = new Map()
): string[] {
  const catalogEntries = getCatalogEntriesByContributor(markdown);
  return [...getDocumentedNames(markdown)].filter(
    name =>
      !entryExports.has(name) &&
      ![...(catalogEntries.get(name) ?? [])].some(entry => exportsByEntry.get(entry)?.has(name))
  );
}

/** Resolve a page using imports plus contributor-specific catalog row attribution. */
function getPageUnreachableNames(
  markdown: string,
  entry: string | undefined,
  exportsByEntry: ReadonlyMap<string, ReadonlySet<string>>
): string[] {
  const documentedEntries = new Set<string>(entry ? [entry] : []);
  for (const match of markdown.matchAll(/@luma-contributors\/analysis\/([a-z-]+)/g)) {
    documentedEntries.add(match[1]);
  }
  const entryExports = new Set(
    [...documentedEntries].flatMap(name => [...(exportsByEntry.get(name) ?? [])])
  );
  return getUnreachableNames(markdown, entryExports, exportsByEntry);
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

function getRelativeImportLayers(
  file: string,
  source: string = readFileSync(file, 'utf8')
): Set<string> {
  const layers = new Set<string>();
  const tree = typescript.createSourceFile(file, source, typescript.ScriptTarget.Latest, true);
  const addSpecifier = (specifier: typescript.Node | undefined) => {
    while (specifier && typescript.isParenthesizedExpression(specifier))
      specifier = specifier.expression;
    if (
      !specifier ||
      !typescript.isStringLiteralLike(specifier) ||
      !/^\.{1,2}\//.test(specifier.text)
    )
      return;
    const target = relative(SOURCE_ROOT, resolve(dirname(file), specifier.text));
    const layer = target.split(sep)[0];
    if (!layer.startsWith('..')) layers.add(layer.replace(/\.ts$/, ''));
  };
  const visit = (node: typescript.Node) => {
    if (typescript.isImportDeclaration(node) || typescript.isExportDeclaration(node)) {
      addSpecifier(node.moduleSpecifier);
    } else if (typescript.isImportTypeNode(node) && typescript.isLiteralTypeNode(node.argument)) {
      addSpecifier(node.argument.literal);
    } else if (
      typescript.isImportEqualsDeclaration(node) &&
      typescript.isExternalModuleReference(node.moduleReference)
    ) {
      addSpecifier(node.moduleReference.expression);
    } else if (
      typescript.isCallExpression(node) &&
      (node.expression.kind === typescript.SyntaxKind.ImportKeyword ||
        (typescript.isIdentifier(node.expression) && node.expression.text === 'require'))
    ) {
      addSpecifier(node.arguments[0]);
    }
    typescript.forEachChild(node, visit);
  };
  visit(tree);
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

it('finds family import edges in either quote style and in every import form', () => {
  const file = join(SOURCE_ROOT, 'gpu-network', 'layering-fixture.ts');
  const source = [
    'import {a} from "../gpu-terrain/index";',
    "import type {B} from '../raster-analysis/index';",
    'export {c} from "../dataframe-analysis/index";',
    'const d = import("../pointcloud-analysis/index");',
    "import './local';",
    "// import {e} from '../geospatial-analysis/index';"
  ].join('\n');
  expect([...getRelativeImportLayers(file, source)].sort()).toEqual([
    'dataframe-analysis',
    'gpu-network',
    'gpu-terrain',
    'pointcloud-analysis',
    'raster-analysis'
  ]);
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
  const spatialOntologyFile = join('gpu-spatial-analysis', 'ontology', 'index.ts');

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
        const familyFile = relative(SOURCE_ROOT, index);
        if (familyFile === spatialOntologyFile) continue;
        checked++;
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
        const familyFile = relative(SOURCE_ROOT, index);
        if (familyFile === spatialOntologyFile) continue;
        const subdirectoryExports = exportsByFile.get(familyFile)!;
        const missing = [...subdirectoryExports].filter(name => !entryExports.has(name));
        expect(missing, familyFile).toEqual([]);
      }
    }
  });

  it('publishes ontology exports exclusively through the ontology entry', () => {
    const ontologyExports = exportsByFile.get(spatialOntologyFile)!;
    const parentExports = exportsByFile.get(join('gpu-spatial-analysis', 'index.ts'))!;
    expect(exportsByFile.get(`runtime:${spatialOntologyFile}`)).toContain(
      'AnalysisGeographicExtentAccumulator'
    );
    expect([...ontologyExports].filter(name => parentExports.has(name))).toEqual([]);
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
      documentedCount += getDocumentedNames(markdown).size;
      if (requiredHeadings.length)
        expect(
          exportsByFile.get(`runtime:${join(entry!, 'index.ts')}`)?.size,
          entry
        ).toBeGreaterThan(0);
      const exportsByEntry = new Map(
        ['analysis-core', 'computation-record', ...ENTRY_ORDER].map(name => [
          name,
          exportsByFile.get(`runtime:${join(name, 'index.ts')}`) ?? new Set<string>()
        ])
      );
      const unreachable = getPageUnreachableNames(markdown, entry, exportsByEntry);
      expect.soft(unreachable, `${page} names missing from its declared entries`).toEqual([]);
    }
    expect(documentedCount, 'documented contributor and recipe headings').toBeGreaterThan(0);
  });

  it('resolves every public contributor identity to canonical docs (CORE-116)', () => {
    const pages = DOCUMENTED_ENTRIES.map(({page, entry}) => ({
      page,
      entry,
      markdown: readFileSync(join(DOCS_ROOT, page), 'utf8')
    }));
    expect(
      getContributorDocIssues(getAnalysisContributors(), pages, GROUPED_CONTRIBUTOR_DOCS)
    ).toEqual([]);
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

it('rejects a removed grouped anchor, the right name on the wrong entry, and stale mappings (CORE-116)', () => {
  const contributors = new Map([['GPUExample', 'gpu-network']]);
  const page = {
    page: 'network.md',
    entry: 'gpu-network',
    markdown: '# Group\n\n`GPUExample` routes directed paths.'
  };
  const grouped = {'gpu-network/GPUExample': 'network.md#group'};
  expect(getContributorDocIssues(contributors, [page], grouped)).toEqual([]);
  expect(
    getContributorDocIssues(
      contributors,
      [{...page, markdown: page.markdown.replace('# Group', '# Changed')}],
      grouped
    )
  ).toEqual(['gpu-network/GPUExample: missing grouped destination network.md#group']);
  expect(
    getContributorDocIssues(
      contributors,
      [{...page, entry: 'gpu-spatial-analysis', markdown: '## `GPUExample`'}],
      {}
    )
  ).toEqual(['gpu-network/GPUExample: missing canonical named heading or grouped destination']);
  expect(getContributorDocIssues(new Map(), [page], grouped)).toEqual([
    'gpu-network/GPUExample: stale grouped mapping'
  ]);
  expect(
    getContributorDocIssues(
      contributors,
      [{...page, markdown: '# Group\n\nA different contributor.'}],
      grouped
    )
  ).toEqual(['gpu-network/GPUExample: network.md#group does not describe GPUExample']);
});

it('keeps two entries exporting the same name distinct in documentation coverage (CORE-116)', () => {
  const contributors = new Map([
    ['gpu-network/addExampleRecipe', 'gpu-network'],
    ['gpu-spatial-analysis/addExampleRecipe', 'gpu-spatial-analysis']
  ]);
  const pages = [
    {
      page: 'spatial.md',
      entry: 'gpu-spatial-analysis',
      markdown: '## `addExampleRecipe`\n\nBuilds a spatial graph.'
    }
  ];
  expect(getContributorDocIssues(contributors, pages, {})).toEqual([
    'gpu-network/addExampleRecipe: missing canonical named heading or grouped destination'
  ]);
  expect(
    getContributorDocIssues(
      new Map([['GPUExample', 'gpu-network']]),
      [
        {
          page: 'network.md',
          entry: 'gpu-network',
          markdown: '## `GPUExample` parameters\n\nGenerated fields.'
        }
      ],
      {}
    )
  ).toEqual(['gpu-network/GPUExample: missing canonical named heading or grouped destination']);
});

it('rejects grouped evidence consisting only of generated child parameters (B7)', () => {
  const contributors = new Map([['GPUExample', 'gpu-network']]);
  const grouped = {'gpu-network/GPUExample': 'network.md#group'};
  const generated = `{/* parameters: GPUExample generated; do not edit */}
#### \`GPUExample\` parameters

| Field | Type | Description |
|---|---|---|
| capacity | number | Number of rows. |
{/* /parameters */}`;
  const page = {page: 'network.md', entry: 'gpu-network', markdown: `# Group\n\n${generated}`};
  expect(getContributorDocIssues(contributors, [page], grouped)).toEqual([
    'gpu-network/GPUExample: network.md#group does not describe GPUExample'
  ]);
  expect(
    getContributorDocIssues(
      contributors,
      [
        {
          ...page,
          markdown:
            '# Group\n\n#### `GPUExample` parameters\n\n| Field | Type |\n|---|---|\n| capacity | number |'
        }
      ],
      grouped
    )
  ).toEqual(['gpu-network/GPUExample: network.md#group does not describe GPUExample']);
  expect(
    getContributorDocIssues(
      contributors,
      [{...page, markdown: `# Group\n\nGPUExample routes paths.\n\n${generated}`}],
      grouped
    )
  ).toEqual([]);
});

it('uses the site renderer heading text and duplicate slugs for linked headings (S4)', () => {
  const contributors = new Map([['GPUExample', 'gpu-network']]);
  const page = {
    page: 'network.md',
    entry: 'gpu-network',
    markdown:
      '# [Overview](url)\n\nGPUExample routes paths.\n\n# [Overview](other)\n\nGPUExample controls.'
  };
  expect(
    getContributorDocIssues(contributors, [page], {'gpu-network/GPUExample': 'network.md#overview'})
  ).toEqual([]);
  expect(
    getContributorDocIssues(contributors, [page], {
      'gpu-network/GPUExample': 'network.md#overview-1'
    })
  ).toEqual([]);
  expect(
    getContributorDocIssues(contributors, [page], {
      'gpu-network/GPUExample': 'network.md#overviewurl'
    })
  ).toEqual(['gpu-network/GPUExample: missing grouped destination network.md#overviewurl']);
});

it('attributes catalog entry exports only to the matching contributor row (S5)', () => {
  const exportsByEntry = new Map([['analysis-core', new Set(['GPUExample', 'GPUOther'])]]);
  const heading = '## `GPUExample`\n\n';
  const catalog = (name: string) =>
    `| Entry | Exports | Reference |\n|---|---|---|\n| \`analysis-core\` | \`${name}\` | [Contract](#gpuexample) |`;
  expect(
    getPageUnreachableNames(heading + catalog('GPUExample'), 'gpu-network', exportsByEntry)
  ).toEqual([]);
  expect(
    getPageUnreachableNames(heading + catalog('GPUOther'), 'gpu-network', exportsByEntry)
  ).toEqual(['GPUExample']);
  expect(
    getPageUnreachableNames(
      heading + '| Entry | Details |\n|---|---|\n| `analysis-core` | unrelated |',
      'gpu-network',
      exportsByEntry
    )
  ).toEqual(['GPUExample']);
});

it('discovers parenthesized literal dynamic imports without treating expressions as imports (review import note)', () => {
  const file = join(SOURCE_ROOT, 'gpu-spatial-analysis', 'parenthesized-fixture.ts');
  expect([
    ...getRelativeImportLayers(file, "const module = import((('../gpu-network/index')));")
  ]).toEqual(['gpu-network']);
  expect([
    ...getRelativeImportLayers(file, "const module = import('../' + family + '/index');")
  ]).toEqual([]);
});

it('DOC-108 pins Rose to its circular-statistics section rather than coordinate rejection', () => {
  const identity = 'gpu-spatial-analysis/GPURoseStatistic';
  const destination =
    'gpu-spatial-analysis/operators-statistics.md#additional-spatial-statistics-and-bounded-optimization';
  expect(GROUPED_CONTRIBUTOR_DOCS[identity]).toBe(destination);
  const markdown = readFileSync(join(DOCS_ROOT, destination.split('#')[0]), 'utf8');
  const target = getDocSections(markdown).find(
    section => section.anchor === destination.split('#')[1]
  );
  expect(target?.body).toContain('GPURoseStatistic');
  expect(GROUPED_CONTRIBUTOR_DOCS[identity]).not.toBe(
    'gpu-spatial-analysis/operators-statistics.md#gpugeographicdistribution'
  );
});

it('W5-15 resolves edge volumes and stop cubes to substantive canonical sections', () => {
  const edgeDestination =
    'gpu-network/map-matched-edge-volumes.md#edge-volumes-of-map-matched-tracks';
  expect(GROUPED_CONTRIBUTOR_DOCS['gpu-network/GPUMapMatchedEdgeVolumes']).toBe(edgeDestination);
  for (const [entry, name, destination] of [
    ['gpu-network', 'GPUMapMatchedEdgeVolumes', edgeDestination],
    [
      'gpu-spatial-analysis',
      'addStopValueCubeRecipe',
      'gpu-spatial-analysis/movement-semantics.md#addstopvaluecuberecipe'
    ]
  ]) {
    const [page, anchor] = destination.split('#');
    const markdown = readFileSync(join(DOCS_ROOT, page), 'utf8');
    const section = getDocSections(markdown).find(candidate => candidate.anchor === anchor);
    expect(section, destination).toBeDefined();
    expect(section!.body, destination).toContain(name);
    expect(
      getContributorDocIssues(
        new Map([[name, entry]]),
        [{page, entry, markdown}],
        name === 'GPUMapMatchedEdgeVolumes' ? {[`${entry}/${name}`]: destination} : {}
      )
    ).toEqual([]);
  }
});

it.each([
  'addBuildingTessellationReferenceRecipe',
  'addLocalBuildingTessellationRecipe'
])('CORE-116 pins %s to substantive urban morphology documentation', name => {
  const destination =
    'gpu-spatial-analysis/urban-morphology.md#addbuildingtessellationrecipe--ur01';
  expect(PINNED_CONTRIBUTOR_REFERENCE_DOCS[`gpu-spatial-analysis/${name}`]).toBe(destination);
  expect(GROUPED_CONTRIBUTOR_DOCS[`gpu-spatial-analysis/${name}`]).toBeUndefined();
  const markdown = readFileSync(
    join(DOCS_ROOT, 'gpu-spatial-analysis/urban-morphology.md'),
    'utf8'
  );
  const section = getDocSections(markdown).find(
    candidate => candidate.anchor === 'addbuildingtessellationrecipe--ur01'
  );
  expect(section, 'rendered heading anchor').toBeDefined();
  expect(section!.body).toContain(`\`${name}\``);
  const route = name === 'addBuildingTessellationReferenceRecipe' ? 'reference' : 'local';
  expect(section!.body).toMatch(new RegExp('`' + name + '`\\s+for the ' + route + ' route\\b'));
  expect(
    getContributorDocIssues(
      new Map([[name, 'gpu-spatial-analysis']]),
      [{page: 'urban.md', entry: 'gpu-spatial-analysis', markdown}],
      {}
    )
  ).toEqual([]);
});

it('REVIEW B6 rejects empty tessellation route headings as substantive evidence', () => {
  const markdown =
    '## `addBuildingTessellationRecipe` — UR01\n\n### `addBuildingTessellationReferenceRecipe`\n\n### `addLocalBuildingTessellationRecipe`\n';
  const parent = getDocSections(markdown).find(
    section => section.anchor === 'addbuildingtessellationrecipe--ur01'
  )!;
  expect(parent.body).not.toContain('addBuildingTessellationReferenceRecipe');
  expect(parent.body).not.toContain('addLocalBuildingTessellationRecipe');
  const issues = getContributorDocIssues(
    new Map([
      ['addBuildingTessellationReferenceRecipe', 'gpu-spatial-analysis'],
      ['addLocalBuildingTessellationRecipe', 'gpu-spatial-analysis']
    ]),
    [{page: 'urban.md', entry: 'gpu-spatial-analysis', markdown}],
    {}
  );
  expect(issues).toHaveLength(2);
});

it('REVIEW B7 discovers literal CommonJS and parenthesized dynamic family imports', () => {
  const file = join(SOURCE_ROOT, 'gpu-network', 'commonjs-fixture.ts');
  expect([
    ...getRelativeImportLayers(file, 'const module = require("../gpu-terrain/index");')
  ]).toEqual(['gpu-terrain']);
  expect([
    ...getRelativeImportLayers(file, 'const module = import(("../gpu-terrain/index"));')
  ]).toEqual(['gpu-terrain']);
});
