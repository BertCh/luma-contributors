// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, relative, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {expect, it} from 'vitest';
import {createKernelAuditContext} from './kernel-audit/kernel-audit-context';
import {
  getGPUNeighborSearchParameterValues,
  getGPUNeighborSearchWeightTransform
} from '../../src/gpu-spatial-analysis/neighbor-search/index';
import {GPUNearestFeatureWeights} from '../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import {GPUPairGather} from '../../src/gpu-spatial-analysis/pair-gather/index';
import {GPUKeyLookup} from '../../src/dataframe-analysis/key-join/index';
import {
  GPUGroupStatistics,
  type GPUGroupStatisticsColumnOutput
} from '../../src/dataframe-analysis/group-statistics/index';
import {getRepositoryCompilerOptions} from './doc-program';
import {
  getPublicDeclarations,
  getPublicDeclarationsFromSources,
  getPublicDeclarationNode
} from './public-declarations';

// Rule 18: resolve qualified public data shapes from the shared declaration inventory, including
// aliases, inheritance, indexed access and mapped members. Props/Settings inputs use canonical
// names; GPU-written output status belongs in status. Output vocabulary is checked through its
// own shape rather than applying input naming rules to analytical result columns. Grid descriptor
// dimensions stay within their descriptor rather than loose construction props.
// Reviewed compatibility exceptions name entry, export, defining declaration and member exactly.
// A new namesake fails, as does an exception whose declaration/member no longer occurs.
//
// Regenerate with `UPDATE_VOCABULARY_ALLOWLIST=1 npx vitest run --project node
// modules/analysis/test/analysis-core/analysis-vocabulary.node.spec.ts`; it keeps existing
// reasons and leaves new entries without one, so they fail until reviewed.

const SOURCE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../src');
const ALLOWLIST_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  'vocabulary-allowlist.json'
);

const ANALYSIS_LAYERS = [
  'analysis-core',
  'computation-record',
  'geospatial-analysis',
  'dataframe-analysis',
  'raster-analysis',
  'pointcloud-analysis',
  'gpu-terrain',
  'gpu-spatial-analysis',
  'gpu-network'
] as const;

type VocabularyRule = {
  /** Canonical replacement, reported with each finding. */
  canonical: string;
  /** True when a property name (with its type text) is non-canonical. */
  matches: (name: string, typeText: string, layer: string) => boolean;
};

const PACKED_VIEW_TYPE = /GraphDataView|GraphVectorView|GPUParameterBuffer/;

/** Coordinate components of bounds and extents, exempt from the `max*` / `min*` rules. */
const COORDINATE_COMPONENT = /^(min|max)[XYZ]$/;

/**
 * Status words (quality roadmap "Canonical vocabulary"): cardinality, completeness, convergence,
 * diagnostics and their stage-prefixed forms (`pairOverflow`, `weightsRequiredCount`).
 */
const STATUS_WORD =
  /^(count|converged|approximation|iterationCount|iterationLimitReached|\w*[oO]verflow|\w*[rR]equired\w*Count|\w*[iI]nvalidCount|\w*[uU]ncertainCount)$/;

/** The canonical-vocabulary table as checks. The rule ID is the replaced spelling. */
const VOCABULARY_RULES: Record<string, VocabularyRule> = {
  settings: {
    canonical: 'parameters (packed per-frame view)',
    matches: (name, typeText) => name === 'settings' && PACKED_VIEW_TYPE.test(typeText)
  },
  wordParameters: {canonical: 'uint32Parameters', matches: name => name === 'wordParameters'},
  'max*': {
    canonical: 'maximum*',
    matches: name => /^max[A-Z0-9]/.test(name) && !COORDINATE_COMPONENT.test(name)
  },
  'min*': {
    canonical: 'minimum*',
    matches: name => /^min[A-Z0-9]/.test(name) && !COORDINATE_COMPONENT.test(name)
  },
  iterations: {
    canonical: 'maximumIterations (input), iterationCount (output)',
    matches: name => name === 'iterations' || name === 'maximumRounds' || name === 'roundCount'
  },
  '*Passes': {canonical: 'maximumIterations', matches: name => /[a-z]Passes$|^passes$/.test(name)},
  maximumCapacity: {canonical: '<noun>Capacity', matches: name => name === 'maximumCapacity'},
  permutations: {
    canonical: 'maximumPermutations (compile time), permutationCount (per frame)',
    matches: name => name === 'permutations'
  },
  'row selection': {
    canonical: 'mask (input), outputMask (output)',
    matches: name => ['keepMask', 'selection', 'valueMask', 'included'].includes(name)
  },
  polygons: {
    canonical: 'GPUPolygonGeometryPort four-level layout',
    matches: name => name === 'polygonPositions' || name === 'featureRingOffsets'
  },
  lines: {
    canonical: 'positions + lineOffsets / trackOffsets',
    matches: name => name === 'pathOffsets' || name === 'startIndices'
  },
  network: {
    canonical: 'topology: {offsets, neighbors, weights, nodePositions}; costBreaks, costLimit',
    // Class `breaks` of classifications and isobands are canonical; only cost bands are renamed.
    matches: (name, _typeText, layer) =>
      ['edgeTargets', 'edgeCosts', 'bandThresholds', 'bandBreaks'].includes(name) ||
      (layer === 'gpu-network' && name === 'breaks')
  },
  'raster grid': {
    canonical: 'grid: GPURasterMetadata',
    matches: (name, _typeText, layer) =>
      (layer === 'raster-analysis' || layer === 'gpu-terrain') &&
      ['width', 'height', 'cellSize'].includes(name)
  },
  'seed starts': {
    canonical: 'start* (graph starts); seed only for random seeds',
    // Distance-field and catchment seeds are sources, not graph starts.
    matches: (name, _typeText, layer) =>
      layer === 'gpu-network' && (name === 'seeds' || /^seed[A-Z]/.test(name))
  },
  results: {canonical: 'output', matches: name => name === 'result' || name === 'results'},
  geographic: {canonical: 'spatialContext', matches: name => name === 'geographic'},
  falseDiscoveryRate: {
    canonical: 'multipleTesting',
    matches: name => name === 'falseDiscoveryRate'
  }
};

/** The status-placement rule: GPU-written status words live in a `status` member. */
const STATUS_RULE_ID = 'status';
const STATUS_CANONICAL = 'status.<word> (for example output.status.overflow)';

/** Status ports themselves (`GPUCardinalityPort`, `GPUIterationStatusPort`, ...) define the words. */
const STATUS_PORT_TYPE = /^GPU\w*(Status|StatusPort|CardinalityPort)$/;

/** True when `path` (`<Type>.<member>...`) places a status word outside a `status` member. */
function isMisplacedStatus(
  typeName: string,
  path: string,
  name: string,
  isScalarStatusView: boolean
): boolean {
  if (!STATUS_WORD.test(name) || !isScalarStatusView) {
    return false;
  }
  if (STATUS_PORT_TYPE.test(typeName)) {
    return false;
  }
  const segments = path.split('.').slice(1, -1);
  if (segments.includes('status')) {
    return false;
  }
  // Top-level Props/Settings views are supplied inputs, including producer loss evidence.
  // GPU-written output/status types are checked separately, as are nested output members.
  return !(segments.length === 0 && /(Props|Settings)$/.test(typeName));
}

/** `{<layer>: {"<Type>.<property path>": rule ID}}`. */
type VocabularyFindings = Record<string, Record<string, string>>;

/** `{<layer>: {"<Type>.<property path>": {rule, reason}}}`. */
type VocabularyAllowlist = Record<string, Record<string, {rule: string; reason: string}>>;

function getVocabularyFindings(sourceRoot: string = SOURCE_ROOT): VocabularyFindings {
  const findings: VocabularyFindings = Object.fromEntries(
    ANALYSIS_LAYERS.map(layer => [layer, {}])
  );
  const program = typescript.createProgram(
    ANALYSIS_LAYERS.map(layer => join(sourceRoot, layer, 'index.ts')),
    sourceRoot === SOURCE_ROOT
      ? getRepositoryCompilerOptions()
      : {
          target: typescript.ScriptTarget.ES2022,
          module: typescript.ModuleKind.ESNext,
          moduleResolution: typescript.ModuleResolutionKind.Bundler,
          strict: true,
          lib: ['lib.es2022.d.ts']
        }
  );
  const checker = program.getTypeChecker();
  // Alias spelling and object typeToString output are not identities. The class/interface
  // symbol remains the framework declaration even when a local type alias names the view.
  const frameworkViews = new Map<typescript.Symbol, string>();
  for (const source of program.getSourceFiles())
    for (const statement of source.statements) {
      if (
        !(
          typescript.isClassDeclaration(statement) || typescript.isInterfaceDeclaration(statement)
        ) ||
        !statement.name
      )
        continue;
      if (!['GraphDataView', 'GraphVectorView', 'GPUParameterBuffer'].includes(statement.name.text))
        continue;
      const symbol = checker.getSymbolAtLocation(statement.name);
      if (symbol) frameworkViews.set(symbol, statement.name.text);
    }
  const getViewKind = (type: typescript.Type): string | undefined => {
    const symbol = type.getNonNullableType().getSymbol();
    return symbol && frameworkViews.get(symbol);
  };
  const isScalarStatusView = (type: typescript.Type): boolean => {
    const resolved = type.getNonNullableType();
    if (getViewKind(resolved) !== 'GraphDataView') return false;
    const format = checker.getPropertyOfType(resolved, 'format');
    if (!format) return false;
    const location = format.valueDeclaration ?? format.declarations?.[0];
    const formatType = location && checker.getTypeOfSymbolAtLocation(format, location);
    return Boolean(formatType?.isStringLiteral() && formatType.value === 'uint32');
  };

  const inventory =
    sourceRoot === SOURCE_ROOT
      ? getPublicDeclarations(program)
      : getPublicDeclarationsFromSources(program, sourceRoot, {
          name: '@luma-contributors/analysis',
          exports: Object.fromEntries(
            ANALYSIS_LAYERS.map(layer => [`./${layer}`, `./dist/${layer}/index.js`])
          )
        });
  const inputTypes = new Set<typescript.Type>();
  const outputTypes = new Set<typescript.Type>();
  const outputDescriptors = new Set<typescript.Type>();
  const registerInputType = (type: typescript.Type): void => {
    if (inputTypes.has(type)) return;
    inputTypes.add(type);
    if (type.isUnionOrIntersection()) for (const part of type.types) registerInputType(part);
  };
  const publicTypes = inventory.flatMap(record => {
    const declaration = getPublicDeclarationNode(program, record.declaration);
    if (
      !declaration ||
      !(
        typescript.isTypeAliasDeclaration(declaration) ||
        typescript.isInterfaceDeclaration(declaration)
      )
    )
      return [];
    const type = checker.getTypeAtLocation(declaration);
    if (/(Props|Settings)$/.test(record.exportName)) registerInputType(type);
    return [type];
  });
  const inspected = new Set<typescript.Type>();
  const collectOutputTypes = (type: typescript.Type): void => {
    if (inspected.has(type) || getViewKind(type)) return;
    inspected.add(type);
    // An intersection can keep the scalar writer and its status fields in separate parts.
    // Classify the merged descriptor before descending so neither part loses ownership.
    const outputMember = checker.getPropertyOfType(type, 'output');
    const outputLocation = outputMember?.valueDeclaration ?? outputMember?.declarations?.[0];
    if (
      outputMember &&
      outputLocation &&
      !inputTypes.has(type) &&
      getViewKind(checker.getTypeOfSymbolAtLocation(outputMember, outputLocation))
    )
      outputDescriptors.add(type);
    if (type.isUnionOrIntersection()) {
      for (const part of type.types) collectOutputTypes(part);
      return;
    }
    for (const kind of [typescript.IndexKind.Number, typescript.IndexKind.String]) {
      const value = checker.getIndexTypeOfType(type, kind);
      if (value) collectOutputTypes(value);
    }
    if (checker.isArrayType(type) || checker.isTupleType(type)) return;
    for (const member of checker.getPropertiesOfType(type)) {
      const location = member.valueDeclaration ?? member.declarations?.[0];
      if (!location || !resolve(location.getSourceFile().fileName).startsWith(sourceRoot + sep))
        continue;
      if (
        typescript.canHaveModifiers(location) &&
        typescript
          .getModifiers(location)
          ?.some(
            modifier =>
              modifier.kind === typescript.SyntaxKind.PrivateKeyword ||
              modifier.kind === typescript.SyntaxKind.ProtectedKeyword
          )
      )
        continue;
      const value = checker.getTypeOfSymbolAtLocation(member, location).getNonNullableType();
      if (member.getName() === 'output' || member.getName() === 'outputs') {
        outputTypes.add(value);
        // Mixed descriptors such as GPUKeyLookupAggregate contain an output view and its loss
        // flag. Props with a scalar output still carry supplied input evidence separately.
        if (getViewKind(value) && !inputTypes.has(type)) outputDescriptors.add(type);
      }
      collectOutputTypes(value);
    }
  };
  for (const type of publicTypes) collectOutputTypes(type);
  for (const record of inventory) {
    const layer = record.entryPoint.replace('@luma-contributors/analysis/', '');
    if (!(ANALYSIS_LAYERS as readonly string[]).includes(layer)) continue;
    const declaration = getPublicDeclarationNode(program, record.declaration);
    if (
      !declaration ||
      !(
        typescript.isTypeAliasDeclaration(declaration) ||
        typescript.isInterfaceDeclaration(declaration)
      )
    )
      continue;
    const typeName = record.exportName;
    if (!/^GPU\w+$/.test(typeName)) continue;
    const checksVocabulary = /(Props|Settings)$/.test(typeName);
    const originSeparator = record.declaration.lastIndexOf('#');
    const origin =
      relative(sourceRoot, record.declaration.slice(0, originSeparator)).split(sep).join('/') +
      record.declaration.slice(originSeparator);
    const qualifiedName = `${record.entryPoint}:${typeName}@${origin}`;
    const walk = (
      type: typescript.Type,
      path: string,
      visiting: Set<typescript.Type>,
      isOutput = false
    ): void => {
      if (visiting.has(type)) return;
      const next = new Set(visiting).add(type);
      const ownsOutput = isOutput || outputDescriptors.has(type);
      if (type.isUnionOrIntersection()) {
        for (const member of type.types) walk(member, path, next, ownsOutput);
        return;
      }
      // A packed view is a port, not an object whose framework implementation is public props.
      if (getViewKind(type)) return;
      for (const kind of [typescript.IndexKind.Number, typescript.IndexKind.String]) {
        const value = checker.getIndexTypeOfType(type, kind);
        if (value) walk(value, `${path}[]`, next, ownsOutput);
      }
      if (checker.isArrayType(type) || checker.isTupleType(type)) return;
      for (const member of checker.getPropertiesOfType(type)) {
        const location =
          member.valueDeclaration ??
          member.declarations?.[0] ??
          type.aliasSymbol?.declarations?.[0] ??
          type.getSymbol()?.declarations?.[0];
        if (!location || !resolve(location.getSourceFile().fileName).startsWith(sourceRoot + sep))
          continue;
        if (
          typescript.canHaveModifiers(location) &&
          typescript
            .getModifiers(location)
            ?.some(
              modifier =>
                modifier.kind === typescript.SyntaxKind.PrivateKeyword ||
                modifier.kind === typescript.SyntaxKind.ProtectedKeyword
            )
        )
          continue;
        const memberName = member.getName();
        const memberPath = `${path}.${memberName}`;
        const memberType = checker.getTypeOfSymbolAtLocation(member, declaration);
        const typeText = getViewKind(memberType) ?? '';
        const key = `${qualifiedName}${memberPath}`;
        if (checksVocabulary && !memberPath.split('.').includes('output'))
          for (const [ruleId, rule] of Object.entries(VOCABULARY_RULES)) {
            // Grid descriptors carry dimensions; the rule replaces loose top-level dimensions.
            if (ruleId === 'raster grid' && memberPath.split('.').length > 2) continue;
            if (rule.matches(memberName, typeText, layer)) findings[layer][key] = ruleId;
          }
        if (
          ownsOutput &&
          isMisplacedStatus(
            typeName,
            typeName + memberPath,
            memberName,
            isScalarStatusView(memberType)
          )
        )
          findings[layer][key] = STATUS_RULE_ID;
        walk(
          memberType,
          memberPath,
          next,
          ownsOutput || memberName === 'output' || memberName === 'outputs'
        );
      }
    };
    const rootType = checker.getTypeAtLocation(declaration);
    walk(rootType, '', new Set(), outputTypes.has(rootType) || outputDescriptors.has(rootType));
  }
  return findings;
}

function getAllowlistIssues(
  findings: VocabularyFindings,
  allowlist: VocabularyAllowlist
): {added: string[]; stale: string[]; unexplained: string[]} {
  const added: string[] = [];
  const stale: string[] = [];
  const unexplained: string[] = [];
  for (const layer of ANALYSIS_LAYERS) {
    const found = findings[layer];
    const allowed = allowlist[layer] || {};
    for (const [path, ruleId] of Object.entries(found)) {
      if (allowed[path]?.rule !== ruleId) {
        const canonical =
          ruleId === STATUS_RULE_ID ? STATUS_CANONICAL : VOCABULARY_RULES[ruleId].canonical;
        added.push(`${layer} ${path}: '${ruleId}' -> ${canonical}`);
      }
    }
    for (const [path, {reason}] of Object.entries(allowed)) {
      if (!(path in found))
        stale.push(`${layer} ${path} (fixed: remove it from vocabulary-allowlist.json)`);
      else if (!reason) unexplained.push(`${layer} ${path}`);
    }
  }
  return {added, stale, unexplained};
}

it('uses the canonical vocabulary in every analysis Props and Settings type', () => {
  const findings = getVocabularyFindings();
  if (process.env['UPDATE_VOCABULARY_ALLOWLIST']) {
    const previous = JSON.parse(readFileSync(ALLOWLIST_PATH, 'utf8')) as VocabularyAllowlist;
    const next: VocabularyAllowlist = {};
    for (const [layer, entries] of Object.entries(findings)) {
      for (const [path, rule] of Object.entries(entries)) {
        const reason = previous[layer]?.[path]?.reason ?? '';
        next[layer] = {...next[layer], [path]: {rule, reason}};
      }
    }
    writeFileSync(ALLOWLIST_PATH, `${JSON.stringify(next, null, 2)}\n`);
  }
  const allowlist = JSON.parse(readFileSync(ALLOWLIST_PATH, 'utf8')) as VocabularyAllowlist;
  const {added, stale, unexplained} = getAllowlistIssues(findings, allowlist);
  expect(stale, 'stale allowlist entries').toEqual([]);
  expect(unexplained, 'allowlist entries without a reason').toEqual([]);
  expect(added, 'non-canonical names').toEqual([]);
});

it('discovers forbidden point-cloud names and accepts the canonical control (CORE-14)', () => {
  const sourceRoot = mkdtempSync(join(tmpdir(), 'analysis-vocabulary-'));
  try {
    for (const layer of [...ANALYSIS_LAYERS, 'pointcloud-analysis']) {
      mkdirSync(join(sourceRoot, layer), {recursive: true});
    }
    writeFileSync(
      join(sourceRoot, 'pointcloud-analysis', 'index.ts'),
      `export type GPUPointCloudFixtureProps = {maxRows: number};
export type GPUPointCloudControlProps = {maximumRows: number; mask?: GraphDataView<'uint32'>};`
    );
    const findings = getVocabularyFindings(sourceRoot);
    expect(Object.values(findings['pointcloud-analysis'])).toEqual(['max*']);
    expect(Object.keys(findings['pointcloud-analysis'])[0]).toContain(
      ':GPUPointCloudFixtureProps@pointcloud-analysis/index.ts#GPUPointCloudFixtureProps.maxRows'
    );
  } finally {
    rmSync(sourceRoot, {recursive: true, force: true});
  }
});

it('resolves aliased, inherited, intersected, indexed and mapped public members (ERG-104)', () => {
  const sourceRoot = mkdtempSync(join(tmpdir(), 'analysis-vocabulary-members-'));
  try {
    for (const layer of [...ANALYSIS_LAYERS, 'computation-record'])
      mkdirSync(join(sourceRoot, layer), {recursive: true});
    writeFileSync(
      join(sourceRoot, 'pointcloud-analysis', 'types.ts'),
      `
interface Base {maxRows: number}
type Container = {selected: Base & {minimumRows: number}; private: {minPrivate: number}};
type Projected = {[Key in 'selection']: number};
class Options {maxOptions = 1; private minPrivate = 0;}
export type GPUPublicProps = Container['selected'] & Projected & {options: Options};
`
    );
    writeFileSync(
      join(sourceRoot, 'pointcloud-analysis', 'index.ts'),
      "export {GPUPublicProps as GPUAliasProps} from './types';"
    );
    writeFileSync(
      join(sourceRoot, 'computation-record', 'index.ts'),
      'export interface GPURecordProps extends Base {iterations: number}; interface Base {maxRecords: number}'
    );
    const findings = getVocabularyFindings(sourceRoot);
    expect(Object.keys(findings['pointcloud-analysis']).some(key => key.endsWith('.maxRows'))).toBe(
      true
    );
    expect(
      Object.keys(findings['pointcloud-analysis']).some(key => key.endsWith('.selection'))
    ).toBe(true);
    expect(
      Object.keys(findings['pointcloud-analysis']).some(key => key.endsWith('.options.maxOptions'))
    ).toBe(true);
    expect(
      Object.keys(findings['pointcloud-analysis']).some(key => key.includes('minPrivate'))
    ).toBe(false);
    expect(
      Object.keys(findings['computation-record']).some(key => key.endsWith('.iterations'))
    ).toBe(true);
    expect(
      Object.keys(findings['computation-record']).some(key => key.endsWith('.maxRecords'))
    ).toBe(true);
  } finally {
    rmSync(sourceRoot, {recursive: true, force: true});
  }
});

it('keeps compatibility exceptions exact and rejects new namesakes and stale exceptions (ERG-104)', () => {
  const sourceRoot = mkdtempSync(join(tmpdir(), 'analysis-vocabulary-compatibility-'));
  try {
    for (const layer of ANALYSIS_LAYERS) mkdirSync(join(sourceRoot, layer), {recursive: true});
    const fixturePath = join(sourceRoot, 'pointcloud-analysis', 'index.ts');
    const allowlist: VocabularyAllowlist = {
      'pointcloud-analysis': {
        '@luma-contributors/analysis/pointcloud-analysis:GPURegistrationFixtureProps@pointcloud-analysis/index.ts#GPURegistrationFixtureProps.iterations':
          {rule: 'iterations', reason: 'Fixture registration budget.'},
        '@luma-contributors/analysis/pointcloud-analysis:GPUGroundFilterFixtureProps@pointcloud-analysis/index.ts#GPUGroundFilterFixtureProps.selection':
          {rule: 'row selection', reason: 'Fixture input mask.'}
      }
    };
    writeFileSync(
      fixturePath,
      `
export type GPURegistrationFixtureProps = {iterations?: number};
export type GPUGroundFilterFixtureProps = {selection?: number};
`
    );
    expect(getAllowlistIssues(getVocabularyFindings(sourceRoot), allowlist)).toEqual({
      added: [],
      stale: [],
      unexplained: []
    });
    writeFileSync(
      fixturePath,
      `
export type GPURegistrationFixtureProps = {maximumIterations?: number};
export type GPUGroundFilterFixtureProps = {mask?: number};
export type GPUOtherRegistrationFixtureProps = {iterations?: number};
export type GPUOtherGroundFilterFixtureProps = {selection?: number};
`
    );
    const issues = getAllowlistIssues(getVocabularyFindings(sourceRoot), allowlist);
    expect(issues.added).toHaveLength(2);
    expect(issues.stale).toHaveLength(2);
    expect(issues.unexplained).toEqual([]);
  } finally {
    rmSync(sourceRoot, {recursive: true, force: true});
  }
});

it('resolves framework view identities without hiding containing objects or aliased status scalars (review B4)', () => {
  const sourceRoot = mkdtempSync(join(tmpdir(), 'vocabulary-view-identity-'));
  try {
    for (const layer of ANALYSIS_LAYERS) mkdirSync(join(sourceRoot, layer), {recursive: true});
    writeFileSync(
      join(sourceRoot, 'pointcloud-analysis', 'index.ts'),
      `
class GraphDataView<Format extends string> {declare format: Format;}
type CountWord = GraphDataView<'uint32'>;
export type GPUProbeProps = {options: {reading: GraphDataView<'float32'>; maxSamples: number}; output: GPUProbeOutput};
export type GPUProbeOutput = {count: CountWord; invalidCount: GraphDataView<'uint32'>};
`
    );
    const findings = getVocabularyFindings(sourceRoot)['pointcloud-analysis'];
    expect(Object.keys(findings).some(key => key.endsWith('.options.maxSamples'))).toBe(true);
    expect(Object.keys(findings).some(key => key.endsWith('.count'))).toBe(true);
    expect(Object.keys(findings).some(key => key.endsWith('.invalidCount'))).toBe(true);
  } finally {
    rmSync(sourceRoot, {recursive: true, force: true});
  }
});

it('visits arrays and dictionaries and detects output-bearing descriptors without an Output suffix (review B5)', () => {
  const sourceRoot = mkdtempSync(join(tmpdir(), 'vocabulary-collections-'));
  try {
    for (const layer of ANALYSIS_LAYERS) mkdirSync(join(sourceRoot, layer), {recursive: true});
    writeFileSync(
      join(sourceRoot, 'dataframe-analysis', 'index.ts'),
      `
class GraphDataView<Format extends string> {declare format: Format;}
type Value = {selection: GraphDataView<'uint32'>};
export type GPUProbeProps = {values: Value[]; dictionary: Record<string, Value>; producerOverflow: GraphDataView<'uint32'>};
export type GPUIntersectAggregate = {output: GraphDataView<'float32'>} & {sumOverflow: GraphDataView<'uint32'>};
export type GPUProbeAggregate = {output: GraphDataView<'float32'>; sumOverflow: GraphDataView<'uint32'>};
export type GPUScalarProps = {output: GraphDataView<'float32'>; producerOverflow: GraphDataView<'uint32'>} & {id: string};
`
    );
    const findings = getVocabularyFindings(sourceRoot)['dataframe-analysis'];
    expect(Object.keys(findings).some(key => key.endsWith('.values[].selection'))).toBe(true);
    expect(Object.keys(findings).some(key => key.endsWith('.dictionary[].selection'))).toBe(true);
    expect(
      Object.keys(findings).some(
        key => key.includes(':GPUIntersectAggregate@') && key.endsWith('.sumOverflow')
      )
    ).toBe(true);
    expect(
      Object.keys(findings).some(
        key => key.includes(':GPUProbeAggregate@') && key.endsWith('.sumOverflow')
      )
    ).toBe(true);
    expect(Object.keys(findings).some(key => key.endsWith('.producerOverflow'))).toBe(false);
  } finally {
    rmSync(sourceRoot, {recursive: true, force: true});
  }
});

it('NAM-41 shared nondefault weighting settings retain packed code and metadata', () => {
  const context = createKernelAuditContext('shared-weight-settings');
  const {view} = context;
  try {
    const weighting = {weightType: 'inverse-distance'} as const;
    const settings = {bounds: [0, 0, 4, 4] as const, ...weighting};
    expect(getGPUNeighborSearchParameterValues(settings)[5]).toBe(1);
    expect(getGPUNeighborSearchWeightTransform(settings)).toBe('raw');
    const neighborWeights = new GPUNearestFeatureWeights({
      ...weighting,
      neighborRows: view('uint32', 1),
      neighborCounts: view('uint32', 1),
      neighborDistances: view('float32', 1),
      slotCapacity: 1,
      output: {
        offsets: view('uint32', 2),
        neighbors: view('uint32', 1),
        weights: view('float32', 1),
        status: {overflow: view('uint32', 1)}
      }
    });
    expect(neighborWeights.outputWeights.information?.transform).toBe('raw');
    const invalid = {...settings, weightType: 'javascript-typo'} as unknown as typeof settings;
    expect(() => getGPUNeighborSearchParameterValues(invalid)).toThrow(
      /weightType.*binary.*inverse-distance.*kernel/
    );
    expect(() => getGPUNeighborSearchWeightTransform(invalid)).toThrow(
      /weightType.*binary.*inverse-distance.*kernel/
    );
  } finally {
    context.device.destroy();
  }
});

it('NAM-41 spread settings require the requested compact output and reject invalid values', () => {
  const context = createKernelAuditContext('shared-join-settings');
  const {view} = context;
  try {
    const settings = {compactMatchedLeftRows: true};
    const ports = {
      leftKeys: view('uint32', 2),
      rightKeys: view('uint32', 2),
      output: {rightRows: view('uint32', 2)}
    };
    expect(() => new GPUKeyLookup({...ports, ...settings})).toThrow(
      /output.rows is required when compactMatchedLeftRows is true/
    );
    const join = new GPUKeyLookup({
      ...ports,
      ...settings,
      output: {
        ...ports.output,
        rows: {
          ids: view('uint32', 2),
          status: {count: view('uint32', 1), overflow: view('uint32', 1)}
        }
      }
    });
    expect(join.compactMatchedLeftRows).toBe(true);
    const gatherPorts = {
      pairs: {
        leftRows: view('uint32', 2),
        rightRows: view('uint32', 2),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      },
      output: {
        leftRows: view('uint32', 2),
        rightRows: view('uint32', 2),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      }
    };
    const gatherSettings = {how: 'left'} as const;
    expect(() => new GPUPairGather({...gatherPorts, ...gatherSettings})).toThrow(
      /how 'left' with pairs needs unmatchedLeft/
    );
    const gather = new GPUPairGather({
      ...gatherPorts,
      ...gatherSettings,
      unmatchedLeft: {
        ids: view('uint32', 2),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      }
    });
    expect(gather.how).toBe('left');
    const invalid = {compactMatchedLeftRows: 'javascript-typo'};
    expect(() => new GPUKeyLookup({...ports, ...(invalid as unknown as typeof settings)})).toThrow(
      /compactMatchedLeftRows must be a boolean.*javascript-typo/
    );
  } finally {
    context.device.destroy();
  }
});

it('NAM-41 numeric sum loss uses the canonical status views', () => {
  const context = createKernelAuditContext('sum-status-aliases');
  const {view} = context;
  try {
    const sumOverflow = view('uint32', 1);
    const grouped = new GPUGroupStatistics({
      keys: view('uint32', 2),
      columns: [
        {
          values: view('float32', 2),
          statistics: ['sum'],
          output: {sums: view('uint32x2', 2), status: {sumOverflow}}
        }
      ],
      output: {
        keys: view('uint32', 2),
        counts: view('uint32', 2),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      }
    });
    const column: GPUGroupStatisticsColumnOutput = grouped.props.columns[0].output;
    expect(column.status?.sumOverflow).toBe(sumOverflow);
    expect(column).not.toHaveProperty('sumOverflow');
    const joined = new GPUKeyLookup({
      leftKeys: view('uint32', 2),
      rightKeys: view('uint32', 2),
      aggregates: [
        {
          operation: 'sum',
          column: view('float32', 2),
          output: view('float32', 2),
          sums: view('uint32x2', 2),
          status: {sumOverflow}
        }
      ],
      output: {rightRows: view('uint32', 2)}
    });
    const aggregate = joined.props.aggregates![0];
    expect(aggregate.status?.sumOverflow).toBe(sumOverflow);
  } finally {
    context.device.destroy();
  }
});
