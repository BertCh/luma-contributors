// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, existsSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import typescript from 'typescript';
import GithubSlugger from 'github-slugger';
import {unified} from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import {describe, expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  addBuildingTessellationRecipe,
  addBuildingTessellationReferenceRecipe
} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation';
import {addLocalBuildingTessellationRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation-local';
import {GPUChunkedSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform';
import type {GPUPolygonOverlayOutput} from '../../../src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay';
import {queryGPUSpatialAnalysisOperators} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-operators';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  ANALYSIS_DOC_PAGES,
  createVirtualProgram,
  DOCS_ROOT,
  formatDiagnostic,
  getRepositoryCompilerOptions,
  REPOSITORY_ROOT
} from '../../analysis-core/doc-program';

import {
  describeOutputs,
  getGPUAnalysisOperatorCatalog,
  getGPUAnalysisOperator,
  queryGPUAnalysisOperators,
  searchGPUAnalysisTasks
} from '../../../src/gpu-spatial-analysis/ontology';

// Kept as a mutation-tested guide assertion, alongside the runtime forwarding fixture.
function getChunkedWeightsStatusIssues(guide: string): string[] {
  return guide.includes('`outputWeights.chunks[i].weights.status`')
    ? []
    : ['missing per-chunk status path'];
}

const GUIDE_PATH = join(DOCS_ROOT, 'analysis-catalog.md');
const CROSS_REFERENCE_PATH = join(DOCS_ROOT, 'gpu-spatial-analysis-cross-reference.md');

function getSections(
  text: string
): {heading: string; anchor: string; text: string; recording: boolean; start: number}[] {
  const sections: {heading: string; anchor: string; start: number; depth: number}[] = [];
  const lines = text.split('\n');
  const slugger = new GithubSlugger();
  let fenced = false;
  for (const [start, line] of lines.entries()) {
    if (/^\s*```/.test(line)) fenced = !fenced;
    if (!fenced && /^#{1,6} /.test(line)) {
      const heading = line.replace(/^#+ /, '');
      const depth = line.match(/^#+/)![0].length;
      sections.push({
        heading,
        anchor: slugger.slug(
          heading
            .replace(/`/g, '')
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
            .replace(/<[^>]+>/g, '')
        ),
        start,
        depth
      });
    }
  }
  return sections.map((section, index) => ({
    heading: section.heading,
    anchor: section.anchor,
    start: section.start,
    recording:
      /(?:executable|recorded|how it computes|computation lenses)/i.test(section.heading) ||
      sections
        .slice(0, index)
        .some(
          (parent, parentIndex) =>
            parent.depth < section.depth &&
            /(?:executable|recorded|how it computes|computation lenses)/i.test(parent.heading) &&
            !sections.slice(parentIndex + 1, index).some(next => next.depth <= parent.depth)
        ),
    text: lines
      .slice(
        section.start,
        sections.slice(index + 1).find(next => next.depth <= section.depth)?.start ?? lines.length
      )
      .join('\n')
  }));
}

/** Follow named and star re-exports, preserving the public entry and export spelling. */
function getPublicClasses(file: string, cache = new Map<string, Set<string>>()): Set<string> {
  file = resolve(file);
  const cached = cache.get(file);
  if (cached) return cached;
  const names = new Set<string>();
  cache.set(file, names);
  if (!existsSync(file)) return names;
  const source = typescript.createSourceFile(
    file,
    readFileSync(file, 'utf8'),
    typescript.ScriptTarget.Latest,
    true
  );
  for (const statement of source.statements) {
    if (
      typescript.isClassDeclaration(statement) &&
      statement.name &&
      statement.modifiers?.some(modifier => modifier.kind === typescript.SyntaxKind.ExportKeyword)
    )
      names.add(statement.name.text);
    if (
      !typescript.isExportDeclaration(statement) ||
      !statement.moduleSpecifier ||
      !typescript.isStringLiteral(statement.moduleSpecifier) ||
      statement.isTypeOnly
    )
      continue;
    const base = resolve(dirname(file), statement.moduleSpecifier.text);
    const children = getPublicClasses(
      existsSync(`${base}.ts`) ? `${base}.ts` : join(base, 'index.ts'),
      cache
    );
    if (!statement.exportClause) for (const name of children) names.add(name);
    else if (typescript.isNamedExports(statement.exportClause))
      for (const element of statement.exportClause.elements) {
        if (!element.isTypeOnly && children.has((element.propertyName ?? element.name).text))
          names.add(element.name.text);
      }
  }
  return names;
}

type Reference = {entry: string; names: string[]; target: string};

function getReferences(guide: string): Reference[] {
  return [...guide.matchAll(/^\| `([^`]+)` \| (.+?) \| \[[^\]]+\]\(([^)]+)\) \|(?:[^\n]*)$/gm)].map(
    match => ({
      entry: match[1],
      names: [...match[2].matchAll(/`(\w+)`/g)].map(name => name[1]),
      target: match[3]
    })
  );
}

function getContractSection(
  text: string,
  anchor?: string
): ReturnType<typeof getSections>[number] | undefined {
  const sections = getSections(text);
  if (!anchor) return sections[0];
  const heading = sections.find(section => section.anchor === anchor);
  if (heading) return heading;
  const marker = `<a id="${anchor}"></a>`;
  if (!text.split('\n').some(line => line.trim() === marker)) return undefined;
  // An alias immediately before a heading targets the following contract, not the previous one.
  const lines = text.split('\n');
  let followingLine = lines.findIndex(line => line.trim() === marker) + 1;
  while (followingLine < lines.length && !lines[followingLine].trim()) followingLine++;
  const following = sections.find(section => section.start === followingLine);
  if (following) return {...following, anchor};
  const containing = sections
    .filter(section => section.text.includes(marker))
    .sort((first, second) => first.text.length - second.text.length)[0];
  return containing
    ? {...containing, anchor, text: containing.text.slice(containing.text.indexOf(marker))}
    : undefined;
}

function getTargetSection(target: string, guide: string): string {
  const [file, anchor] = target.split('#');
  const text = file ? readFileSync(resolve(DOCS_ROOT, file), 'utf8') : guide;
  const section = getContractSection(text, anchor);
  expect(section, `missing reference section: ${target}`).toBeDefined();
  return section?.text ?? '';
}

type OwnedHeading = {entry: string | undefined; heading: string; target?: string};

/** The entry that owns a page, rather than the first same-name heading anywhere in the docs. */
function getDocumentEntry(page: string): string | undefined {
  const family = page.split('/')[0];
  if (
    [
      'analysis-core',
      'dataframe-analysis',
      'gpu-spatial-analysis',
      'gpu-network',
      'gpu-terrain'
    ].includes(family)
  )
    return family;
  const entries: Record<string, string> = {
    'gpu-raster': 'raster-analysis',
    'gpu-dataframe-analysis.md': 'dataframe-analysis',
    'forest-inference.md': 'dataframe-analysis',
    'variable-standardization.md': 'dataframe-analysis',
    'pointcloud-analysis.md': 'pointcloud-analysis',
    'geospatial.md': 'geospatial-analysis',
    'computation-record.md': 'computation-record',
    'operation-forms-and-cpu-twins.md': 'analysis-core',
    'gpu-geometry-refinement.md': 'gpu-spatial-analysis'
  };
  return entries[family];
}

function hasOwnedHeading(entry: string, name: string, headings: readonly OwnedHeading[]): boolean {
  // The package root re-exports recording infrastructure; ontology aliases the spatial entry.
  const owner =
    entry === '.'
      ? 'computation-record'
      : entry === 'gpu-spatial-analysis/ontology'
        ? 'gpu-spatial-analysis'
        : entry;
  return headings.some(
    section =>
      section.entry === owner &&
      new RegExp(`\\b${name}\\b`).test(section.heading) &&
      (!section.target || hasResolvingContract(section.target, [name], ''))
  );
}

const INFRASTRUCTURE_CLASSES = new Set([
  'GPUParameterBuffer',
  'GPUContributorOperation',
  'GPURegionStatisticsReadback',
  'AnalysisGeographicExtentAccumulator',
  'FlowFieldViewportTracker',
  'AnalysisValidationError'
]);

const GROUPED_CONTRACT_EXPECTATIONS: Readonly<Record<string, {target: string; phrase: string}>> = {
  'gpu-spatial-analysis:GPUSphericalConvexContains': {
    target:
      './gpu-spatial-analysis/global-coordinates.md#convex-spherical-geometry-and-overlap-weights',
    phrase: 'Containment uses a query-to-polygon index column.'
  },
  'gpu-spatial-analysis:GPUSphericalConvexIntersection': {
    target:
      './gpu-spatial-analysis/global-coordinates.md#convex-spherical-geometry-and-overlap-weights',
    phrase: 'Intersection uses paired polygon-offset columns.'
  }
};

const publicClassCache = new Map<string, Set<string>>();
const publicFunctionCache = new Map<string, Map<string, {file: string; line: number}>>();

/** A recording caption cannot certify the operation contract. Explicit local contract links
 * resolve recursively to prose carrying the qualified producer/form identity. */
function hasResolvingContract(
  target: string,
  names: string[],
  guide: string,
  visited = new Set<string>()
): boolean {
  if (visited.has(target) || visited.size >= 8) return false;
  visited.add(target);
  const [file, anchor] = target.split('#');
  const text = file ? readFileSync(resolve(DOCS_ROOT, file), 'utf8') : guide;
  const section = getContractSection(text, anchor);
  if (!section || /parameters$/i.test(section.heading)) return false;
  const prose = getSubstantiveContract(section.text);
  const identityText = section.heading + ' ' + prose;
  if (
    !section.recording &&
    !file?.endsWith('-lenses.md') &&
    prose &&
    names.some(name => new RegExp('\\b' + name + '\\b').test(identityText))
  )
    return true;
  for (const match of section.text.matchAll(/\[([^\]]+)\]\(([^)]+)\)/g)) {
    const link = match[2];
    if (/^(?:[a-z]+:|\/)/i.test(link) || !/(?:\.md(?:#|$)|^#)/.test(link)) continue;
    const [linkedFile, linkedAnchor] = link.split('#');
    const resolvedFile = linkedFile
      ? join(dirname(file || 'analysis-catalog.md'), linkedFile)
      : file;
    const resolvedTarget =
      (resolvedFile ? './' + resolvedFile.replace(/^\.\//, '') : '') +
      (linkedAnchor ? '#' + linkedAnchor : '');
    if (hasResolvingContract(resolvedTarget, names, guide, new Set(visited))) return true;
  }
  return false;
}

function getReferenceTargetIssues(reference: Reference, guide: string): string[] {
  const [file, anchor] = reference.target.split('#');
  const targetText = file ? readFileSync(resolve(DOCS_ROOT, file), 'utf8') : guide;
  const section = getContractSection(targetText, anchor);
  if (!section) return [`missing section: ${reference.target}`];
  if (/parameters$/i.test(section.heading))
    return [`generated Parameters section: ${reference.target}`];
  const isGuide = !file || resolve(DOCS_ROOT, file) === GUIDE_PATH;
  return reference.names.flatMap(name => {
    const entryFile = join(
      REPOSITORY_ROOT,
      'modules/analysis/src',
      reference.entry === '.' ? '' : reference.entry,
      'index.ts'
    );
    if (
      !getPublicClasses(entryFile, publicClassCache).has(name) &&
      !getPublicFunctions(entryFile, publicFunctionCache).has(name)
    )
      return [`wrong public entry: ${reference.entry}:${name}`];
    const isInfrastructureSection =
      INFRASTRUCTURE_CLASSES.has(name) && anchor === name.toLowerCase();
    if (isGuide && !isInfrastructureSection) return [`index-only destination: ${name}`];
    const grouped = GROUPED_CONTRACT_EXPECTATIONS[`${reference.entry}:${name}`];
    if (grouped)
      return reference.target === grouped.target &&
        getSubstantiveContract(section.text).includes(grouped.phrase) &&
        !section.recording
        ? []
        : [`wrong grouped contract: ${name}`];
    const preferred = PREFERRED_CONTRACT_TARGETS[`${reference.entry}:${name}`];
    if (preferred && reference.target !== preferred) return [`wrong contract destination: ${name}`];
    const operator = getGPUAnalysisOperatorCatalog().find(
      operator =>
        operator.package === '@luma-contributors/analysis/' + reference.entry &&
        [operator.forms?.cpu?.exportName, operator.forms?.gpu?.exportName].includes(name)
    );
    return hasResolvingContract(reference.target, [name, operator?.exportName ?? name], guide)
      ? []
      : [`missing prose contract or resolving link: ${name}`];
  });
}

function getCooperativeTimingIssues(row: string): string[] {
  // Literal comparisons from the separate October 24 refinement record; keep the warm and
  // changed-input GPU figures attached to their own measurement boundaries.
  return [
    ...(row.includes('fit at 0.7 ms warm GPU execution') ? [] : ['warm comparison']),
    ...(row.includes(
      'changed-input GPU execution was 0.7 ms versus a 2.3–2.5 ms CPU round trip including transfers'
    )
      ? []
      : ['changed-input comparison'])
  ];
}

const EXISTING_OPERATION_REFERENCES = [
  [
    'gpu-spatial-analysis',
    'GPUPopulationSupportAlignmentOperation',
    './gpu-spatial-analysis/landscape-ecology.md#gpupopulationsupportalignmentoperation'
  ],
  [
    'gpu-spatial-analysis',
    'CPUPopulationSupportAlignmentOperation',
    './gpu-spatial-analysis/landscape-ecology.md#cpupopulationsupportalignmentoperation'
  ],
  [
    'gpu-network',
    'GPUKShortestPathsOperation',
    './gpu-network/k-shortest-paths.md#gpukshortestpathsoperation'
  ],
  [
    'gpu-network',
    'CPUKShortestPathsOperation',
    './gpu-network/k-shortest-paths.md#cpukshortestpathsoperation-and-computekshortestpathsoncpu'
  ],
  [
    'dataframe-analysis',
    'GPUCoregionalizationFitOperation',
    './dataframe-analysis/coregionalization-fit.md#gpucoregionalizationfitoperation'
  ],
  [
    'dataframe-analysis',
    'CPUCoregionalizationFitOperation',
    './dataframe-analysis/coregionalization-fit.md#cpucoregionalizationfitoperation-and-supplied-kriging-model'
  ]
] as const;

// Independent, entry-qualified expectations: these are the public fitting producers and the
// specific sections that document their contracts, rather than a count or a packer mention.
const FITTING_REFERENCES = [
  ['GPUFittingCoordinateEncoder', './analysis-core/fitting-kernels.md#gpufittingcoordinateencoder'],
  ['GPUFittingGradientSums', './analysis-core/fitting-kernels.md#gpufittinggradientsums'],
  ['GPUFittingMinibatchOrder', './analysis-core/fitting-kernels.md#gpufittingminibatchorder'],
  ['GPUAdamStep', './analysis-core/fitting-kernels.md#gpuadamstep']
] as const;

function getFittingReferenceIssues(guide: string): string[] {
  const references = getReferences(guide);
  const introduction = guide.slice(
    guide.indexOf('## The operator catalog'),
    guide.indexOf('Each operator has')
  );
  return FITTING_REFERENCES.flatMap(([name, target]) =>
    introduction.includes(`[\`${name}\`](${target})`) &&
    references.some(
      reference =>
        reference.entry === 'analysis-core' &&
        reference.names.includes(name) &&
        reference.target === target
    )
      ? []
      : [name]
  );
}

// DST-10: independently selected prose contracts from the 32 reported marker-only anchors.
// The last four records also pin DST-14's seven misleading links across the two lookup pages.
const PREFERRED_CONTRACT_TARGETS: Readonly<Record<string, string>> = {
  'gpu-network:GPUNetworkInhomogeneousK':
    './gpu-spatial-analysis/point-process.md#inhomogeneous-network-k',
  'raster-analysis:GPURasterFocalPairs': './gpu-raster/focal-pairs.md#focal-cross-band-statistics',
  'raster-analysis:GPURasterConservativeRemapping':
    './gpu-raster/remote-sensing.md#conservative-remapping-rs03',
  'raster-analysis:GPURasterTexture': './gpu-raster/remote-sensing.md#glcm-texture-rs04',
  'raster-analysis:GPURasterTemporal': './gpu-raster/remote-sensing.md#temporal-stacks-rs06',
  'raster-analysis:GPURasterRegistration':
    './gpu-raster/remote-sensing.md#translation-registration-rs07',
  'raster-analysis:GPURasterSpectralUnmixing':
    './gpu-raster/remote-sensing.md#spectral-unmixing-rs10',
  'raster-analysis:GPURasterSeasonalBreak':
    './gpu-raster/remote-sensing.md#seasonal-decomposition-breaks-and-monitoring-rs11',
  'raster-analysis:GPURasterNDWI': './gpu-raster/spectral-indices.md#formula-and-band-contracts',
  'raster-analysis:GPURasterNBR': './gpu-raster/spectral-indices.md#formula-and-band-contracts',
  'raster-analysis:GPURasterEVI': './gpu-raster/spectral-indices.md#formula-and-band-contracts',
  'raster-analysis:GPURasterSAVI': './gpu-raster/spectral-indices.md#formula-and-band-contracts',
  'gpu-spatial-analysis:GPUInstrumentalVariablesCovariance':
    './gpu-spatial-analysis/coverage-statistics.md#iv-covariance-and-robust-spatial-2sls',
  'gpu-spatial-analysis:GPURestrictedRegimeRegression':
    './gpu-spatial-analysis/coverage-statistics.md#regimes-panels-and-sur',
  'gpu-spatial-analysis:GPUProbitRegression':
    './gpu-spatial-analysis/coverage-statistics.md#probit-and-local-collinearity',
  'gpu-spatial-analysis:GPULocalCollinearity':
    './gpu-spatial-analysis/coverage-statistics.md#probit-and-local-collinearity',
  'gpu-spatial-analysis:GPUFuzzyContiguity':
    './gpu-spatial-analysis/coverage-statistics.md#weights-and-spatial-evaluation',
  'gpu-spatial-analysis:GPUIslandAttachment':
    './gpu-spatial-analysis/coverage-statistics.md#weights-and-spatial-evaluation',
  'gpu-spatial-analysis:GPUSpatialCrossValidation':
    './gpu-spatial-analysis/coverage-statistics.md#weights-and-spatial-evaluation',
  'gpu-spatial-analysis:GPUSpatialCrossValidatedOLS':
    './gpu-spatial-analysis/coverage-statistics.md#weights-and-spatial-evaluation',
  'gpu-spatial-analysis:GPUS2PolygonCovering':
    './gpu-spatial-analysis/global-coordinates.md#adaptive-polygon-covers',
  'gpu-spatial-analysis:GPUSphericalPolygonContains':
    './gpu-spatial-analysis/global-coordinates.md#general-spherical-topology',
  'gpu-spatial-analysis:GPUSphericalPolygonOverlay':
    './gpu-spatial-analysis/global-coordinates.md#general-spherical-topology',
  'gpu-spatial-analysis:GPUSphericalArcOverlay':
    './gpu-spatial-analysis/global-coordinates.md#general-spherical-topology',
  'gpu-spatial-analysis:GPUInhomogeneousPointPattern':
    './gpu-spatial-analysis/point-process.md#intensity-and-polygon-corrections',
  'gpu-spatial-analysis:GPUPointPatternEnvelope':
    './gpu-spatial-analysis/point-process.md#simulation-curves-and-global-tests',
  'gpu-spatial-analysis:GPUSpatialBalancedSample':
    './gpu-spatial-analysis/point-process.md#spatial-probability-sampling',
  'gpu-spatial-analysis:GPUOPTICS': './gpu-spatial-analysis/point-process.md#optics-ordering',
  'gpu-terrain:GPUTerrainFlowFractions': './gpu-terrain/terrain-processing.md#branching-consumers',
  'gpu-terrain:GPUTerrainDepressionHierarchy':
    './gpu-terrain/terrain-processing.md#depression-hierarchy-and-stage-curves',
  'gpu-terrain:GPUTerrainCatchmentSummary':
    './gpu-terrain/terrain-processing.md#basin-summaries-and-profiles',
  'gpu-terrain:GPUTerrainDEMCoregistration': './gpu-terrain/terrain-processing.md#dem-comparison',
  'gpu-spatial-analysis:GPULocalGeary':
    './gpu-spatial-analysis/operators-statistics.md#gpulocalgeary',
  'gpu-spatial-analysis:GPUSpatialPearson':
    './gpu-spatial-analysis/operators-statistics.md#gpuspatialpearson-and-gpugammastatistic',
  'gpu-spatial-analysis:GPUGammaStatistic':
    './gpu-spatial-analysis/operators-statistics.md#gpuspatialpearson-and-gpugammastatistic',
  'gpu-spatial-analysis:GPUNearestFeatureWeights':
    './gpu-spatial-analysis/operators-joins.md#gpunearestfeatureweights',
  'raster-analysis:GPURasterFillHoles': './gpu-raster/expansion.md#operations-and-contracts',
  'raster-analysis:GPURasterFractionalZonalDistribution':
    './gpu-raster/fractional-zonal-distribution.md#gpurasterfractionalzonaldistribution',
  'gpu-spatial-analysis:GPUSpatialWeightsAlgebra':
    './gpu-spatial-analysis/operators-weights.md#gpuspatialweightsalgebra'
};

function getSubstantiveContract(text: string): string {
  const source = text
    .replace(/^#{1,6} .*$/gm, '')
    .replace(/\{\/\*[\s\S]*?\*\/\}/g, '')
    .replace(/<!--[^]*?-->/g, '')
    .replace(/::lens[^\n]*/g, '')
    .replace(/\{\/\* parameters:[^]*?\*\/\}/g, '')
    .trim();
  type MarkdownNode = {type: string; value?: string; children?: MarkdownNode[]};
  const tree = unified().use(remarkParse).use(remarkGfm).parse(source) as MarkdownNode;
  const prose: string[] = [];
  const visit = (node: MarkdownNode): void => {
    // Headings, source examples and author markup alone cannot establish a prose contract.
    if (['heading', 'code', 'html', 'link', 'linkReference'].includes(node.type)) return;
    if (node.type === 'text' || node.type === 'inlineCode') prose.push(node.value ?? '');
    node.children?.forEach(visit);
  };
  visit(tree);
  return prose.join(' ').replace(/\s+/g, ' ').trim();
}

/** Independent spatial barrel inventory, without importing the catalog being checked. */
function getSpatialBrowseExports(
  file: string,
  cache = new Map<string, Set<string>>()
): Set<string> {
  file = resolve(file);
  const cached = cache.get(file);
  if (cached) return cached;
  const names = new Set<string>();
  cache.set(file, names);
  if (!existsSync(file)) return names;
  const source = typescript.createSourceFile(
    file,
    readFileSync(file, 'utf8'),
    typescript.ScriptTarget.Latest,
    true
  );
  for (const statement of source.statements) {
    if (
      typescript.canHaveModifiers(statement) &&
      typescript
        .getModifiers(statement)
        ?.some(modifier => modifier.kind === typescript.SyntaxKind.ExportKeyword)
    ) {
      if (
        typescript.isClassDeclaration(statement) &&
        statement.name &&
        statement.members.some(member => member.name?.getText(source) === 'getCommandNodes')
      )
        names.add(statement.name.text);
      if (
        typescript.isFunctionDeclaration(statement) &&
        statement.name &&
        /^add\w+Recipe$/.test(statement.name.text)
      )
        names.add(statement.name.text);
    }
    if (
      !typescript.isExportDeclaration(statement) ||
      !statement.moduleSpecifier ||
      !typescript.isStringLiteral(statement.moduleSpecifier) ||
      statement.isTypeOnly
    )
      continue;
    const base = resolve(dirname(file), statement.moduleSpecifier.text);
    const children = getSpatialBrowseExports(
      existsSync(base + '.ts') ? base + '.ts' : join(base, 'index.ts'),
      cache
    );
    if (!statement.exportClause) for (const name of children) names.add(name);
    else if (typescript.isNamedExports(statement.exportClause))
      for (const element of statement.exportClause.elements) {
        if (!element.isTypeOnly && children.has((element.propertyName ?? element.name).text))
          names.add(element.name.text);
      }
  }
  return names;
}

type NavigationItem = string | {type: string; label: string; items: NavigationItem[]};
function getNavigationPages(items: NavigationItem[]): string[] {
  return items.flatMap(item =>
    typeof item === 'string' ? [item] : getNavigationPages(item.items)
  );
}

/** Follow actual Markdown link nodes; fences, markers and sidebar entries do not create edges. */
function getReachablePages(start: string): Set<string> {
  const visited = new Set<string>();
  const pending = [start];
  const byRoute = new Map(
    ANALYSIS_DOC_PAGES.map(page => [page.replace(/\.md$/, '').replace(/(^|\/)README$/, ''), page])
  );
  type MarkdownNode = {type: string; url?: string; children?: MarkdownNode[]};
  while (pending.length) {
    const page = pending.pop()!;
    if (visited.has(page)) continue;
    visited.add(page);
    const tree = unified()
      .use(remarkParse)
      .use(remarkGfm)
      .parse(readFileSync(join(DOCS_ROOT, page), 'utf8')) as MarkdownNode;
    const visit = (node: MarkdownNode): void => {
      if (node.type === 'link' && node.url) {
        const target = node.url.split('#')[0].split('?')[0];
        let destination: string | undefined;
        if (target.startsWith('/docs/api'))
          destination = byRoute.get(target.replace(/^\/docs\/api\/?/, '').replace(/\/$/, ''));
        else if (target.endsWith('.md') && !/^(?:[a-z]+:|\/\/)/i.test(target)) {
          const absolute = resolve(dirname(join(DOCS_ROOT, page)), target);
          if (absolute.startsWith(`${DOCS_ROOT}/`))
            destination = absolute.slice(DOCS_ROOT.length + 1);
        }
        if (destination && ANALYSIS_DOC_PAGES.includes(destination) && !visited.has(destination))
          pending.push(destination);
      }
      node.children?.forEach(visit);
    };
    visit(tree);
  }
  return visited;
}

const ORIGINAL_CONTRACT_IDENTITIES = [
  'gpu-network:GPUNetworkInhomogeneousK',
  'raster-analysis:GPURasterFocalPairs',
  'raster-analysis:GPURasterConservativeRemapping',
  'raster-analysis:GPURasterTexture',
  'raster-analysis:GPURasterTemporal',
  'raster-analysis:GPURasterRegistration',
  'raster-analysis:GPURasterSpectralUnmixing',
  'raster-analysis:GPURasterSeasonalBreak',
  'raster-analysis:GPURasterNDWI',
  'raster-analysis:GPURasterNBR',
  'raster-analysis:GPURasterEVI',
  'raster-analysis:GPURasterSAVI',
  'gpu-spatial-analysis:GPUInstrumentalVariablesCovariance',
  'gpu-spatial-analysis:GPURestrictedRegimeRegression',
  'gpu-spatial-analysis:GPUProbitRegression',
  'gpu-spatial-analysis:GPULocalCollinearity',
  'gpu-spatial-analysis:GPUFuzzyContiguity',
  'gpu-spatial-analysis:GPUIslandAttachment',
  'gpu-spatial-analysis:GPUSpatialCrossValidation',
  'gpu-spatial-analysis:GPUSpatialCrossValidatedOLS',
  'gpu-spatial-analysis:GPUS2PolygonCovering',
  'gpu-spatial-analysis:GPUSphericalPolygonContains',
  'gpu-spatial-analysis:GPUSphericalPolygonOverlay',
  'gpu-spatial-analysis:GPUSphericalArcOverlay',
  'gpu-spatial-analysis:GPUInhomogeneousPointPattern',
  'gpu-spatial-analysis:GPUPointPatternEnvelope',
  'gpu-spatial-analysis:GPUSpatialBalancedSample',
  'gpu-spatial-analysis:GPUOPTICS',
  'gpu-terrain:GPUTerrainFlowFractions',
  'gpu-terrain:GPUTerrainDepressionHierarchy',
  'gpu-terrain:GPUTerrainCatchmentSummary',
  'gpu-terrain:GPUTerrainDEMCoregistration'
] as const;

const EDITORIAL_QUESTIONS: readonly (readonly [string, string])[] = [
  [
    'Which points fall in which polygon?',
    './gpu-spatial-analysis-cross-reference.md#all-matches-or-one-coverage-zone'
  ],
  ['Drive-time areas?', './gpu-network/operators.md#gpunetworkisochrones'],
  ['Hot spots over time?', './gpu-spatial-analysis/operators-statistics.md#gpuemerginghotspots'],
  [
    'Smooth rates for small populations?',
    './gpu-spatial-analysis/operators-statistics.md#gpuempiricalbayesrates'
  ],
  ['Viewshed?', './gpu-terrain/operators.md#gputerrainviewshed'],
  ['Fill holes in a raster?', './gpu-raster/expansion.md#operations-and-contracts'],
  ['Extract point-cloud ground?', './pointcloud-analysis.md'],
  ['Snap GPS traces to roads?', './gpu-network/operators.md#gpumapmatching'],
  ['Where has vegetation declined?', './gpu-raster/raster-change-monitoring.md'],
  ['How high above drainage?', './gpu-terrain/height-above-drainage.md'],
  ['Find an alternative route?', './gpu-network/alternative-routes.md'],
  [
    'Compare two tracks?',
    './gpu-spatial-analysis/track-similarity-measures.md#lower-bounds-and-pruned-search'
  ],
  [
    'Find a change point?',
    './gpu-spatial-analysis/change-points-and-expectation-based-scan.md#pettitt-test-gpupettitttest'
  ],
  ['Summarize land cover by polygon?', './gpu-raster/fractional-categorical-statistics.md'],
  ['Find similar pixels?', './gpu-raster/raster-embeddings.md']
];
function getEditorialIssues(guide: string): string[] {
  const section =
    getSections(guide).find(section => section.anchor === 'choose-by-question')?.text ?? '';
  return EDITORIAL_QUESTIONS.flatMap(([question, target]) => {
    const row = section.split('\n').find(line => line.startsWith('| ' + question + ' |'));
    return row?.includes('[Read contract](' + target + ')') ? [] : [question];
  });
}

/** Public functions independently resolved from the entry's AST and re-export edges. */
function getPublicFunctions(
  file: string,
  cache = new Map<string, Map<string, {file: string; line: number}>>()
): Map<string, {file: string; line: number}> {
  file = resolve(file);
  const cached = cache.get(file);
  if (cached) return cached;
  const names = new Map<string, {file: string; line: number}>();
  cache.set(file, names);
  if (!existsSync(file)) return names;
  const source = typescript.createSourceFile(
    file,
    readFileSync(file, 'utf8'),
    typescript.ScriptTarget.Latest,
    true
  );
  for (const statement of source.statements) {
    if (
      typescript.isFunctionDeclaration(statement) &&
      statement.name &&
      statement.modifiers?.some(modifier => modifier.kind === typescript.SyntaxKind.ExportKeyword)
    )
      names.set(statement.name.text, {
        file,
        line: source.getLineAndCharacterOfPosition(statement.getStart()).line + 1
      });
    if (
      !typescript.isExportDeclaration(statement) ||
      !statement.moduleSpecifier ||
      !typescript.isStringLiteral(statement.moduleSpecifier) ||
      statement.isTypeOnly
    )
      continue;
    const base = resolve(dirname(file), statement.moduleSpecifier.text);
    const children = getPublicFunctions(
      existsSync(base + '.ts') ? base + '.ts' : join(base, 'index.ts'),
      cache
    );
    if (!statement.exportClause) for (const [name, value] of children) names.set(name, value);
    else if (typescript.isNamedExports(statement.exportClause))
      for (const element of statement.exportClause.elements) {
        const value = children.get((element.propertyName ?? element.name).text);
        if (!element.isTypeOnly && value) names.set(element.name.text, value);
      }
  }
  return names;
}

function getHelperAvailabilityIssues(section: string, available: ReadonlySet<string>): string[] {
  const names = ['describeOutputs', 'searchGPUAnalysisTasks'];
  const imports = new Set<string>();
  const calls = new Set<string>();
  const unavailable = new Set<string>();
  type MarkdownNode = {
    type: string;
    value?: string;
    children?: MarkdownNode[];
    lang?: string | null;
  };
  const plainText = (node: MarkdownNode): string =>
    node.value ?? node.children?.map(plainText).join('') ?? '';
  const visit = (node: MarkdownNode): void => {
    if (node.type === 'code' && node.lang === 'ts') {
      const source = typescript.createSourceFile(
        'example.ts',
        node.value ?? '',
        typescript.ScriptTarget.Latest,
        true
      );
      const visitCode = (statement: typescript.Node): void => {
        if (
          typescript.isImportDeclaration(statement) &&
          typescript.isStringLiteral(statement.moduleSpecifier) &&
          statement.moduleSpecifier.text ===
            '@luma-contributors/analysis/gpu-spatial-analysis/ontology'
        ) {
          const bindings = statement.importClause?.namedBindings;
          if (bindings && typescript.isNamedImports(bindings))
            for (const element of bindings.elements) imports.add(element.name.text);
        }
        if (typescript.isCallExpression(statement) && typescript.isIdentifier(statement.expression))
          calls.add(statement.expression.text);
        typescript.forEachChild(statement, visitCode);
      };
      visitCode(source);
    }
    if (node.type === 'paragraph') {
      const paragraphUnavailable = new Set<string>();
      const clauses = plainText(node).split(/[.;!?]\s+|;\s*|\s+(?:but|while|whereas)\s+/);
      for (const clause of clauses) {
        const negative =
          /unavailable|not (?:yet )?(?:exported|available|callable)|pending (?:public exports|API)/i;
        const markUnavailable = (name: string): void => {
          unavailable.add(name);
          paragraphUnavailable.add(name);
        };
        if (!negative.test(clause)) continue;
        if (/both helpers/i.test(clause)) {
          names.forEach(markUnavailable);
          continue;
        }
        const mentions = [...clause.matchAll(/\b(?:describeOutputs|searchGPUAnalysisTasks)\b/g)];
        // A predicate shared by two subjects applies to both; separate predicates remain local.
        const pendingSubjects: string[] = [];
        const prefix = clause.slice(0, mentions[0]?.index ?? clause.length);
        if (negative.test(prefix)) mentions.forEach(mention => markUnavailable(mention[0]));
        for (const [index, mention] of mentions.entries()) {
          pendingSubjects.push(mention[0]);
          const predicate = clause.slice(
            mention.index + mention[0].length,
            mentions[index + 1]?.index
          );
          if (negative.test(predicate)) {
            pendingSubjects.forEach(markUnavailable);
            pendingSubjects.length = 0;
          } else if (/\b(?:available|exported|callable)\b/i.test(predicate)) {
            pendingSubjects.length = 0;
          }
        }
      }
      const visitInline = (inline: MarkdownNode): void => {
        if (inline.type === 'inlineCode')
          for (const name of names)
            if (
              new RegExp('\\b' + name + '\\s*\\(').test(inline.value ?? '') &&
              !paragraphUnavailable.has(name)
            )
              calls.add(name);
        inline.children?.forEach(visitInline);
      };
      visitInline(node);
    }
    node.children?.forEach(visit);
  };
  visit(unified().use(remarkParse).parse(section));
  return names.flatMap(name => {
    if (available.has(name))
      return unavailable.has(name) || !imports.has(name) || !calls.has(name)
        ? [`${name}: stale or missing public example`]
        : [];
    return imports.has(name) || calls.has(name)
      ? [`${name}: unavailable helper advertised as callable`]
      : [];
  });
}

function getArraySourceIssues(guide: string): string[] {
  const issues: string[] = [];
  const cache = new Map<string, Map<string, {file: string; line: number}>>();
  for (const operator of getGPUAnalysisOperatorCatalog()) {
    if (!operator.forms?.cpu || !operator.forms.gpu) continue;
    for (const form of [operator.forms.cpu, operator.forms.gpu]) {
      const entry = form.package.replace('@luma-contributors/analysis/', '');
      const row = guide
        .split('\n')
        .find(
          line =>
            line.startsWith(
              '| ' + String.fromCharCode(96) + entry + String.fromCharCode(96) + ' |'
            ) && line.includes(String.fromCharCode(96) + form.exportName + String.fromCharCode(96))
        );
      for (const array of operator.forms.arrays ?? []) {
        const declaration = getPublicFunctions(
          join(
            REPOSITORY_ROOT,
            'modules/analysis/src',
            array.package.replace('@luma-contributors/analysis/', ''),
            'index.ts'
          ),
          cache
        ).get(array.exportName);
        const link =
          row &&
          [...row.matchAll(/\[\x60(\w+)\x60\]\(([^)]+)\)/g)].find(
            match => match[1] === array.exportName
          )?.[2];
        if (!link || !declaration) {
          issues.push(
            form.package +
              ':' +
              form.exportName +
              ':' +
              array.exportName +
              ' missing qualified declaration/link'
          );
          continue;
        }
        const expected =
          'https://github.com/visgl/luma-contributors/blob/master/' +
          declaration.file.slice(REPOSITORY_ROOT.length + 1);
        if (link !== expected && link !== expected + '#L' + declaration.line)
          issues.push(array.package + ':' + array.exportName + ' wrong source target: ' + link);
      }
    }
  }
  return issues;
}

describe('catalog reference targets', () => {
  it.each([
    [
      'GPUSpatialInteraction',
      {
        name: 'auto-dense-pair-crossover',
        kind: 'auto-crossover',
        quantity: 'pairCount',
        unit: 'rows',
        maximum: 4096,
        scope: 'domain',
        appliesWhen: 'method auto, constraint doubly, calibration grid'
      }
    ],
    [
      'GPUGeographicDistribution',
      {
        name: 'auto-chunked-crossover',
        kind: 'auto-crossover',
        quantity: 'groupCount',
        unit: 'elements',
        maximum: 64,
        scope: 'domain',
        appliesWhen: 'reduction auto'
      }
    ],
    [
      'GPUSpatialPredicateJoin',
      {
        name: 'relate-auto-crossover',
        kind: 'auto-crossover',
        minimum: 1024,
        scope: 'domain',
        appliesWhen: 'engine auto; intersects/contains/within; non-point singlepart operands'
      }
    ],
    [
      'GPUCellPyramid',
      {
        name: 'level-count-slots',
        kind: 'allocation',
        unit: 'bytes',
        scope: 'binding',
        byteExpression:
          'resolutions.length * 64 * 4 <= min(device.maxBufferSize, device.maxStorageBufferBindingSize)',
        appliesWhen: 'output.levelCounts is requested'
      }
    ]
  ] as const)('R1 B1 publishes qualified crossover and allocation descriptors for %s', (exportName, expected) => {
    const operator = getGPUAnalysisOperator({
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName
    });
    const record = operator?.routes?.find(route => route.name === expected.name);
    const {appliesWhen, ...properties} = expected;
    expect(record).toMatchObject(properties);
    expect(record?.appliesWhen).toContain(appliesWhen);
    const plannerRoute = {
      GPUSpatialInteraction: 'auto-dense-pair-crossover',
      GPUGeographicDistribution: 'auto-chunked-crossover',
      GPUSpatialPredicateJoin: 'relate-auto-crossover',
      GPUCellPyramid: 'input rows'
    }[exportName];
    expect(operator?.routes?.find(route => route.name === plannerRoute)?.planner).toEqual({
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName:
        exportName === 'GPUSpatialPredicateJoin' ? 'GPU_SPATIAL_PREDICATE_JOIN_PLANNER' : exportName
    });
  });

  it('R1 B2 rejects unavailable prose and inline calls independently for each helper', () => {
    const section =
      getSections(readFileSync(GUIDE_PATH, 'utf8')).find(
        section => section.anchor === 'forms-task-search-and-output-plans'
      )?.text ?? '';
    const both = new Set(['describeOutputs', 'searchGPUAnalysisTasks']);
    expect(
      getHelperAvailabilityIssues(
        section + '\nBoth helpers are unavailable in the public ontology entry.',
        both
      )
    ).not.toEqual([]);
    expect(
      getHelperAvailabilityIssues(
        'Call `describeOutputs(identity, mode, sizes)` from the public ontology entry.',
        new Set()
      )
    ).toContain('describeOutputs: unavailable helper advertised as callable');
    const mixed =
      "```ts\nimport {describeOutputs} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';\ndescribeOutputs(identity, mode, sizes);\n```\n\n`describeOutputs` is available; `searchGPUAnalysisTasks` is unavailable in the public ontology entry.";
    expect(getHelperAvailabilityIssues(mixed, new Set(['describeOutputs']))).toEqual([]);
    expect(
      getHelperAvailabilityIssues(
        mixed.replace('available; ', 'available and '),
        new Set(['describeOutputs'])
      )
    ).toEqual([]);
    expect(
      getHelperAvailabilityIssues(
        mixed.replace('searchGPUAnalysisTasks` is unavailable', 'describeOutputs` is unavailable'),
        new Set(['describeOutputs'])
      )
    ).not.toEqual([]);
  });

  it('R1 B3 explicit aliases retain the recording ancestry of their target heading', () => {
    const caption =
      '## Recorded examples\n\n<a id="example-alias"></a>\n\n### GPUExample\n\nGPUExample writes the recorded output rows.\n';
    expect(hasResolvingContract('#gpuexample', ['GPUExample'], caption)).toBe(false);
    expect(hasResolvingContract('#example-alias', ['GPUExample'], caption)).toBe(false);
    expect(getContractSection(caption, 'example-alias')?.recording).toBe(true);
  });

  it('review unfinished output plans distinguish unchecked sizes and retained-piece requirements', () => {
    const identity = {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPULineChunk'
    } as const;
    const partial = describeOutputs(identity, 'chunk', {lineCapacity: 5, vertexCapacity: 20});
    expect(partial.constraints).toEqual([
      {expression: 'lineCount * (lineCapacity + 1)', checked: false},
      {expression: '2 * lineCapacity + positionCount', checked: false}
    ]);
    const full = describeOutputs(identity, 'chunk', {
      lineCapacity: 5,
      vertexCapacity: 20,
      lineCount: 2,
      positionCount: 8
    });
    expect(full.constraints?.map(constraint => constraint.checked)).toEqual([true, true]);
    // Hand arithmetic: 2*(5+1)=12 and 2*5+8=18; neither is an output-content proof.
    expect(full.outputs.find(output => output.path === 'output.sourceLineIndices')?.tail).toBe(
      '0xffffffff past output.status.count'
    );
    expect(
      full.outputs.find(output => output.path === 'output.status.requiredVertexCount')?.meaning
    ).toContain('retained pieces only');
    const ols = describeOutputs(
      {package: identity.package, exportName: 'GPUOrdinaryLeastSquares'},
      'default',
      {predictorCount: 2, rowCount: 100}
    );
    // Two slopes plus the intercept give three readable coefficients, independently of reservation.
    expect(ols.outputs.find(output => output.path === 'output.coefficients')).toMatchObject({
      capacity: 16,
      minimumCapacity: 3,
      activeLength: 3
    });
    expect(ols.outputs.find(output => output.path === 'output.summary')?.activeLength).toBe(22);
    expect(
      ols.outputs.find(output => output.path === 'output.predictions')?.activeLengthRule
    ).toContain('finite predictor rows including masked rows or missing responses');
    for (const [name, route, maximum] of [
      ['GPUSpatialInteraction', 'auto-dense-pair-crossover', 4096],
      ['GPUGeographicDistribution', 'auto-chunked-crossover', 64],
      ['GPUNeighborSearch', 'knn', 256],
      ['GPUExpectationBasedSpaceTimeScan', 'neighborhood-sort', 64]
    ] as const) {
      const record = getGPUAnalysisOperator({
        package: identity.package,
        exportName: name
      })?.routes?.find(candidate => candidate.name === route);
      expect(record?.maximum).toBe(maximum);
      expect(record?.planner).toEqual({package: identity.package, exportName: name});
    }
    const breaks = getGPUAnalysisOperator({
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'GPUClassBreaks'
    });
    for (const [route, maximum] of [
      ['materialized', 1024],
      ['recompute', 8192]
    ] as const) {
      expect(breaks?.routes?.find(candidate => candidate.name === route)).toMatchObject({
        planner: {
          package: '@luma-contributors/analysis/dataframe-analysis',
          exportName: 'GPUClassBreaks'
        },
        maximum,
        scope: 'domain',
        unit: 'rows'
      });
    }
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    expect(guide).toContain('`@luma-contributors/analysis/dataframe-analysis:GPUClassBreaks`');
    expect(guide).toContain('1,024 total value rows');
    expect(guide).toContain('8,192 total value rows');
  });

  it('review B4 rejects an independent marker-only canonical heading', () => {
    const fixture =
      '### GPURasterNDWI\n{/* lens:GPURasterNDWI */}\n::lens{scenario=ndwi rung=graph}';
    expect(hasResolvingContract('#gpurasterndwi', ['GPURasterNDWI'], fixture)).toBe(false);
  });

  it('review B4 original canonical headings resolve contracts independently of the catalog', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    const issues: string[] = [];
    for (const identity of ORIGINAL_CONTRACT_IDENTITIES) {
      const [entry, name] = identity.split(':');
      const preferred = PREFERRED_CONTRACT_TARGETS[identity];
      const page =
        name === 'GPUNetworkInhomogeneousK'
          ? './gpu-network/point-process.md'
          : preferred.split('#')[0];
      const sections = getSections(readFileSync(resolve(DOCS_ROOT, page), 'utf8')).filter(
        section =>
          !/parameters$/i.test(section.heading) &&
          new RegExp('\\b' + name + '\\b').test(section.heading)
      );
      if (
        !sections.length ||
        !sections.some(section => hasResolvingContract(page + '#' + section.anchor, [name], guide))
      )
        issues.push(entry + ':' + name);
    }
    expect(issues).toEqual([]);
    expect(
      hasResolvingContract(
        '#marker-only',
        ['GPUExample'],
        '### Marker only\n{/* lens:GPUExample */}\n::lens{scenario=example rung=graph}'
      )
    ).toBe(false);
  });

  it('review B3 rejects a real recording caption and pins the S2 form contract', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    expect(
      getReferenceTargetIssues(
        {
          entry: 'gpu-spatial-analysis',
          names: ['GPUS2CapCovering'],
          target: './gpu-spatial-analysis/global-coordinates.md#gpus2capcovering'
        },
        guide
      )
    ).not.toEqual([]);
    const form = getReferences(guide).find(
      reference =>
        reference.entry === 'gpu-spatial-analysis' &&
        reference.names.includes('CPUS2CapCoveringOperation')
    );
    expect(form?.target).toBe(
      './gpu-spatial-analysis/global-coordinates.md#operation-forms-s2-covers-and-intrinsic-arc-overlay'
    );
  });

  it('review B4 rejects a valid destination attached to the wrong public entry', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    expect(
      getReferenceTargetIssues(
        {
          entry: 'gpu-terrain',
          names: ['GPURasterNDWI'],
          target: './gpu-raster/spectral-indices.md#formula-and-band-contracts'
        },
        guide
      )
    ).not.toEqual([]);
  });

  it.each([
    'gpu-spatial-analysis/operators.md',
    'gpu-spatial-analysis-cross-reference.md'
  ])('review B4 checks published DST-14 destinations in %s', page => {
    const text = readFileSync(join(DOCS_ROOT, page), 'utf8');
    const issues: string[] = [];
    const found = new Set<string>();
    for (const name of [
      'GPULocalGeary',
      'GPUSpatialPearson',
      'GPUGammaStatistic',
      'GPUNearestFeatureWeights'
    ]) {
      for (const match of text.matchAll(/\[\x60?(\w+)\x60?\]\(([^)]+)\)/g)) {
        if (match[1] !== name) continue;
        found.add(name);
        const [file, anchor] = match[2].split('#');
        const destination = file.startsWith('/docs/api/')
          ? file.slice('/docs/api/'.length)
          : join(dirname(page), file);
        const target =
          './' +
          (destination.endsWith('.md') ? destination : destination + '.md') +
          (anchor ? '#' + anchor : '');
        issues.push(
          ...getReferenceTargetIssues(
            {entry: 'gpu-spatial-analysis', names: [name], target},
            readFileSync(GUIDE_PATH, 'utf8')
          ).map(issue => page + ': ' + issue)
        );
      }
    }
    expect([...found].sort()).toEqual(
      ['GPULocalGeary', 'GPUSpatialPearson', 'GPUGammaStatistic', 'GPUNearestFeatureWeights'].sort()
    );
    expect(issues).toEqual([]);
  });

  it('DST-11 matches callable examples and availability claims to actual public exports', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    const exports = getPublicFunctions(
      join(REPOSITORY_ROOT, 'modules/analysis/src/gpu-spatial-analysis/ontology/index.ts')
    );
    const section =
      getSections(guide).find(section => section.anchor === 'forms-task-search-and-output-plans')
        ?.text ?? '';
    const available = new Set(exports.keys());
    expect(getHelperAvailabilityIssues(section, available)).toEqual([]);
    expect(
      getHelperAvailabilityIssues(
        section + '\nPending public exports: describeOutputs and searchGPUAnalysisTasks.',
        available
      )
    ).not.toEqual([]);
    for (const name of ['describeOutputs', 'searchGPUAnalysisTasks']) {
      const missing = new Set(available);
      missing.delete(name);
      expect(getHelperAvailabilityIssues(section, missing)).toContain(
        `${name}: unavailable helper advertised as callable`
      );
    }
    expect(searchGPUAnalysisTasks('NDVI')).toContainEqual(
      expect.objectContaining({
        owner: 'upstream',
        identity: {package: '@luma.gl/graph/gpu-raster', exportName: 'GPURasterNDVI'}
      })
    );
    expect(
      searchGPUAnalysisTasks('change point', {question: 'what-changed'}).length
    ).toBeGreaterThan(0);
    expect(
      searchGPUAnalysisTasks('change point', {question: 'what-changed'}).every(result =>
        result.questions.includes('what-changed')
      )
    ).toBe(true);
    expect(
      getGPUAnalysisOperatorCatalog().some(operator => operator.package.startsWith('@luma.gl/'))
    ).toBe(false);
  });

  it('review S1 checks qualified function source paths and declarations and rejects swapped URLs', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    expect(getArraySourceIssues(guide)).toEqual([]);
    const links = [...guide.matchAll(/\[\x60(compute\w+OnCPU)\x60\]\(([^)]+)\)/g)];
    expect(links.length).toBeGreaterThan(125);
    const first = links[0];
    const second = links.find(link => link[2] !== first?.[2]);
    if (!first || !second) throw new Error('missing independent CPU references');
    expect(getArraySourceIssues(guide.replace(first[2], second[2]))).not.toEqual([]);
  });

  it('review S2 preserves complete sibling distinctions and all editorial questions', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    expect(getEditorialIssues(guide)).toEqual([]);
    const section = getSections(guide).find(section => section.anchor === 'choose-by-question');
    if (!section) throw new Error('missing task chooser');
    expect(getEditorialIssues(guide.replace(section.text, ''))).toHaveLength(15);
    const index =
      getSections(guide).find(section => section.anchor === 'spatial-export-index')?.text ?? '';
    const issues = index
      .split('\n')
      .filter(line => line.startsWith('| ' + String.fromCharCode(96)))
      .filter(line => {
        const description =
          line.match(/^\| [^|]+ \| [^|]+ \| \[[^\]]+\]\([^)]+\) \| (.*) \|$/)?.[1] ?? '';
        return (
          /…|See the named contract|from$|see \.$/.test(description) ||
          [...description.matchAll(/\x60/g)].length % 2 !== 0
        );
      });
    expect(issues).toEqual([]);
  });

  it('review S3 scopes the PySAL switch to the global empirical-Bayes form', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    const row =
      guide.split('\n').find(line => line.startsWith('| Smooth rates for small populations?')) ??
      '';
    expect(row).toContain('GPUEmpiricalBayesRates({clampPriorVariance: false})');
    expect(row).toContain('spatial form always clamps');
  });

  it('review S4 names selected coverage and distinguishes the owned recording links', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    for (const match of guide.matchAll(/\[([^\]]+)\]\(\.\/gpu-spatial-analysis\/operators\.md\)/g))
      expect(match[1]).toBe('Selected spatial operator families');
    expect(guide).not.toContain(
      '[GPU Spatial Analysis operators](./gpu-spatial-analysis/operators.md)'
    );
    expect(guide).not.toContain('[Expansion computation lenses]');
    expect(guide).toContain('[Terrain hydrology recordings]');
    expect(guide).toContain('[Raster operation recordings]');
  });

  it('review unfinished status scopes and output/route qualifications are source-backed', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    for (const phrase of [
      "stage: 'downstream-distance'",
      "stage: 'stream-network'",
      'shared-path coverage',
      'fallback islands',
      "selected level's occupied rows",
      'configured round budget',
      'finite predictor rows including masked rows or missing responses',
      'sourceLineIndices',
      '0xffffffff past output.status.count',
      'vertices for retained pieces only',
      'checked: false',
      'Source-validated route constraints',
      'gridSize[0] * gridSize[1] * 12 <= min(134217728',
      'lineCount * (lineCapacity + 1)',
      '2 * lineCapacity + positionCount'
    ])
      expect(guide, phrase).toContain(phrase);
  });

  it.each([
    ['Spatial analysis', 'gpu-spatial-analysis/README.md'],
    ['Network', 'gpu-network/README.md'],
    ['Terrain', 'gpu-terrain/README.md'],
    ['Raster analysis', 'gpu-raster/README.md']
  ])('DST-12 reaches every sidebar destination from the %s family body', (label, start) => {
    const navigation = JSON.parse(
      readFileSync(join(REPOSITORY_ROOT, 'docs/table-of-contents.json'), 'utf8')
    ) as NavigationItem[];
    const root = navigation[0];
    if (!root || typeof root === 'string') throw new Error('missing root category');
    const family = root.items.find(item => typeof item !== 'string' && item.label === label);
    if (!family || typeof family === 'string') throw new Error(`missing family ${label}`);
    const reached = getReachablePages(start);
    expect(
      getNavigationPages(family.items)
        .map(page => page.replace(/^api\//, '') + '.md')
        .filter(page => !reached.has(page))
    ).toEqual([]);
  });
  it('DOC-107 publishes the validation catalog at its canonical repository URL', () => {
    expect(readFileSync(GUIDE_PATH, 'utf8')).toContain(
      'https://github.com/visgl/luma-contributors/blob/master/examples/deck/kepler-compute/VALIDATION-CATALOG.md'
    );
  });

  it('DST-09 takes every shared pair to its domain and names its exported array functions', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    const references = getReferences(guide);
    for (const operator of getGPUAnalysisOperatorCatalog()) {
      if (!operator.forms?.cpu || !operator.forms.gpu) continue;
      for (const form of [operator.forms.cpu, operator.forms.gpu]) {
        const reference = references.find(
          reference =>
            reference.entry === form.package.split('/').pop() &&
            reference.names.includes(form.exportName)
        );
        expect(reference, form.exportName).toBeDefined();
        expect(reference?.target.startsWith('./'), form.exportName).toBe(true);
      }
      for (const array of operator.forms.arrays ?? [])
        expect(guide, array.exportName).toContain(
          '[' + String.fromCharCode(96) + array.exportName + String.fromCharCode(96) + ']('
        );
    }
    for (const [name, target] of [
      [
        'CPUPointToCellOperation',
        './gpu-spatial-analysis/operators-cells.md#gpupointtocell-and-gpucellgeometry'
      ],
      ['CPUAlternativeRoutesOperation', './gpu-network/alternative-routes.md#alternative-routes'],
      ['CPURasterFillHolesOperation', './gpu-raster/expansion.md#operations-and-contracts']
    ])
      expect(references.find(reference => reference.names.includes(name))?.target).toBe(target);
  });

  it('DST-10 requires prose at all reported qualified targets and rejects marker-only controls', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    for (const [identity, target] of Object.entries(PREFERRED_CONTRACT_TARGETS)) {
      const [entry, name] = identity.split(':');
      const reference = getReferences(guide).find(
        reference => reference.entry === entry && reference.names.includes(name)
      );
      expect(reference?.target, name).toBe(target);
      if (!reference) throw new Error('missing qualified contract ' + name);
      expect(getReferenceTargetIssues(reference, guide), name).toEqual([]);
    }
    const negative = {
      entry: 'raster-analysis',
      names: ['GPURasterNDWI'],
      target: './gpu-raster/spectral-indices.md#gpurasterndwi'
    };
    expect(getReferenceTargetIssues(negative, guide)).not.toEqual([]);
    expect(
      getSubstantiveContract(
        '### GPUExample and GPUOther\n{/* lens:GPUExample */}\n::lens{scenario=example rung=graph}'
      )
    ).toBe('');
  });

  it('DST-14 rejects wrong but existing class fragments', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    for (const [name, anchor] of [
      ['GPULocalGeary', 'gpuhotspotanalysis-and-gpulocalmoran'],
      ['GPUSpatialPearson', 'gpulocalpermutationtest-and-gpuglobalpermutationtest'],
      ['GPUGammaStatistic', 'additional-spatial-statistics-and-bounded-optimization'],
      ['GPUNearestFeatureWeights', 'gpupointinpolygonjoin-and-gpunearestfeaturejoin'],
      ['GPULocalGeary', 'additional-spatial-statistics-and-bounded-optimization'],
      ['GPUSpatialPearson', 'additional-spatial-statistics-and-bounded-optimization'],
      ['GPUGammaStatistic', 'additional-spatial-statistics-and-bounded-optimization']
    ]) {
      const file = name === 'GPUNearestFeatureWeights' ? 'operators-joins' : 'operators-statistics';
      const target = './gpu-spatial-analysis/' + file + '.md#' + anchor;
      expect(getTargetSection(target, guide)).not.toBe('');
      expect(
        getReferenceTargetIssues({entry: 'gpu-spatial-analysis', names: [name], target}, guide)
      ).toEqual([`wrong contract destination: ${name}`]);
    }
  });

  it('DST-08 exactly covers the spatial barrel with contract destinations and decisions', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    const section = getSections(guide).find(section => section.anchor === 'spatial-export-index');
    const references = getReferences(section?.text ?? '');
    const expected = getSpatialBrowseExports(
      join(REPOSITORY_ROOT, 'modules/analysis/src/gpu-spatial-analysis/index.ts')
    );
    expect(expected.size).toBeGreaterThanOrEqual(311); // All 311 reviewed exports, plus additive arrivals.
    expect(references.flatMap(reference => reference.names).sort()).toEqual([...expected].sort());
    expect(
      references.flatMap(reference => [
        ...(reference.entry === 'gpu-spatial-analysis' ? [] : ['wrong entry ' + reference.entry]),
        ...getReferenceTargetIssues(reference, guide).map(
          issue => reference.names.join(',') + ': ' + issue
        )
      ])
    ).toEqual([]);
    expect(guide).toContain(
      '[Selected spatial operator families](./gpu-spatial-analysis/operators.md)'
    );
  });

  it('DST-12 derives body chooser coverage from every sidebar page', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    const navigation = JSON.parse(
      readFileSync(join(REPOSITORY_ROOT, 'docs/table-of-contents.json'), 'utf8')
    ) as NavigationItem[];
    for (const page of getNavigationPages(navigation)) {
      if (page === 'api/analysis-catalog') continue;
      expect(guide, page).toContain('](' + './' + page.replace(/^api\//, '') + '.md)');
    }
    expect(readFileSync(join(DOCS_ROOT, 'README.md'), 'utf8')).toMatch(
      /\]\((?:\.\/)?analysis-catalog\.md(?:#[^)]*)?\)/
    );
  });

  it('DST-15 retains exact API coverage while grouping spatial navigation by task', () => {
    const navigation = JSON.parse(
      readFileSync(join(REPOSITORY_ROOT, 'docs/table-of-contents.json'), 'utf8')
    ) as NavigationItem[];
    const pages = getNavigationPages(navigation);
    expect(pages.slice().sort()).toEqual(
      ANALYSIS_DOC_PAGES.map(page => 'api/' + page.replace(/\.md$/, '')).sort()
    );
    const root = navigation[0];
    if (typeof root === 'string') throw new Error('expected root category');
    if (!root) throw new Error('missing root category');
    const spatial = root.items.find(
      item => typeof item !== 'string' && item.label === 'Spatial analysis'
    );
    if (!spatial || typeof spatial === 'string') throw new Error('missing spatial category');
    expect(spatial.items.length).toBeLessThanOrEqual(12);
    expect(spatial.items.filter(item => typeof item !== 'string').length).toBeGreaterThanOrEqual(7);
  });

  it('DST-15 separates compatibility URLs from canonical task destinations without losing anchors', () => {
    const navigation = JSON.parse(
      readFileSync(join(REPOSITORY_ROOT, 'docs/table-of-contents.json'), 'utf8')
    ) as NavigationItem[];
    const root = navigation[0];
    if (!root || typeof root === 'string') throw new Error('missing root category');
    for (const [label, canonical, legacy] of [
      [
        'Spatial analysis',
        ['gpu-spatial-analysis/home-ranges-and-recursions'],
        ['gpu-spatial-analysis/home-range-and-recursion-mob07']
      ],
      [
        'Network',
        [
          'gpu-network/network-time-geography',
          'gpu-network/origin-destination-flow-models',
          'gpu-network/map-matched-edge-volumes'
        ],
        [
          'gpu-network/network-time-geography-mob09',
          'gpu-network/flow-models-mob10',
          'gpu-network/map-matched-edge-volumes-mob05'
        ]
      ],
      [
        'Raster analysis',
        ['gpu-raster/flow-map-and-particle-diagnostics'],
        ['gpu-raster/lagrangian-diagnostics-mob06']
      ]
    ] as const) {
      const family = root.items.find(item => typeof item !== 'string' && item.label === label);
      if (!family || typeof family === 'string') throw new Error(`missing family ${label}`);
      const compatibility = family.items.find(
        item => typeof item !== 'string' && item.label === 'Compatibility URLs'
      );
      if (!compatibility || typeof compatibility === 'string')
        throw new Error(`missing compatibility group ${label}`);
      expect(getNavigationPages(compatibility.items).sort()).toEqual(
        legacy.map(page => 'api/' + page).sort()
      );
      const tasks = getNavigationPages(family.items.filter(item => item !== compatibility));
      for (const page of canonical) expect(tasks).toContain('api/' + page);
      for (const page of legacy) expect(tasks).not.toContain('api/' + page);
    }
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    expect(
      getReferences(guide).find(reference => reference.names.includes('GPUMapMatchedEdgeVolumes'))
        ?.target
    ).toBe('./gpu-network/map-matched-edge-volumes.md#edge-volumes-of-map-matched-tracks');
    // The rendered heading is the label alone; a historical URL in it must not affect the slug.
    expect(
      getSections('## [Edge volumes of map-matched tracks](./map-matched-edge-volumes.md)')[0]
        ?.anchor
    ).toBe('edge-volumes-of-map-matched-tracks');
    const alias =
      '## Previous\nPrevious prose.\n\n<a id="retained-contract"></a>\n\n## GPUExample\nGPUExample writes its output rows.';
    expect(getContractSection(alias, 'retained-contract')?.heading).toBe('GPUExample');
    expect(hasResolvingContract('#retained-contract', ['GPUExample'], alias)).toBe(true);
    expect(
      hasResolvingContract(
        '#retained-contract',
        ['GPUExample'],
        alias.replace('GPUExample writes its output rows.', '{/* lens:GPUExample */}')
      )
    ).toBe(false);
    for (const [page, anchor] of [
      ['gpu-network/map-matched-edge-volumes-mob05.md', 'edge-volumes-of-map-matched-tracks'],
      ['gpu-network/flow-models-mob10.md', 'gpuradiationflowmodel'],
      ['gpu-network/network-time-geography-mob09.md', 'gpunetworktraveltimepercentiles']
    ])
      expect(getContractSection(readFileSync(join(DOCS_ROOT, page), 'utf8'), anchor)).toBeDefined();
  });

  it('AMG-01 excludes packed transpose input cardinalities and documents the retained count', () => {
    // Dropping an out-of-range input column changes cardinality: input offsets [0, 2]
    // with neighbors [0, 9] and columnCount 1 retain one slot, not the input's two.
    for (const query of [queryGPUAnalysisOperators, queryGPUSpatialAnalysisOperators]) {
      for (const status of ['count', 'requiredCount'] as const) {
        expect(query({status}).map(record => record.exportName)).not.toContain(
          'GPUSpatialWeightsTranspose'
        );
      }
    }
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    expect(guide).toContain('Packed transpose excludes input `count` and `requiredCount`');
    expect(guide).toContain('`outputWeights.offsets[columnCount]`');
  });
  it('DOC-103 identifies aggregate, candidate and final join loss separately', () => {
    const page = readFileSync(CROSS_REFERENCE_PATH, 'utf8');
    const row =
      page
        .split('\n')
        .find(
          line => line.startsWith('| ') && line.includes('`output.matches.status.requiredCount`')
        ) ?? '';
    for (const path of [
      'output.status.overflow',
      'output.status.candidateOverflow',
      'output.matches.status.overflow',
      'output.matches.status.requiredCount'
    ])
      expect(row).toContain(`\`${path}\``);
    expect(row).toContain('retained assignments before final compaction');
    expect(row).toContain('Optional `output.matches.status.requiredCount`');
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    expect(guide).toContain(
      'When supplied, `matches.status.requiredCount` counts retained assignments before final compaction'
    );
    expect(guide).not.toContain('Optional compact matches also publish `matches.status.count` and');
    expect(page).not.toContain(
      'Every contributor runs on the GPU with bounded, caller-sized outputs and an overflow or count word'
    );
  });

  it('DOC-108 gives each public class an actual heading or an entry-qualified reference section', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    const references = getReferences(guide);
    const headings = ANALYSIS_DOC_PAGES.filter(page => page !== 'analysis-catalog.md').flatMap(
      page =>
        getSections(readFileSync(join(DOCS_ROOT, page), 'utf8'))
          .filter(section => !/parameters$/i.test(section.heading))
          .map(section => ({
            ...section,
            entry: getDocumentEntry(page),
            target: './' + page + '#' + section.anchor
          }))
    );
    const manifest = JSON.parse(
      readFileSync(join(REPOSITORY_ROOT, 'modules/analysis/package.json'), 'utf8')
    ) as {exports: Record<string, {import: string}>};
    const missing: string[] = [];
    for (const [entry, target] of Object.entries(manifest.exports)) {
      const file = join(
        REPOSITORY_ROOT,
        'modules/analysis',
        target.import.replace('./dist/', 'src/').replace(/\.js$/, '.ts')
      );
      for (const name of getPublicClasses(file)) {
        const entryName = entry === '.' ? '.' : entry.slice(2);
        const reference = references.find(
          reference => reference.entry === entryName && reference.names.includes(name)
        );
        if (reference) {
          missing.push(
            ...getReferenceTargetIssues(reference, guide).map(
              issue => `${entryName}:${name}: ${issue}`
            )
          );
        } else if (!hasOwnedHeading(entryName, name, headings))
          missing.push(`${entryName}:${name}`);
      }
    }
    expect(missing).toEqual([]);
  });

  it('B1 pins Rose to its inputs, circular convention, summary and bins contract', () => {
    const reference = getReferences(readFileSync(GUIDE_PATH, 'utf8')).find(
      reference =>
        reference.entry === 'gpu-spatial-analysis' && reference.names.includes('GPURoseStatistic')
    );
    expect(reference?.target).toBe(
      './gpu-spatial-analysis/operators-statistics.md#additional-spatial-statistics-and-bounded-optimization'
    );
  });

  it.each([
    ['GPUAZPRegions', './gpu-spatial-analysis/operators-statistics.md#gpuazpregions-parameters'],
    ['GPURoseStatistic', '#producers-documented-with-a-family']
  ])('B2 rejects the index or generated-parameters redirect for %s', (name, target) => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    const original = getReferences(guide).find(
      reference => reference.entry === 'gpu-spatial-analysis' && reference.names.includes(name)
    );
    expect(original).toBeDefined();
    if (!original) throw new Error(`missing reference ${name}`);
    const redirected = guide.replace(original.target, target);
    expect(getReferenceTargetIssues({...original, target}, redirected)).not.toEqual([]);
  });

  it('B2 rejects a same-name class heading from another owning entry', () => {
    expect(
      hasOwnedHeading('gpu-spatial-analysis', 'GPUExample', [
        {entry: 'gpu-network', heading: 'GPUExample'}
      ])
    ).toBe(false);
    expect(
      hasOwnedHeading('gpu-spatial-analysis', 'GPUExample', [
        {entry: 'gpu-spatial-analysis', heading: 'GPUExample'}
      ])
    ).toBe(true);
  });

  it('S3 preserves direct sections for the six already documented operation forms', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    const references = getReferences(guide);
    for (const [entry, name, target] of EXISTING_OPERATION_REFERENCES) {
      expect(
        references.find(reference => reference.entry === entry && reference.names.includes(name))
          ?.target,
        name
      ).toBe(target);
    }
  });

  it.each([
    ['warm', 'fit at 0.7 ms warm', 'fit at 70 ms warm'],
    ['changed-input', 'execution was 0.7 ms versus', 'execution was 70 ms versus'],
    ['CPU round trip', '2.3–2.5 ms', '0.1 ms']
  ])('S1 rejects a changed cooperative %s figure', (_comparison, original, replacement) => {
    const matrix = readFileSync(join(REPOSITORY_ROOT, 'dev-docs/coverage-matrix.md'), 'utf8');
    const row = matrix.split('\n').find(line => line.startsWith('| IN01 | Historical ')) ?? '';
    expect(row).toContain(original);
    expect(getCooperativeTimingIssues(row)).toEqual([]);
    expect(getCooperativeTimingIssues(row.replace(original, replacement))).not.toEqual([]);
  });

  it('CAT-102 pins publisher and forwarding examples to public records', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    // Independent publisher contracts already probed in catalog-review-round1.node.spec.ts:
    // the finest table publishes final loss; rollup and coarser levels also inherit source loss.
    const cellNames = ['GPUCellAggregation', 'GPUCellRollup', 'GPUCellPyramid'];
    for (const [status, expected] of [
      ['overflow', cellNames],
      ['candidateOverflow', ['GPUCellRollup', 'GPUCellPyramid']]
    ] as const) {
      expect(
        queryGPUAnalysisOperators({status})
          .map(record => record.exportName)
          .filter(name => cellNames.includes(name))
          .sort()
      ).toEqual([...expected].sort());
    }
    expect(getGPUAnalysisOperator('GPUShapeGenerator')?.status).toEqual(['vertexCount']);
    expect(getGPUAnalysisOperator('GPUEmergingHotSpots')?.status).toEqual(['count']);
    expect(getGPUAnalysisOperator('GPUSegmentSnap')?.status).toEqual([]);
    expect(getGPUAnalysisOperator('addSpatialRegressionRecipe')?.status).toEqual(['invalidCount']);
    const regressionStatus = guide.match(
      /^- `addSpatialRegressionRecipe` exposes[^\n]*(?:\n {2}[^\n]*)*/m
    )?.[0];
    expect(regressionStatus).toMatch(/exposes solver `stage\.code`/);
    expect(regressionStatus).toMatch(
      /`stage\.status\.invalidCount` counts excluded rows in each\s+stage's input domain/
    );
    expect(regressionStatus).toMatch(/per-observation `rowCodes`/);
    expect(regressionStatus).not.toMatch(/alias|must not be summed/i);
    for (const name of [
      'GPUSpatialWeightsTransform',
      'GPUSpatialWeightsTranspose',
      'GPUChunkedSpatialWeightsTransform'
    ])
      expect(getGPUAnalysisOperator(name)?.status).toContain('candidateOverflow');
    for (const phrase of [
      'publishes or forwards',
      'GPUCellAggregation',
      'GPUCellRollup',
      'GPUCellPyramid',
      'optional `vertexCount`',
      'whole weights',
      'per-chunk',
      'per-line',
      'input population'
    ])
      expect(guide, phrase).toContain(phrase);
  });

  it('B3 distinguishes local recipe scalars from reference overlay scalars', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      // Two buildings each have four boundary segments. Only graph view shapes are constructed;
      // this contract check allocates no GPU buffers and makes no numerical-output claim.
      const inputs = {
        segments: createTransientView(graph, 'review-segments', 'float32x4', 8),
        segmentBuildingIds: createTransientView(graph, 'review-owners', 'uint32', 8),
        enclosure: createTransientView(graph, 'review-enclosure', 'float32x2', 4),
        buildingCount: 2,
        sampleSpacing: 1,
        sampleCapacity: 8,
        vertexTolerance: 1 / 1024
      };
      const reference = addBuildingTessellationRecipe(graph, {
        ...inputs,
        id: 'review-reference',
        mode: 'reference',
        capacity: {intersections: 16, nodedSegments: 32, vertices: 32, rings: 4}
      });
      const local = addBuildingTessellationRecipe(graph, {
        ...inputs,
        id: 'review-local',
        mode: 'local',
        candidateRadius: 10
      });
      // Both public queries must find the dispatcher using its reference overlays' ring count.
      for (const query of [queryGPUAnalysisOperators, queryGPUSpatialAnalysisOperators]) {
        expect(query({status: 'requiredRingCount'}).map(record => record.exportName)).toContain(
          'addBuildingTessellationRecipe'
        );
      }
      expect(getGPUAnalysisOperator('addBuildingTessellationRecipe')?.limitation).toContain(
        'result.outputs[i].status.{count,overflow,candidateOverflow,requiredRingCount}'
      );
      const namedReference = addBuildingTessellationReferenceRecipe(graph, {
        ...inputs,
        id: 'review-named-reference',
        capacity: {intersections: 16, nodedSegments: 32, vertices: 32, rings: 4}
      });
      const namedLocal = addLocalBuildingTessellationRecipe(graph, {
        ...inputs,
        id: 'review-named-local',
        candidateRadius: 10
      });
      // Selected boundary-edge cardinality differs from the geometry's ring requirement.
      expect(namedReference.outputs[0].boundary.status.requiredCount.format).toBe('uint32');
      expect(namedReference.outputs[0].boundary.status.requiredCount.length).toBe(1);
      expect(
        queryGPUAnalysisOperators({status: 'requiredCount'}).map(record => record.exportName)
      ).toContain('addBuildingTessellationReferenceRecipe');
      expect(
        getGPUAnalysisOperator('addBuildingTessellationReferenceRecipe')?.limitation
      ).toContain('result.outputs[i].boundary.status.requiredCount');
      expect(namedReference.outputs).toHaveLength(2);
      expect(namedReference.status).not.toHaveProperty('overflow');
      expect(namedReference.outputs[0].status.requiredRingCount?.format).toBe('uint32');
      expect(namedLocal.status.overflow.format).toBe('uint32');
      expect(namedLocal.outputs).toHaveProperty('cells');
      expect(reference.outputs).toHaveLength(2);
      expect(reference.status).not.toHaveProperty('overflow');
      expect(reference.status.sampling.format).toBe('uint32x4');
      expect(reference.status.complete.length).toBe(2);
      for (const output of reference.outputs) {
        for (const name of [
          'count',
          'overflow',
          'candidateOverflow',
          'requiredRingCount'
        ] as const) {
          expect(output.status[name]?.format, name).toBe('uint32');
          expect(output.status[name]?.length, name).toBe(1);
        }
      }
      for (const name of [
        'count',
        'requiredCount',
        'overflow',
        'candidateOverflow',
        'invalidCount'
      ] as const) {
        expect(local.status[name].format, name).toBe('uint32');
        expect(local.status[name].length, name).toBe(1);
      }
      const guide = readFileSync(GUIDE_PATH, 'utf8');
      for (const path of [
        'result.status.overflow',
        'result.outputs[i].status.overflow',
        'result.outputs[i].status.requiredRingCount',
        'result.outputs[i].boundary.status.requiredCount'
      ])
        expect(guide).toContain(`\`${path}\``);
      expect(guide).not.toContain('count/overflow kinds exist only in local mode');
    } finally {
      device.destroy();
    }
  });

  it('review B2 discovers caller-owned reference overlay status without requiring a ring requirement', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const props = {
        segments: createTransientView(graph, 'supplied-segments', 'float32x4', 4),
        segmentBuildingIds: createTransientView(graph, 'supplied-owners', 'uint32', 4),
        enclosure: createTransientView(graph, 'supplied-enclosure', 'float32x2', 4),
        buildingCount: 1,
        sampleSpacing: 1,
        sampleCapacity: 4,
        vertexTolerance: 1 / 1024,
        capacity: {intersections: 16, nodedSegments: 32, vertices: 32, rings: 4}
      };
      const defaults = addBuildingTessellationReferenceRecipe(graph, {
        ...props,
        id: 'supplied-defaults'
      });
      const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
      const supplied: GPUPolygonOverlayOutput = {
        ...defaults.outputs[0],
        status: {
          count: scalar('supplied-count'),
          overflow: scalar('supplied-overflow'),
          candidateOverflow: scalar('supplied-candidates'),
          ringCount: scalar('supplied-rings'),
          uncertainCount: scalar('supplied-uncertain'),
          nodedSegmentRequiredCount: scalar('supplied-noded-requirement')
        }
      };
      for (const recipe of [
        addBuildingTessellationRecipe,
        addBuildingTessellationReferenceRecipe
      ]) {
        const result = recipe(graph, {
          ...props,
          id: `supplied-${recipe.name}`,
          mode: 'reference',
          output: {overlays: [supplied]}
        });
        // The public reference contract promises the supplied overlay object itself.
        expect(result.outputs[0]).toBe(supplied);
        expect(result.outputs[0].status?.overflow).toBe(supplied.status.overflow);
        expect(result.outputs[0].status).not.toHaveProperty('requiredRingCount');
      }
      for (const status of ['ringCount', 'uncertainCount', 'nodedSegmentRequiredCount'] as const) {
        expect(supplied.status[status]?.format).toBe('uint32');
        expect(supplied.status[status]?.length).toBe(1);
        expect(
          queryGPUSpatialAnalysisOperators({status}).map(record => record.exportName)
        ).toContain('addBuildingTessellationRecipe');
        for (const exportName of [
          'addBuildingTessellationRecipe',
          'addBuildingTessellationReferenceRecipe'
        ]) {
          expect(queryGPUAnalysisOperators({status}).map(record => record.exportName)).toContain(
            exportName
          );
        }
      }
      const guide = readFileSync(GUIDE_PATH, 'utf8');
      expect(guide).toContain('default reference layout');
      expect(guide).toContain('caller-owned overlay');
      for (const path of [
        'result.outputs[i].status.ringCount',
        'result.outputs[i].status.uncertainCount',
        'result.outputs[i].status.nodedSegmentRequiredCount'
      ])
        expect(guide).toContain(`\`${path}\``);
    } finally {
      device.destroy();
    }
  });

  it('review B3 describes variable-vertex convex polygon pieces', () => {
    // Hand-worked Voronoi fixture: for site (-1,-1) versus (1,-1),(-1,1),
    // bisectors are x=0 and y=0. Clipping [-2,2]^2 gives the FOUR vertices
    // (-2,-2),(0,-2),(0,0),(-2,0). Pieces therefore cannot be described as triangles.
    const local = getGPUAnalysisOperator('addLocalBuildingTessellationRecipe');
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    expect(local?.summary).toContain('convex polygon pieces');
    expect(local?.limitation).toContain('vertex count');
    expect(guide).toContain("Read each `cells` record's first word for that piece's vertex count");
    expect(guide).not.toContain('triangle unions');
  });

  it('review S1 pins the chunk status path and independently forwards all three status records', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    expect(getChunkedWeightsStatusIssues(guide)).toEqual([]);
    expect(
      getChunkedWeightsStatusIssues(
        guide.replaceAll('outputWeights.chunks[i].weights.status', 'outputWeights.chunks[i].status')
      )
    ).toEqual(['missing per-chunk status path']);
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
      const statuses = ['outer', 'first', 'second'].map(name => ({
        overflow: scalar(`${name}-overflow`),
        candidateOverflow: scalar(`${name}-candidates`)
      }));
      const chunks = [0, 1].map(index => ({
        rowStart: 2 * index,
        rowCount: 2,
        weights: {
          offsets: createTransientView(graph, `chunk-${index}-offsets`, 'uint32', 3),
          neighbors: createTransientView(graph, `chunk-${index}-neighbors`, 'uint32', 2),
          weights: createTransientView(graph, `chunk-${index}-weights`, 'float32', 2),
          status: statuses[index + 1]
        }
      }));
      const transformed = new GPUChunkedSpatialWeightsTransform({
        operation: 'binary',
        weights: {rowCount: 4, status: statuses[0], chunks}
      });
      const actual = [
        transformed.outputWeights.status,
        ...transformed.outputWeights.chunks.map(chunk => chunk.weights.status)
      ];
      for (const [index, status] of actual.entries()) {
        expect(status?.overflow).toBe(statuses[index].overflow);
        expect(status?.candidateOverflow).toBe(statuses[index].candidateOverflow);
      }
      expect(new Set(actual.map(status => status?.overflow)).size).toBe(3);
      expect(new Set(actual.map(status => status?.candidateOverflow)).size).toBe(3);
    } finally {
      device.destroy();
    }
  });

  it('review earlier-fix note discovers polygon-precision edge requirements independently of rings', () => {
    // gpu-polygonize directed-edge-status reads the noded requirement and writes the directed
    // requirement; both are edge counts, not output feature counts or geometry ring counts.
    for (const query of [queryGPUAnalysisOperators, queryGPUSpatialAnalysisOperators]) {
      expect(query({status: 'requiredCount'}).map(record => record.exportName)).toContain(
        'GPUPolygonPrecisionReduction'
      );
    }
    const record = getGPUAnalysisOperator('GPUPolygonPrecisionReduction');
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    for (const path of [
      'output.nodedSegments.status.requiredCount',
      'output.directedEdges.status.requiredCount'
    ]) {
      expect(record?.limitation).toContain(path);
      expect(guide).toContain(`\`${path}\``);
    }
  });

  it('labels upstream raster symbols individually in mixed-origin rows', () => {
    const page = readFileSync(CROSS_REFERENCE_PATH, 'utf8');
    const measurements =
      page.split('\n').find(line => line.includes('| `skimage.measure.regionprops`')) ?? '';
    const components =
      page.split('\n').find(line => line.includes('| `scipy.ndimage.label`')) ?? '';
    expect(measurements).toMatch(
      /\[\`GPURasterRegionMeasurements\`\]\([^)]+\) \(\`@luma\.gl\/graph\/gpu-raster\`\)/
    );
    expect(measurements).toMatch(
      /\[\`GPURasterPatchMetrics\`\]\([^)]+\) \(\`@luma-contributors\/analysis\/raster-analysis\`\)/
    );
    expect(components).toContain('(`@luma.gl/graph/gpu-raster`)');
  });

  it('keeps same-name visibility recipes linked by their owning entry', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    const references = getReferences(guide);
    for (const [entry, target] of [
      [
        'gpu-spatial-analysis',
        './gpu-spatial-analysis/urban-morphology.md#addisovistsrecipe-and-addvisibilitygraphrecipe--ur05'
      ],
      ['gpu-network', './gpu-network/visibility-graph.md#addvisibilitygraphrecipe']
    ]) {
      expect(references).toContainEqual({entry, names: ['addVisibilityGraphRecipe'], target});
      expect(getTargetSection(target, guide)).toContain('addVisibilityGraphRecipe');
    }
  });

  it('DOC-109 attributes the serial and cooperative comparisons to separate dated sections', () => {
    const matrix = readFileSync(join(REPOSITORY_ROOT, 'dev-docs/coverage-matrix.md'), 'utf8');
    expect(matrix).toContain('10.0.0-alpha.10');
    expect(matrix).toContain('not walked again for the integration target `10.0.0-alpha.16`');
    const row =
      matrix
        .split('\n')
        .find(
          line => line.startsWith('| IN01 | The ') || line.startsWith('| IN01 | Historical ')
        ) ?? '';
    expect(row).toContain(
      'interpolation-expansion.md#direct-reference-parity-and-expanded-scaling'
    );
    expect(row).toContain(
      'integration-notes.md#variogram-refinement-integration-2026-10-24md-variogram-model-fit'
    );
    expect(row).toContain('256-bin, three-model, pair-weighted');
    expect(row).toContain('warm GPU execution');
    expect(row).toContain('including transfers');
    expect(row).toContain('GPU execution excludes output readback');
    expect(row).toContain('4.3 / 51.3 ms for 20 / 256 bins');
    expect(row).toContain('1.3 / 3.2 ms');
  });

  it('keeps the printed unknown, audited-empty and mode-dependent status examples accurate', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    const tap = {
      package: '@luma-contributors/analysis/computation-record',
      exportName: 'GPUSnapshotTap'
    } as const;
    const applicability = {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUAreaOfApplicability'
    } as const;
    expect(getGPUAnalysisOperator(tap)?.status).toBe('unknown');
    expect(getGPUAnalysisOperator(applicability)?.status).toEqual([]);
    expect(
      queryGPUAnalysisOperators({status: 'unknown'}).some(
        operator => operator.exportName === tap.exportName && operator.package === tap.package
      )
    ).toBe(true);
    expect(
      queryGPUAnalysisOperators({status: 'overflow'}).some(
        operator => operator.exportName === tap.exportName && operator.package === tap.package
      )
    ).toBe(false);
    expect(
      queryGPUAnalysisOperators({status: 'overflow'}).some(
        operator => operator.exportName === 'addBuildingTessellationRecipe'
      )
    ).toBe(true);
    for (const phrase of [
      'GPUSnapshotTap',
      'GPUAreaOfApplicability',
      '`result.status.overflow`',
      "queryGPUAnalysisOperators({status: 'unknown'})",
      'historical query keeps its array-membership behavior'
    ])
      expect(guide).toContain(phrase);
  });

  it('DOC-04 pins each fitting producer to its analysis-core section and rejects omissions or redirects', () => {
    const guide = readFileSync(GUIDE_PATH, 'utf8');
    expect(getFittingReferenceIssues(guide)).toEqual([]);
    for (const [name, target] of FITTING_REFERENCES) {
      expect(getFittingReferenceIssues(guide.replaceAll(name, 'RemovedProducer'))).toContain(name);
      expect(
        getFittingReferenceIssues(guide.replace(`[\`${name}\`](${target})`, `\`${name}\``))
      ).toContain(name);
      expect(
        getFittingReferenceIssues(guide.replaceAll(target, './computation-record.md#snapshots'))
      ).toContain(name);
    }
  });

  it('compiles the printed linear-referencing fragment and rejects retired props and misspelled metrics', () => {
    const page = readFileSync(CROSS_REFERENCE_PATH, 'utf8');
    const section = getSections(page).find(
      section => section.anchor === 'great-circle-linear-referencing-fragment'
    );
    const snippet = section?.text.match(/```ts\n([\s\S]*?)\n```/)?.[1];
    expect(snippet).toBeDefined();
    if (!snippet) return;
    expect(snippet).toContain("from '@luma-contributors/analysis/gpu-spatial-analysis'");
    expect(snippet).toContain(
      "spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}"
    );
    expect(snippet).not.toMatch(/\bany\b|@ts-|\.\.\./);
    const snippets = [
      snippet,
      snippet.replace("metric: 'great-circle'", "metric: 'great-circel'"),
      snippet.replace('spatialContext:', "coordinateSystem: 'spherical', spatialContext:")
    ];
    const files = snippets.map(
      (text, index) =>
        [join(REPOSITORY_ROOT, `catalog-linear-reference-${index}.ts`), text] as const
    );
    const program = createVirtualProgram(new Map(files), {
      ...getRepositoryCompilerOptions(),
      strict: true
    });
    const diagnostics = files.map(([file]) => {
      const source = program.getSourceFile(file);
      if (!source) throw new Error(`missing fragment ${file}`);
      return [
        ...program.getSyntacticDiagnostics(source),
        ...program.getSemanticDiagnostics(source)
      ].map(formatDiagnostic);
    });
    expect(diagnostics[0]).toEqual([]);
    expect(diagnostics[1].join('\n')).toContain('great-circel');
    expect(diagnostics[2].join('\n')).toContain('coordinateSystem');
  }, 60000);
});
