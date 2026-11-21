// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import type {Plugin} from 'vite';
import {DOMParser} from '@xmldom/xmldom';
import {
  getGPUAnalysisOperator,
  queryGPUAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';
import {queryGPUSpatialAnalysisOperators} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-operators';
import {
  findGPUAnalysisOperatorPaths,
  queryGPUAnalysisOperatorsByKind
} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-operator-kind-supplements';

function hasOrderedPrefixSubmissionExample(snapshots: string): boolean {
  const steps = [
    'compiled.encode(earlierEncoder, {parameters: undefined});',
    'const endCheckpoint = arena.checkpoint();',
    'compiled.encode(laterEncoder, {parameters: undefined});',
    'arena.markSubmitted(endCheckpoint);',
    'device.submit(earlierEncoder.finish());',
    'arena.discard(endCheckpoint);'
  ];
  return [...snapshots.matchAll(/^```ts\n([\s\S]*?)^```/gm)].some(match => {
    let previous = -1;
    return steps.every(step => {
      const position = match[1].indexOf(step);
      if (position < 0 || position <= previous) return false;
      previous = position;
      return true;
    });
  });
}

it('R1 S1 checks the complete prefix-submission fence and rejects missing or reordered steps', () => {
  const guide = readFileSync('docs/api/computation-record.md', 'utf8');
  expect(hasOrderedPrefixSubmissionExample(guide)).toBe(true);
  const steps = [
    'compiled.encode(earlierEncoder, {parameters: undefined});',
    'const endCheckpoint = arena.checkpoint();',
    'compiled.encode(laterEncoder, {parameters: undefined});',
    'arena.markSubmitted(endCheckpoint);',
    'device.submit(earlierEncoder.finish());',
    'arena.discard(endCheckpoint);'
  ];
  for (const step of steps)
    expect(hasOrderedPrefixSubmissionExample(guide.replace(step, '')), step).toBe(false);
  expect(
    hasOrderedPrefixSubmissionExample(
      guide
        .replace(steps[0], steps[5])
        .replace(/arena\.discard\(endCheckpoint\);(?=\n```)/, steps[0])
    )
  ).toBe(false);
});

it('R1 B4 qualifies optional execution stages and the fallback dispatch evidence', () => {
  const guide = readFileSync('docs/api/analysis-catalog.md', 'utf8').split(
    '## The operator catalog'
  )[0];
  for (const phrase of [
    '`result.stages` is optional',
    'Raster embedding PCA and projection supply it',
    '`result.dispatchedContributors`',
    'does not reconstruct CPU stages'
  ])
    expect(guide.includes(phrase), phrase).toBe(true);
});

it('R1 S2 distinguishes index relation rows from the destination key-space size', () => {
  const guide = readFileSync('docs/api/computation-record.md', 'utf8');
  expect(guide.includes('`relation.targets.length === source.length`')).toBe(true);
  expect(guide.includes('`target.length` is the size of the destination key space')).toBe(true);
  expect(guide).not.toContain('Target and source lengths match');
});

it('R1 earlier-fix note documents packed-step underflow separately from duration admission', () => {
  const guide = readFileSync('docs/api/analysis-catalog.md', 'utf8').split(
    '## The operator catalog'
  )[0];
  expect(guide.includes('Math.fround(timeStep)')).toBe(true);
  expect(guide.includes('`-1e-46`')).toBe(true);
  expect(guide.includes('packs to negative zero')).toBe(true);
});

it('TYP-06 documents synchronous rejection and independent asynchronous recorder namespaces', () => {
  const guide = readFileSync('docs/api/computation-record.md', 'utf8');
  const section = guide.split('## Recording a command graph')[1]?.split('\n## ')[0] ?? '';
  expect(section).toContain('recorder.withSynchronousScope(');
  expect(section).toContain('native async callback is rejected before invocation');
  expect(section).toContain('thenable result is rejected after the callback returns');
  expect(section).toContain('already-started work cannot be cancelled');
  expect(section).toContain('await recorder.withAsyncScope(');
  expect(section).toMatch(/await prepareNode\('basin'\);\s*scopedRecorder\.addNode/);
  expect(section).toContain("namespace: 'basin'");
  expect(section).toContain("namespace: 'coast'");
  expect(section).toContain('await Promise.all([basinTask, coastTask])');
  expect(section).not.toContain('nodes recorded after `await` lose that');
});

it('LIF-13 documents sealed retryable readback and encoder-aligned prefix submission', () => {
  const guide = readFileSync('docs/api/computation-record.md', 'utf8');
  const snapshots = guide.split('## Snapshots')[1]?.split('## `GPUTransferMask`')[0] ?? '';
  for (const phrase of [
    'first read permanently seals',
    'one in-flight mapping',
    'successful bytes are cached',
    'retry after rejection',
    'count * frameByteLength',
    'Empty byte ranges are not mapped',
    'independent typed-array copies',
    'arena.markSubmitted(endCheckpoint)',
    'Neither submission nor discard can split an encoder',
    '[13, 15]',
    'Destroying the arena also loses unread submitted frames'
  ])
    expect(snapshots, phrase).toContain(phrase);
  expect(snapshots).not.toContain('The arena maps once');
  expect(hasOrderedPrefixSubmissionExample(snapshots)).toBe(true);
});

it('UCO-13 documents native transport, inherited geometry and route-specific admission', () => {
  const guide = readFileSync('docs/api/analysis-catalog.md', 'utf8');
  const section = guide.split('## The operator catalog')[0];
  for (const phrase of [
    'acceptsTable',
    'geometryFromInput',
    'feature_index',
    'runDataGpu',
    'dispatchedContributors',
    'CPU eigensolver',
    'decimal/scientific',
    'Math.fround(timeStep * 10)',
    'CRS units',
    '20,000 total input points',
    '6,000 total input points'
  ])
    expect(section, phrase).toContain(phrase);
  expect(section).toContain('^[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?$');
  expect(section).toContain('geographic metric-grid controls use metres');
  expect(section).toContain('planar controls use coordinate units');
});

it('VAL-04 documents immutable snapshot slots, own dictionaries and strict transfer selectors', () => {
  const guide = readFileSync('docs/api/computation-record.md', 'utf8');
  const snapshots = guide.split('## Snapshots')[1]?.split('## `GPUTransferMask`')[0] ?? '';
  for (const phrase of [
    '`f32`, `i32` or `u32`',
    'before allocation',
    'descriptor order',
    'immutable',
    'own-property',
    '`sources`, `resources` and `keySpaces`'
  ])
    expect(snapshots, phrase).toContain(phrase);
  const transfer = guide.split('## `GPUTransferMask`')[1]?.split('\n## ')[0] ?? '';
  expect(transfer).toContain('mode === undefined');
  expect(transfer).toContain('no default layout');
  expect(transfer).toContain('unknown values');
  expect(transfer).toContain('before graph authoring');
  expect(transfer).toContain('accepted values and the received value');
});

function parseRenderedPage(result: unknown): {html: string} {
  if (
    !result ||
    typeof result !== 'object' ||
    !('code' in result) ||
    typeof result.code !== 'string'
  ) {
    throw new Error('Expected compiled Markdown');
  }
  return JSON.parse(result.code.replace(/^export default /, '').replace(/;$/, '')) as {
    html: string;
  };
}

/** Accessible-name rules used by these static links: labelledby, label, alt, exposed content,
 * and title fallback. Hidden links/content are excluded; explicitly referenced labels may be hidden.
 * No browser or stylesheet computation is used by this host-only renderer check. */
function getAccessibleLinkName(html: string, target: string): string | undefined {
  const document = new DOMParser().parseFromString(
    '<html><body>' + html + '</body></html>',
    'text/html'
  );
  const link = Array.from(document.getElementsByTagName('a')).find(
    element => element.getAttribute('href') === target
  );
  if (!link) return undefined;
  const isHidden = (element: Element): boolean =>
    element.hasAttribute('hidden') ||
    element.getAttribute('aria-hidden') === 'true' ||
    /(?:display\s*:\s*none|visibility\s*:\s*(?:hidden|collapse))/i.test(
      element.getAttribute('style') ?? ''
    );
  for (
    let parent: Element | null = link;
    parent;
    parent = parent.parentNode?.nodeType === 1 ? (parent.parentNode as Element) : null
  )
    if (isHidden(parent)) return '';
  const visited = new Set<Node>();
  const name = (node: Node, referenced = false): string => {
    if (visited.has(node)) return '';
    visited.add(node);
    if (node.nodeType === 3) return node.nodeValue ?? '';
    if (node.nodeType !== 1) return '';
    const element = node as Element;
    if (
      (!referenced && isHidden(element)) ||
      ['script', 'style'].includes(element.tagName.toLowerCase())
    )
      return '';
    const labelledBy = element.getAttribute('aria-labelledby')?.trim();
    if (labelledBy) {
      const references = labelledBy
        .split(/\s+/)
        .map(identifier => document.getElementById(identifier))
        .filter(label => label !== null);
      if (references.length) return references.map(label => name(label, true)).join(' ');
    }
    const label = element.getAttribute('aria-label')?.trim();
    if (label) return label;
    if (element.tagName.toLowerCase() === 'img') return element.getAttribute('alt') ?? '';
    const content = Array.from(element.childNodes)
      .map(child => name(child, referenced))
      .join(' ');
    return content.trim() || element.getAttribute('title') || '';
  };
  return name(link).replace(/\s+/g, ' ').trim();
}

it.each([
  undefined,
  null,
  {},
  {code: undefined},
  {code: {toString: () => 'export default {}'}}
])('review B2 refuses non-string transform code: %j', result => {
  expect(() => parseRenderedPage(result)).toThrow('Expected compiled Markdown');
});

it.each([
  [
    '<a href="/explore/" aria-label="Spatial catalog"><svg aria-hidden="true" /></a>',
    'Spatial catalog'
  ],
  ['<a href="/explore/"><span aria-hidden="true">Spatial catalog</span></a>', ''],
  [
    '<a href="/explore/" aria-label="Spatial catalog"><span aria-hidden="true">Wrong text</span></a>',
    'Spatial catalog'
  ],
  ['<a href="/explore/" hidden="hidden" aria-label="Spatial catalog">Spatial catalog</a>', '']
])('review S5 checks the accessible link name: %s', (html, expected) => {
  expect(getAccessibleLinkName(html, '/explore/')).toBe(expected);
});

async function renderLensMarkdown(source: string): Promise<string> {
  const compiler = (await import(resolve('website/build/markdown-plugin.ts'))) as {
    markdownPlugin: (options: {docsDirectory: string}) => Plugin;
  };
  const plugin = compiler.markdownPlugin({docsDirectory: resolve('docs')});
  if (typeof plugin.transform !== 'function') throw new Error('Expected Markdown transform');
  const result = await plugin.transform.call(
    {} as never,
    source,
    resolve('docs/api/analysis-catalog.md')
  );
  return parseRenderedPage(result).html;
}

it.each([
  ['::lens-catalog{entry=gpu-spatial-analysis}', '/explore/', /spatial|catalog/i],
  [
    '::lens{scenario=gpu-spatial-analysis.point-density rung=graph}',
    '/fn/point-density',
    /point[ -]density/i
  ]
])('DST-07 renders a destination and accessible label for %s', async (source, target, label) => {
  const html = await renderLensMarkdown(source);
  const accessibleName = getAccessibleLinkName(html, target);
  expect(accessibleName, html).toBeDefined();
  expect(accessibleName).toMatch(label);
  expect(html).not.toContain('::lens');
});

it('DST-07 links an unlisted scenario to its published recording', async () => {
  const scenario = 'gpu-dataframe.coregionalization-fit';
  const model = JSON.parse(readFileSync(`website/src/lens/generated/${scenario}.json`, 'utf8'));
  expect(model).toMatchObject({id: scenario});

  const html = await renderLensMarkdown(`::lens{scenario=${scenario} rung=graph}`);
  const label = getAccessibleLinkName(html, `/lens/${scenario}`);
  expect(label, html).toBeDefined();
  expect(label, html).toMatch(/coregionalization[ -]fit/i);
});

it('DST-07 renders four adjacent directives as separate links and preserves fenced code', async () => {
  const directives = ['datum', 'formula', 'kernel', 'graph'].map(
    rung => `::lens{scenario=gpu-dataframe.time-window-filter rung=${rung}}`
  );
  const html = await renderLensMarkdown(
    [...directives, '', '~~~text', directives[0], '~~~'].join('\n')
  );
  const document = new DOMParser().parseFromString(
    '<html><body>' + html + '</body></html>',
    'text/html'
  );
  const links = Array.from(document.getElementsByTagName('a'));
  expect(
    links.map(link => link.getAttribute('href')),
    html
  ).toEqual(Array<string>(4).fill('/fn/time-window-filter'));
  expect(links.every(link => /time window filter/i.test(link.textContent ?? ''))).toBe(true);
  expect(document.getElementsByTagName('code')[0]?.textContent).toContain(directives[0]);
});

it.each([
  'GPULocalMoran',
  'GPULocalGeary',
  'GPUHotSpotAnalysis',
  'GPUGlobalSpatialStatistics',
  'GPUPointInPolygonJoin'
])('F5 discovers candidate loss through both public APIs: %s', exportName => {
  for (const query of [queryGPUSpatialAnalysisOperators, queryGPUAnalysisOperators]) {
    expect(query({status: 'candidateOverflow'}).map(operator => operator.exportName)).toContain(
      exportName
    );
  }
});

it('F5 preserves the distinct polygon-join compact and aggregate overflow documentation', () => {
  const source = readFileSync(
    'modules/analysis/src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join.ts',
    'utf8'
  );
  expect(source).toContain('Its overflow flag reports final match-capacity loss only');
  expect(source).toContain('1 when BVH leaf, candidate, or `matches` capacity overflowed');
});

it('F6 discovers gather candidate loss without promising an unavailable requiredCount', () => {
  for (const query of [queryGPUSpatialAnalysisOperators, queryGPUAnalysisOperators]) {
    expect(query({status: 'candidateOverflow'}).map(operator => operator.exportName)).toContain(
      'GPUPairGather'
    );
    expect(query({status: 'requiredCount'}).map(operator => operator.exportName)).not.toContain(
      'GPUPairGather'
    );
  }
  const gather = getGPUAnalysisOperator('GPUPairGather')!;
  expect(gather.status).toEqual(['count', 'overflow', 'candidateOverflow', 'matchedCount']);
  expect(gather.limitation).not.toContain('requiredCount');
});

it('F7 distinguishes retained-candidate lower bounds from visibility unknown counts', () => {
  const guide = readFileSync('docs/api/analysis-catalog.md', 'utf8');
  expect(guide.includes('publishers that count retained candidates')).toBe(true);
  expect(guide.includes('0xffffffff')).toBe(true);
  expect(guide.includes('never use it as an allocation size')).toBe(true);
  expect(guide.includes('CSR failures suppress asymmetric graphs and misleading metrics')).toBe(
    true
  );
});

it.each([
  [
    'addZonalDistributionRecipe',
    ['points', 'polygons', 'attributes', 'parameters'],
    ['attributes', 'summary']
  ],
  [
    'addRasterZonalDistributionRecipe',
    ['surface', 'attributes', 'parameters'],
    ['attributes', 'summary']
  ],
  [
    'addInterpolationDiagnosticsRecipe',
    ['points', 'attributes', 'parameters'],
    ['surface', 'attributes', 'summary']
  ],
  ['addExposureByZoneRecipe', ['surface', 'attributes', 'parameters'], ['attributes', 'summary']],
  [
    'addRegionalRasterChangeRecipe',
    ['surface', 'attributes', 'parameters'],
    ['surface', 'attributes', 'summary']
  ],
  [
    'addDrainageNetworkRecipe',
    ['surface', 'attributes', 'parameters'],
    ['surface', 'network', 'summary']
  ],
  ['addLocalVariabilityRecipe', ['spatial-weights', 'attributes', 'parameters'], ['attributes']],
  [
    'addCellFlowTopologyRecipe',
    ['cells', 'pairs', 'attributes'],
    ['cells', 'pairs', 'network', 'attributes', 'render-geometry']
  ],
  ['addSurfacePreparationRecipe', ['surface', 'parameters'], ['surface', 'attributes', 'summary']]
])('F8 uses actual recipe ports: %s', (exportName, inputs, outputs) => {
  const operator = getGPUAnalysisOperator(exportName as string)!;
  expect(operator.inputs).toEqual(inputs);
  expect(operator.outputs).toEqual(outputs);
});

it('F8 recommends local variability for weights to attributes, never surface to network', () => {
  expect(
    queryGPUAnalysisOperatorsByKind({input: 'surface', output: 'network'}).map(
      operator => operator.exportName
    )
  ).not.toContain('addLocalVariabilityRecipe');
  expect(
    findGPUAnalysisOperatorPaths('surface', 'network', 1)
      .paths.flat()
      .map(step => step.exportName)
  ).not.toContain('addLocalVariabilityRecipe');
  expect(
    queryGPUAnalysisOperatorsByKind({input: 'spatial-weights', output: 'attributes'}).map(
      operator => operator.exportName
    )
  ).toContain('addLocalVariabilityRecipe');
  expect(
    findGPUAnalysisOperatorPaths('spatial-weights', 'attributes', 1)
      .paths.flat()
      .map(step => step.exportName)
  ).toContain('addLocalVariabilityRecipe');
});

it.each([
  '/',
  '/luma-contributors/'
])('G5 renders catalog links within the configured base %s', async basePath => {
  expect(existsSync('docs/lens-records/gpu-spatial-analysis.point-density.json')).toBe(true);
  const compiler = (await import(resolve('website/build/markdown-plugin.ts'))) as {
    markdownPlugin: (options: {docsDirectory: string}) => Plugin;
  };
  const plugin = compiler.markdownPlugin({docsDirectory: resolve('docs')});
  if (typeof plugin.configResolved !== 'function') throw new Error('Expected config hook');
  await plugin.configResolved.call({} as never, {base: basePath} as never);
  if (typeof plugin.transform !== 'function') throw new Error('Expected Markdown transform');
  const path = resolve('docs/api/analysis-catalog.md');
  const source = `${readFileSync(path, 'utf8')}

[Already prefixed](${basePath}explore/?tab=figures#recorded)
[External](https://example.org/explore/)
[Protocol relative](//example.org/readme.md)
[Fragment](#the-operator-catalog)
`;
  const result = await plugin.transform.call({} as never, source, path);
  if (!result || typeof result === 'string' || typeof result.code !== 'string') {
    throw new Error('Expected compiled Markdown module');
  }
  const rendered = JSON.parse(result.code.replace(/^export default /, '').replace(/;$/, '')) as {
    html: string;
  };
  expect(rendered.html.includes('::lens-catalog')).toBe(false);
  const hrefs = [...rendered.html.matchAll(/href="([^"]+)"/g)].map(match => match[1]);
  expect(hrefs).toContain(`${basePath}fn/point-density`);
  expect(hrefs).toContain(`${basePath}explore/`);
  expect(hrefs).toContain(`${basePath}explore/?tab=figures#recorded`);
  expect(hrefs).toContain('https://example.org/explore/');
  expect(hrefs).toContain(
    'https://github.com/visgl/luma-contributors/blob/master/examples/deck/kepler-compute/VALIDATION-CATALOG.md'
  );
  expect(hrefs.some(href => href.includes('/docs/../examples/'))).toBe(false);
  expect(hrefs).toContain('//example.org/readme.md');
  expect(hrefs).toContain('#the-operator-catalog');
});
