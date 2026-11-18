// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import type {Plugin} from 'vite';
import {
  getGPUAnalysisOperator,
  queryGPUAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';
import {queryGPUSpatialAnalysisOperators} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-operators';
import {
  findGPUAnalysisOperatorPaths,
  queryGPUAnalysisOperatorsByKind
} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-operator-kind-supplements';

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
  expect(gather.status).toEqual(['overflow', 'candidateOverflow', 'matchedCount']);
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
  expect(hrefs).toContain('//example.org/readme.md');
  expect(hrefs).toContain('#the-operator-catalog');
});
