// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readdirSync, writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {expect, it} from 'vitest';
import {ANALYSIS_DOC_PAGES, DOCS_ROOT, getSidebarPageIssues, readDocPage} from './doc-program';
import {collectClassParameters, generateDocPages, type ClassParameters} from './doc-parameters';

let reviewClasses: ClassParameters[] | undefined;
function getReviewClasses(): ClassParameters[] {
  reviewClasses ??= collectClassParameters();
  return reviewClasses;
}

it('R1 documents only seed for line sampling while retaining every dot-density setting', () => {
  const classes = getReviewClasses();
  expect(
    classes
      .find(entry => entry.className === 'GPURandomPointsOnLine')
      ?.packers[0].fields.map(field => field.name)
  ).toEqual(['seed']);
  expect(
    classes
      .find(entry => entry.className === 'GPUDotDensity')
      ?.packers[0].fields.map(field => field.name)
  ).toEqual(['seed', 'dotsPerUnit', 'maskExtent']);
});

it('R3 generates Parameters blocks for the public fitting contributors', () => {
  const entries = getReviewClasses().filter(entry =>
    ['GPUAdamStep', 'GPUFittingMinibatchOrder'].includes(entry.className)
  );
  expect(entries.map(entry => entry.className)).toEqual([
    'GPUAdamStep',
    'GPUFittingMinibatchOrder'
  ]);
  const {pages, undocumented} = generateDocPages(
    entries,
    new Map([
      ['fitting.md', '## `GPUAdamStep`\n\nAdam.\n\n## `GPUFittingMinibatchOrder`\n\nOrder.\n']
    ])
  );
  expect(undocumented).toEqual([]);
  expect(pages.get('fitting.md')).toContain('parameters: GPUAdamStep ');
  expect(pages.get('fitting.md')).toContain('parameters: GPUFittingMinibatchOrder ');
});

it('R5 documents all reviewed caller-supplied shared parameter ports', () => {
  const triples = getReviewClasses().flatMap(entry =>
    entry.packers.map(packer => [entry.className, packer.port, packer.call.split('(')[0]])
  );
  expect(triples).toEqual(
    expect.arrayContaining([
      ['GPUSpaceTimeRelativeRisk', 'parameters', 'getGPUSpaceTimeKernelDensityParameterValues'],
      ['GPUQuantileLISA', 'parameters', 'getGPUPermutationParameterValues'],
      ...[
        'GPUPointCloudHeight',
        'GPUPointCloudSurfaces',
        'GPUPointCloudNeighborhoods',
        'GPUPointCloudNormalScale',
        'GPUPointCloudGround'
      ].map(className => [className, 'gridParameters', 'getGPUPointCloudGridParameterValues'])
    ])
  );
});

it.each([
  false,
  true
])('R4 prefers an exact class heading in either page order (reverse=%s)', reverse => {
  const entry: ClassParameters = {className: 'GPUSpatialScanStatistic', packers: []};
  const source: [string, string][] = [
    ['comparison.md', '## Prospective mode of `GPUSpatialScanStatistic`\n\nMode.\n'],
    ['reference.md', '### `GPUSpatialScanStatistic`\n\nGeneral.\n\n### Next class\n']
  ];
  if (reverse) source.reverse();
  const {pages} = generateDocPages([entry], new Map(source));
  expect(pages.get('comparison.md')).not.toContain('parameters: GPUSpatialScanStatistic ');
  const reference = pages.get('reference.md')!;
  expect(reference.indexOf('parameters: GPUSpatialScanStatistic ')).toBeGreaterThan(
    reference.indexOf('### `GPUSpatialScanStatistic`')
  );
  expect(reference.indexOf('parameters: GPUSpatialScanStatistic ')).toBeLessThan(
    reference.indexOf('### Next class')
  );
});

it('R4 refuses ambiguous exact owners instead of silently depending on page order', () => {
  const source = new Map([
    ['first.md', '## `GPUExample`\n\nFirst.\n'],
    ['second.md', '## GPUExample\n\nSecond.\n']
  ]);
  expect(() => generateDocPages([{className: 'GPUExample', packers: []}], source)).toThrow(
    /ambiguous documentation owner for GPUExample/
  );
  for (const page of ['first.md', 'second.md']) {
    const {pages} = generateDocPages([{className: 'GPUExample', packers: []}], source, {
      GPUExample: {page, heading: 'GPUExample'}
    });
    expect(
      [...pages]
        .filter(([, text]) => text.includes('parameters: GPUExample '))
        .map(([target]) => target)
    ).toEqual([page]);
  }
});

it('R4 resolves every live class owner without ambiguity', () => {
  const source = new Map(ANALYSIS_DOC_PAGES.map(page => [page, readDocPage(page)]));
  const failures: string[] = [];
  for (const entry of getReviewClasses()) {
    try {
      const {pages} = generateDocPages([entry], source);
      if (entry.className === 'GPUSpatialScanStatistic') {
        const page = pages.get('gpu-spatial-analysis/operators-dynamics.md')!;
        expect(page).toContain('parameters: GPUSpatialScanStatistic ');
        expect(page.indexOf('parameters: GPUSpatialScanStatistic ')).toBeGreaterThan(
          page.indexOf('### `GPUSpatialScanStatistic`')
        );
        expect(
          pages.get('gpu-spatial-analysis/change-points-and-expectation-based-scan.md')
        ).not.toContain('parameters: GPUSpatialScanStatistic ');
      }
    } catch (error) {
      failures.push(String(error));
    }
  }
  expect(failures).toEqual([]);
}, 600_000);

it('discovers every API page, including pages outside the old allowlist', () => {
  const pages = readdirSync(DOCS_ROOT, {recursive: true})
    .map(String)
    .filter(page => page.endsWith('.md'))
    .sort();
  expect([...ANALYSIS_DOC_PAGES].sort()).toEqual(pages);
});

it('lists every discovered API page exactly once in the sidebar', () => {
  expect(getSidebarPageIssues()).toEqual([]);
});

it('rejects a new page missing from navigation and duplicate or nonexistent sidebar targets', () => {
  expect(getSidebarPageIssues(['new-guide.md'], [])).toEqual([
    'new-guide.md: expected one sidebar entry, found 0'
  ]);
  expect(
    getSidebarPageIssues(['guide.md'], [{items: ['api/guide', 'api/guide', 'api/typo']}])
  ).toEqual(['guide.md: expected one sidebar entry, found 2', 'typo.md: missing API page']);
});

it('documents public shared packers without exposing composite implementation packers', () => {
  const classes = collectClassParameters();
  expect(
    classes.find(entry => entry.className === 'GPUAnalysisContiguityStatistics')
  ).toBeUndefined();
  expect(
    classes.find(entry => entry.className === 'GPUTerrainCumulativeViewshed')?.packers
  ).toEqual([
    expect.objectContaining({
      call: 'getGPUTerrainLineOfSightParameterValues(settings, grid)',
      port: 'parameters'
    })
  ]);
  expect(
    classes
      .find(entry => entry.className === 'GPUBufferSurface')
      ?.packers.map(packer => packer.call)
  ).toEqual(['getGPUOffsetCurveParameterValues(settings)']);
  expect(
    classes
      .find(entry => entry.className === 'GPUHierarchicalDensityClustering')
      ?.packers.some(packer => packer.call.startsWith('getGPUNeighborSearch'))
  ).toBe(false);
}, 600_000);

it.each([
  'GPUPointCloudHeight',
  '`GPUPointCloudHeight`'
])('moves misplaced and duplicate blocks to canonical class heading %s', headingText => {
  const entry: ClassParameters = {
    className: 'GPUPointCloudHeight',
    packers: [
      {
        call: 'getGPUPointCloudHeightParameterValues(settings)',
        format: 'uint32',
        length: null,
        identifiers: [],
        arguments: [],
        fields: []
      }
    ]
  };
  const misplaced = [
    '## `GPUPointCloudSurfaces`',
    '',
    'Surface documentation.',
    '',
    '{/* parameters: GPUPointCloudHeight old */}',
    '',
    '### `GPUPointCloudHeight` parameters',
    '',
    'Wrong place.',
    '',
    '{/* /parameters */}',
    '',
    `## ${headingText}`,
    '',
    'Height documentation.',
    ''
  ].join('\n');
  const source = new Map([
    ['pointcloud-analysis.md', misplaced],
    ['duplicate.md', misplaced.replace(`\n## ${headingText}\n`, '\n## Duplicate notes\n')]
  ]);
  const {pages, undocumented} = generateDocPages([entry], source);
  const page = pages.get('pointcloud-analysis.md')!;
  expect(undocumented).toEqual([]);
  expect(page.indexOf('{/* parameters: GPUPointCloudHeight')).toBeGreaterThan(
    page.indexOf(`## ${headingText}`)
  );
  expect(
    [...pages.values()].join('\n').match(/\{\/\* parameters: GPUPointCloudHeight /g)
  ).toHaveLength(1);
  expect(page).toContain('Surface documentation.');
  expect(page).toContain('Height documentation.');
  expect(generateDocPages([entry], pages).pages).toEqual(pages);
});

// A-22: the generated "Parameters" subsections on the analysis doc pages match the packers and
// their TSDoc. Regenerate with `UPDATE_DOC_PARAMETERS=1 npx vitest run --project node
// modules/analysis/test/analysis-core/doc-parameters.node.spec.ts`.

it('analysis doc "Parameters" subsections are in sync with the packers', () => {
  const {pages, undocumented} = generateDocPages();
  const stale: string[] = [];
  const updated: string[] = [];
  for (const [page, text] of pages) {
    if (text === readDocPage(page)) {
      continue;
    }
    if (process.env['UPDATE_DOC_PARAMETERS']) {
      writeFileSync(join(DOCS_ROOT, page), text);
      updated.push(`docs/api/${page}`);
    } else {
      stale.push(page);
    }
  }
  expect(stale, 'run with UPDATE_DOC_PARAMETERS=1 to regenerate').toEqual([]);
  if (updated.length) {
    console.log(`regenerated Parameters pages:\n${updated.join('\n')}`);
  }
  // Every class with a packer has a section on an analysis page to hold its parameters.
  expect(undocumented).toEqual([]);
}, 600_000);
