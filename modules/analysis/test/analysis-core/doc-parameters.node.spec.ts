// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readdirSync, writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {expect, it} from 'vitest';
import {
  ANALYSIS_DOC_PAGES,
  DOCS_ROOT,
  REPOSITORY_ROOT,
  createVirtualProgram,
  getSidebarPageIssues,
  readDocPage
} from './doc-program';
import {
  collectClassParameters,
  collectClassParametersFromInventory,
  generateDocPages,
  renderClassParameters,
  type ClassParameters
} from './doc-parameters';
import {getPublicDeclarationsFromSources} from './public-declarations';

it('ENT-09 generates installed imports from qualified manifest identities despite route mismatches and namesakes', () => {
  const sourceRoot = join(REPOSITORY_ROOT, 'modules/analysis/src');
  const program = createVirtualProgram(
    new Map([
      [
        join(sourceRoot, 'import-fixture-first.ts'),
        'export class GPUExample {}\nexport function getGPUExampleParameterValues(settings: {first: number}): Float32Array {return new Float32Array([settings.first]);}'
      ],
      [
        join(sourceRoot, 'import-fixture-second.ts'),
        'export class GPUExample {}\nexport function getGPUExampleParameterValues(settings: {second: number}): Uint32Array {return new Uint32Array([settings.second]);}'
      ]
    ])
  );
  const inventory = getPublicDeclarationsFromSources(program, sourceRoot, {
    name: '@luma-contributors/analysis',
    exports: {
      './raster-analysis': './dist/import-fixture-first.js',
      './gpu-network': './dist/import-fixture-second.js'
    }
  });
  const parameters = collectClassParametersFromInventory(program, inventory);
  expect(parameters).toHaveLength(2);
  const {pages} = generateDocPages(
    parameters,
    new Map([
      ['gpu-raster/example.md', '# Raster\n\n## `GPUExample`\n\nFirst implementation.\n'],
      ['network-example.md', '# Network\n\n## `GPUExample`\n\nSecond implementation.\n']
    ]),
    {
      '@luma-contributors/analysis/raster-analysis:GPUExample': {
        page: 'gpu-raster/example.md',
        heading: 'GPUExample'
      },
      '@luma-contributors/analysis/gpu-network:GPUExample': {
        page: 'network-example.md',
        heading: 'GPUExample'
      }
    }
  );
  for (const [page, entryPoint] of [
    ['gpu-raster/example.md', 'raster-analysis'],
    ['network-example.md', 'gpu-network']
  ]) {
    expect(pages.get(page)).toContain(
      `Import from \`@luma-contributors/analysis/${entryPoint}\`: \`GPUExample\`.`
    );
    expect(pages.get(page)).toContain('## `GPUExample`');
    expect(pages.get(page)?.match(/parameters: GPUExample /g)).toHaveLength(1);
  }
});

let reviewClasses: ClassParameters[] | undefined;
function getReviewClasses(): ClassParameters[] {
  reviewClasses ??= collectClassParameters();
  return reviewClasses;
}

it('W5-16 reads calibration layout from the implementation and presents both public overloads', () => {
  const entry = getReviewClasses().find(
    candidate => candidate.className === 'GPURasterStandardizedIndexCalibration'
  )!;
  expect(entry.packers).toHaveLength(2);
  expect(entry.packers.map(packer => packer.call)).toEqual([
    'getGPURasterStandardizedIndexCalibrationParameterValues(settings, chunkSliceCount)',
    'getGPURasterStandardizedIndexCalibrationParameterValues(plan, frame)'
  ]);
  for (const packer of entry.packers) {
    // Seven words: reset, live slices, location, three exponents, magnitude bound.
    expect(packer.length).toBe('`GPU_RASTER_STANDARDIZED_CALIBRATION_PARAMETER_LENGTH` (7)');
  }
});

it('W5-16 documents the three public neighbor-search settings forms', () => {
  const entry = getReviewClasses().find(candidate => candidate.className === 'GPUNeighborSearch')!;
  const fields = entry.packers.map(packer => packer.fields);
  expect(fields.map(branch => branch.find(field => field.name === 'mode')?.type)).toEqual([
    "'radius'",
    "'knn'",
    'never'
  ]);
  const [radius, nearest, modeFree] = fields;
  expect(radius).toContainEqual(
    expect.objectContaining({name: 'searchRadius', type: 'number', optional: false})
  );
  expect(radius).toContainEqual(expect.objectContaining({name: 'maximumDistance', type: 'never'}));
  expect(nearest).toContainEqual(expect.objectContaining({name: 'searchRadius', type: 'never'}));
  expect(nearest).toContainEqual(
    expect.objectContaining({name: 'maximumDistance', type: 'number', optional: true})
  );
  for (const name of ['searchRadius', 'maximumDistance']) {
    expect(modeFree).toContainEqual(
      expect.objectContaining({name, type: 'number', optional: true})
    );
  }
  for (const branch of fields) expect(branch.map(field => field.name)).not.toContain('radius');
});

it('keeps the Vecchia destination slot before search anisotropy', () => {
  const entry = getReviewClasses().find(
    candidate => candidate.className === 'GPUVecchiaSpaceTimeLogLikelihood'
  )!;
  const rendered = renderClassParameters(entry, 3);
  expect(rendered).toContain(
    '`getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target?, searchAnisotropy?)`'
  );
  expect(rendered).toContain('`target` (`Float32Array`): Optional destination');
});

it('documents Calendar Buckets as a signed packer without a destination settings table', () => {
  const entry = getReviewClasses().find(candidate => candidate.className === 'GPUCalendarBuckets')!;
  const rendered = renderClassParameters(entry, 3);
  expect(rendered).toContain(
    '`getGPUCalendarBucketsParameterValues(utcOffsetMinutes, firstDayOfWeek?)`, writes `GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH` (2) `sint32` values.'
  );
  expect(rendered).not.toContain('Settings of');
});

it('W5-17 pins flow diagnostics to the canonical real page even with compatibility headings', () => {
  const names = ['GPULagrangianFlowMap', 'GPUFiniteTimeLyapunovExponent', 'GPUScalarFieldRidges'];
  const entries = getReviewClasses().filter(entry => names.includes(entry.className));
  expect(entries).toHaveLength(3);
  const canonical = 'gpu-raster/flow-map-and-particle-diagnostics.md';
  const compatibility = 'gpu-raster/lagrangian-diagnostics-mob06.md';
  const source = new Map([canonical, compatibility].map(page => [page, readDocPage(page)]));
  for (const compatibilityHeadings of [false, true]) {
    const input = new Map(source);
    if (compatibilityHeadings)
      input.set(
        compatibility,
        source.get(compatibility)! +
          names.map(name => `\n## \`${name}\`\n\nCompatibility reference.\n`).join('')
      );
    const {pages, undocumented} = generateDocPages(entries, input);
    expect(undocumented).toEqual([]);
    for (const name of names) {
      const marker = new RegExp(`parameters: ${name} `, 'g');
      expect(pages.get(canonical)?.match(marker)).toHaveLength(1);
      expect(pages.get(compatibility)).not.toContain(`parameters: ${name} `);
      expect([...pages.values()].join('\n').match(marker)).toHaveLength(1);
      expect(pages.get(canonical)).toContain(
        `Import from \`@luma-contributors/analysis/raster-analysis\`: \`${name}\`.`
      );
    }
  }
});

it('round 1 S1 distinguishes the class import from borrowed parameter packer entries', () => {
  const entry = getReviewClasses().find(candidate => candidate.className === 'GPUFlowAggregation')!;
  const {pages} = generateDocPages(
    [entry],
    new Map([['flow.md', '## `GPUFlowAggregation`\n\nFlow aggregation.\n']])
  );
  const markdown = pages.get('flow.md')!;
  expect(markdown).toContain(
    'Import from `@luma-contributors/analysis/gpu-network`: `GPUFlowAggregation`.'
  );
  for (const name of [
    'getGPUTimeWindowFilterParameterValues',
    'getGPUTimeWindowFilterWordParameterValues'
  ]) {
    expect(markdown).toContain(
      `Import \`${name}\` from \`@luma-contributors/analysis/dataframe-analysis\`.`
    );
  }
});

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
  ).toEqual(['seed', 'dotsPerUnit', 'maskPlacement']);
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

it('ENT-09 preserves a declared shared-section owner when another page adds a named heading', () => {
  const entry: ClassParameters = {
    entryPoint: '@luma-contributors/analysis/raster-analysis',
    declaration: 'fixture.ts#GPUExample',
    className: 'GPUExample',
    packers: []
  };
  const {pages} = generateDocPages(
    [entry],
    new Map([
      ['shared.md', '## Shared contract\n\nGPUExample owns the fractional tuples.\n'],
      ['comparison.md', '## `GPUExample`\n\nA comparison with other contributors.\n']
    ]),
    {GPUExample: {page: 'shared.md', heading: 'Shared contract'}}
  );
  expect(pages.get('shared.md')).toContain('parameters: GPUExample ');
  expect(pages.get('shared.md')).toContain(
    'Import from `@luma-contributors/analysis/raster-analysis`: `GPUExample`.'
  );
  expect(pages.get('comparison.md')).not.toContain('parameters: GPUExample ');
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
      .find(entry => entry.className === 'GPUHDBSCAN')
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
