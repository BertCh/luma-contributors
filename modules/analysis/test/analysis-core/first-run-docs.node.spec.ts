// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {Console} from 'node:console';
import {expect, it} from 'vitest';
import {computeNetworkReachabilityOnCPU} from '@luma-contributors/analysis/gpu-network';
import {
  computeVariogramModelFitOnCPU,
  fitVariogramModel,
  fitVariogramModelWithDiagnostics
} from '@luma-contributors/analysis/dataframe-analysis';
import typescript from 'typescript';
import {getReadmeLinkIssues} from './first-run-readme-links.test-utils';
import {
  DOCS_ROOT,
  REPOSITORY_ROOT,
  createVirtualProgram,
  formatDiagnostic,
  readDocPage
} from './doc-program';

const WALKTHROUGH = 'operation-forms-and-cpu-twins.md';
const GPU_HEADING = '## Point density from device to readback';

it('UDF-15 the guide decoder maps occupied slot five to its cell, bucket and label', () => {
  const code = [...readDocPage('gpu-dataframe-analysis.md').matchAll(/^```ts\n([\s\S]*?)^```/gm)]
    .map(match => match[1])
    .find(fence => fence.includes('const slot = 5;'));
  expect(code, 'literal occupied-slot decoder').toBeDefined();
  const executable = typescript.transpileModule(
    `${code}\nreturn {cell, bucket, label: labels[cell]};`,
    {compilerOptions: {target: typescript.ScriptTarget.ES2022}}
  ).outputText;
  // Euclidean division: 5 = 1 * 3 + 2. The remainder must not become a fractional cell.
  expect(new Function('labels', executable)(['north', 'south'])).toEqual({
    cell: 1,
    bucket: 2,
    label: 'south'
  });
});

it('the CPU variogram walkthrough recovers the expected exponential model without a device', () => {
  const code = /^```ts\n([\s\S]*?)^```/m.exec(readDocPage(WALKTHROUGH))?.[1];
  expect(code).toBeDefined();
  const source = typescript.createSourceFile(
    'variogram-walkthrough.ts',
    code!,
    typescript.ScriptTarget.ES2022,
    true
  );
  let executableSource = code!;
  for (const statement of [...source.statements].reverse()) {
    if (typescript.isImportDeclaration(statement)) {
      executableSource =
        executableSource.slice(0, statement.getStart(source)) +
        executableSource.slice(statement.end);
    }
  }
  const executable = typescript.transpileModule(executableSource, {
    compilerOptions: {target: typescript.ScriptTarget.ES2022}
  }).outputText;
  const fits: ReturnType<typeof computeVariogramModelFitOnCPU>[] = [];
  const commandConsole = new Console({stdout: process.stdout, stderr: process.stderr});
  const displayed: {
    model: string;
    nugget: number;
    partialSill: number;
    effectiveRange: number;
    status: number;
  }[] = [];
  new Function(
    'computeVariogramModelFitOnCPU',
    'fitVariogramModel',
    'fitVariogramModelWithDiagnostics',
    'console',
    executable
  )(
    (...parameters: Parameters<typeof computeVariogramModelFitOnCPU>) => {
      const result = computeVariogramModelFitOnCPU(...parameters);
      fits.push(result);
      return result;
    },
    fitVariogramModel,
    fitVariogramModelWithDiagnostics,
    {
      log: (...values: unknown[]) => {
        const result = values[0] as Partial<(typeof displayed)[number]> | null;
        if (
          values.length === 1 &&
          result &&
          typeof result === 'object' &&
          result.model === 'exponential'
        ) {
          displayed.push(result as (typeof displayed)[number]);
        }
        commandConsole.log(...values);
      }
    }
  );
  expect(fits).toHaveLength(1);
  expect(displayed).toHaveLength(1);
  expect(displayed[0].model).toBe('exponential');
  expect(displayed[0].status).toBe(0);
  expect(fits[0].status[1]).toBe(0);
  const exponentialParameters = fits[0].parameters.subarray(4, 8);
  const displayedParameters = [
    displayed[0].nugget,
    displayed[0].partialSill,
    displayed[0].effectiveRange
  ];
  // Synthetic gamma(h) = .2 + 1.5 * (1 - exp(-3h/2.2)) defines these coefficients.
  for (const [index, expected] of [0.2, 1.5, 2.2].entries()) {
    expect(Number.isFinite(displayedParameters[index])).toBe(true);
    expect(displayedParameters[index]).toBeCloseTo(expected, 3);
    expect(Number.isFinite(exponentialParameters[index])).toBe(true);
    expect(exponentialParameters[index]).toBeCloseTo(expected, 3);
  }
  expect(Number.isFinite(exponentialParameters[3])).toBe(true);
  expect(exponentialParameters[3]).toBeGreaterThanOrEqual(0);
  expect(exponentialParameters[3]).toBeLessThan(1e-8);
});

it('both READMEs lead to a first computation and state the integration target', () => {
  for (const page of ['README.md', 'modules/analysis/README.md']) {
    const text = readFileSync(join(REPOSITORY_ROOT, page), 'utf8');
    expect(text).toContain('Run your first computation');
    expect(text).toContain('operation-forms-and-cpu-twins');
    expect(text).toContain('10.0.0-alpha.16');
    expect(text.indexOf('Run your first computation')).toBeLessThan(text.indexOf('|'));
  }
});

it('installed-package README links remain usable outside the checkout', () => {
  const text = readFileSync(join(REPOSITORY_ROOT, 'modules/analysis/README.md'), 'utf8');
  const manifest = JSON.parse(
    readFileSync(join(REPOSITORY_ROOT, 'modules/analysis/package.json'), 'utf8')
  );
  // Metadata supplies the provisional identity; release-owner confirmation is tracked in status.
  expect(getReadmeLinkIssues(text, manifest.repository.url, manifest.files)).toEqual([]);
  expect(text).not.toContain('## MGWR');
  expect(readDocPage('gpu-spatial-analysis/operators-regression.md')).toContain(
    '## MGWR bandwidth selection and limits'
  );
});

async function assertWalkthroughLink(
  markdown: string,
  page: string,
  label: string,
  target: string,
  root = DOCS_ROOT,
  readTarget = (path: string) => readFileSync(path, 'utf8')
) {
  // Loading this dependency only for navigation checks keeps isolated variogram fixture
  // collection independent of Markdown rendering and its fixture import registry.
  const {getRenderedWalkthroughDestinations, resolveWalkthroughTarget} = await import(
    './documented-walkthrough-links'
  );
  const matching = getRenderedWalkthroughDestinations(markdown).links.filter(
    link => link.label === label
  );
  expect(matching, `${page}: ${label}`).toHaveLength(1);
  const {path: absolute, anchor} = resolveWalkthroughTarget(root, page, matching[0].href);
  expect(`${absolute}#${anchor}`).toBe(
    `${join(root, target.split('#')[0])}#${target.split('#')[1]}`
  );
  expect(getRenderedWalkthroughDestinations(readTarget(absolute)).anchors, target).toContain(
    anchor
  );
}

it('REVIEW B6 resolves a valid walkthrough in a checkout containing spaces', async () => {
  await assertWalkthroughLink(
    '[walk](../target.md#target)',
    'family/README.md',
    'walk',
    'target.md#target',
    '/tmp/My Checkout/docs/api',
    () => '### Target\n'
  );
});

it('REVIEW S1 rejects links shown only in a comment or code fence', async () => {
  for (const markdown of [
    '<!-- [walk](./target.md#target) -->',
    '```md\n[walk](./target.md#target)\n```'
  ]) {
    await expect(async () =>
      assertWalkthroughLink(
        markdown,
        'README.md',
        'walk',
        'target.md#target',
        DOCS_ROOT,
        () => '### Target\n'
      )
    ).rejects.toThrow();
  }
});

it('DOC-111 family quick starts link by label to their actual runnable destination and anchor', async () => {
  for (const [page, label, target] of [
    [
      'gpu-spatial-analysis/README.md',
      'point-density walkthrough',
      'operation-forms-and-cpu-twins.md#point-density-from-device-to-readback'
    ],
    [
      'gpu-network/README.md',
      'complete GPU walkthrough',
      'operation-forms-and-cpu-twins.md#point-density-from-device-to-readback'
    ],
    [
      'gpu-terrain/README.md',
      'complete GPU walkthrough',
      'README.md#first-run-terrain-derivatives'
    ],
    [
      'gpu-raster/README.md',
      'complete exact-distance-field walkthrough',
      'README.md#first-run-exact-distance-field'
    ],
    [
      'gpu-dataframe-analysis.md',
      'complete first-run time-window filter',
      'README.md#first-run-time-window-filter'
    ],
    [
      'pointcloud-analysis.md',
      'complete statistical-filter example',
      'README.md#first-run-point-cloud-statistical-filter'
    ]
  ]) {
    await assertWalkthroughLink(readDocPage(page), page, label, target);
  }
});

it('the documented network inputs seed only node one and initialize useful cost bands', () => {
  const section = readDocPage('gpu-network/README.md').split('## Quick start')[1];
  const code = /^```ts\n([\s\S]*?)^```/m.exec(section)?.[1];
  expect(code).toBeDefined();
  expect(code).toContain('const reachabilityInputs');
  const source = readFileSync(
    join(REPOSITORY_ROOT, 'modules/analysis/test/analysis-core/network-first-run.test-utils.ts'),
    'utf8'
  )
    .replace(/^\/\/.*\n/gm, '')
    .trim();
  expect(code?.trim()).toBe(source);
  const input = new Function(`${code?.replace('export ', '')}\nreturn reachabilityInputs;`)();
  const result = computeNetworkReachabilityOnCPU(input);
  expect(Array.from(result.costs)).toEqual([Infinity, 0, 2]);
  expect(Array.from(result.bands)).toEqual([0xffffffff, 0, 1]);
  expect(Array.from(result.bandCounts)).toEqual([1, 1, 0, 0]);
  // This is the old documented mistake: spare zero slots silently add node zero as a source.
  const extraSources = computeNetworkReachabilityOnCPU({
    ...input,
    startNodes: Uint32Array.of(1, 0, 0, 0)
  });
  expect(extraSources.costs[0]).toBe(0);
});

it('the GPU walkthrough is exactly the executable example and has no implicit inputs', () => {
  const section = readDocPage(WALKTHROUGH).split(GPU_HEADING)[1];
  expect(section, 'complete GPU walkthrough section').toBeDefined();
  const snippets = [...section.matchAll(/^```ts\n([\s\S]*?)^```/gm)];
  expect(snippets.length).toBe(1);
  const source = readFileSync(
    join(
      REPOSITORY_ROOT,
      'modules/analysis/test/analysis-core/point-density-first-run.test-utils.ts'
    ),
    'utf8'
  )
    .split('// BEGIN LITERAL FENCES\n')[1]
    ?.split('\n// END LITERAL FENCES')[0]
    .trim();
  expect(snippets[0][1].trim()).toBe(source);
  const path = join(DOCS_ROOT, '__doc_snippets__', 'point-density-first-run.ts');
  const program = createVirtualProgram(new Map([[path, snippets[0][1]]]));
  const file = program.getSourceFile(path);
  expect(
    [...program.getSyntacticDiagnostics(file), ...program.getSemanticDiagnostics(file)].map(
      formatDiagnostic
    )
  ).toEqual([]);
}, 120_000);
