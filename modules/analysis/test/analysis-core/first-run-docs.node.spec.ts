// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {expect, it} from 'vitest';
import {computeNetworkReachabilityOnCPU} from '@luma-contributors/analysis/gpu-network';
import {computeVariogramModelFitOnCPU} from '@luma-contributors/analysis/dataframe-analysis';
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

it('the CPU variogram walkthrough recovers the expected exponential model without a device', () => {
  const code = /^```ts\n([\s\S]*?)^```/m.exec(readDocPage(WALKTHROUGH))?.[1];
  expect(code).toBeDefined();
  const executable = typescript.transpileModule(
    code!.replace(/^import .*;\n/m, '') + '\nreturn {cpuFit, exponentialParameters};',
    {compilerOptions: {target: typescript.ScriptTarget.ES2022}}
  ).outputText;
  const result = new Function('computeVariogramModelFitOnCPU', executable)(
    computeVariogramModelFitOnCPU
  );
  expect(result.cpuFit.status[1]).toBe(0);
  for (const [index, expected] of [0.2, 1.5, 2.2].entries()) {
    expect(Number.isFinite(result.exponentialParameters[index])).toBe(true);
    expect(result.exponentialParameters[index]).toBeCloseTo(expected, 3);
  }
  expect(result.exponentialParameters[3]).toBeLessThan(1e-8);
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

it('point density and five other family quick starts link to the complete GPU lifecycle', () => {
  for (const page of [
    'gpu-spatial-analysis/README.md',
    'gpu-network/README.md',
    'gpu-terrain/README.md',
    'gpu-raster/README.md',
    'gpu-dataframe-analysis.md',
    'pointcloud-analysis.md'
  ]) {
    expect(readDocPage(page)).toContain(
      'operation-forms-and-cpu-twins.md#point-density-from-device-to-readback'
    );
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
  expect(snippets.length).toBe(2);
  const source = readFileSync(
    join(
      REPOSITORY_ROOT,
      'modules/analysis/test/analysis-core/point-density-first-run.test-utils.ts'
    ),
    'utf8'
  )
    .replace(/^\/\/ SPDX-.*\n/gm, '')
    .trim();
  expect(snippets.map(match => match[1].trim()).join('\n\n')).toBe(source);
  const path = join(DOCS_ROOT, '__doc_snippets__', 'point-density-first-run.ts');
  const program = createVirtualProgram(
    new Map([[path, snippets.map(match => match[1]).join('\n')]])
  );
  const file = program.getSourceFile(path);
  expect(
    [...program.getSyntacticDiagnostics(file), ...program.getSemanticDiagnostics(file)].map(
      formatDiagnostic
    )
  ).toEqual([]);
}, 120_000);
