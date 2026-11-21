// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import type {Plugin} from 'vite';
import {
  GPUCoregionalizationFit,
  type GPUCoregionalizationFitProps
} from '../../src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit';
import {createCoregionalizationAuditProps} from './kernel-audit/cases-coregionalization-fit';
import {
  createKernelAuditContext,
  type KernelAuditContext
} from './kernel-audit/kernel-audit-context';
import {getGPUCommandNodeKernelSources} from '../utils/gpu-contributor-test-utils';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '../../../..');
const WALKTHROUGH_PATH = 'dev-docs/authoring-walkthrough.md';
const SCENARIO_ID = 'gpu-dataframe.coregionalization-fit';

function readRepositoryFile(path: string): string {
  return readFileSync(resolve(REPOSITORY_ROOT, path), 'utf8');
}

function readAuthoringStep(step: number): string {
  return readRepositoryFile(WALKTHROUGH_PATH)
    .split(`## ${step}.`)[1]
    .split(`## ${step + 1}.`)[0]
    .split('\n')
    .slice(1)
    .join('\n');
}

function createSmallFitProps(context: KernelAuditContext): GPUCoregionalizationFitProps {
  return {
    ...createCoregionalizationAuditProps(context),
    variableCount: 2,
    structureCount: 1,
    maximumIterations: 0,
    distances: context.view('float32', 1),
    semivariances: context.view('float32', 3),
    pairCounts: context.view('uint32', 1),
    parameters: context.view('float32', 5)
  };
}

it('R1 distinguishes a runnable lens capture from published result and graph links', async () => {
  const familyPath = 'docs/api/gpu-dataframe-analysis.md';
  const section = readRepositoryFile(familyPath)
    .split('## `GPUCoregionalizationFit`')[1]
    .split('\n## ')[0];
  // Load the website module at runtime: its own TypeScript project differs from analysis tests.
  const compilerPath = resolve(REPOSITORY_ROOT, 'website/build/markdown-plugin.ts');
  const compiler = (await import(compilerPath)) as {
    markdownPlugin: (options: {docsDirectory: string}) => Plugin;
  };
  const plugin = compiler.markdownPlugin({docsDirectory: resolve(REPOSITORY_ROOT, 'docs')});
  const transform = plugin.transform;
  if (typeof transform !== 'function') throw new Error('Markdown transform must be callable');
  const result = await transform.call({} as never, section, resolve(REPOSITORY_ROOT, familyPath));
  if (!result || typeof result === 'string' || typeof result.code !== 'string') {
    throw new Error('Markdown transform must return code');
  }
  const rendered = JSON.parse(result.code.replace(/^export default /, '').replace(/;$/, '')) as {
    html: string;
  };
  const recordPath = `docs/lens-records/${SCENARIO_ID}.json`;
  const recordingPath = `/lens/${SCENARIO_ID}`;
  const recordingRungs = [...rendered.html.matchAll(/<a href="([^"]+)">[^<]+ \(([^)]+)\)<\/a>/g)]
    .filter(([, href]) => href === recordingPath)
    .map(([, , rung]) => rung);
  expect(recordingRungs).toEqual(['result', 'graph']);
  expect(rendered.html).not.toContain(`::lens{scenario=${SCENARIO_ID}`);
  expect(rendered.html).not.toMatch(/<figure\b/);
  expect(existsSync(resolve(REPOSITORY_ROOT, recordPath))).toBe(true);
  const lensStep = readAuthoringStep(10);
  expect(lensStep).toContain('result and graph directives both link');
  expect(lensStep).toContain('website/build/markdown-plugin.ts');
  expect(lensStep).toContain(recordPath);
  expect(lensStep).toContain(
    'modules/lenses/test/computation-lenses/coregionalization-fit.spec.ts'
  );
  expect(lensStep).toContain('modules/lenses/test/computation-lenses/capture-records.spec.ts');
  expect(lensStep).toContain('VITE_CAPTURE_LENS_RECORDS=1');
});

it('R2 preserves the snapshotted dimension after caller mutation', () => {
  const context = createKernelAuditContext('authoring-mutable-props');
  try {
    const props = createSmallFitProps(context);
    const contributor = new GPUCoregionalizationFit(props);
    props.variableCount = 5;
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(context.graph));
    const dimensions = sources.map(kernel =>
      kernel.source.match(/const VARIABLE_COUNT: u32 = (\d+)u/)
    );
    expect(dimensions.length).toBeGreaterThan(0);
    expect(dimensions.every(match => match !== null)).toBe(true);
    expect(dimensions.map(match => match![1])).toEqual(sources.map(() => '2'));
    const validationStep = readAuthoringStep(3);
    expect(validationStep).toContain('snapshots compile-time options');
    expect(validationStep).toContain('coregionalization-fit.node.spec.ts#L29');
    expect(validationStep).not.toContain("currently retains the caller's props");
    expect(validationStep).not.toContain('it currently fails');
    expect(validationStep).toContain('VARIABLE_COUNT = 2u');
  } finally {
    context.device.destroy();
  }
});

it('R3 sends numerical edits to the same canonical guide as the catalog', () => {
  const catalog = readRepositoryFile(
    'modules/analysis/src/gpu-spatial-analysis/ontology/all-entry-points/entry-point-operator-details.ts'
  );
  const record = catalog.split('GPUCoregionalizationFit: {')[1].split('\n  },')[0];
  const documentation = record.match(/documentation: '([^']+)'/);
  expect(documentation).not.toBeNull();
  const canonicalGuide = documentation![1];
  expect(existsSync(resolve(REPOSITORY_ROOT, canonicalGuide))).toBe(true);
  const documentationStep = readAuthoringStep(9);
  const editTarget = documentationStep.match(/^\s*Update the \[[^\]]+\]\(\.\.\/([^)]+)\)/);
  expect(editTarget, 'the numerical-edit instruction needs a linked owner').not.toBeNull();
  expect(editTarget![1]).toBe(canonicalGuide);
  expect(documentationStep).toContain('gpu-dataframe-analysis.md#gpucoregionalizationfit');
  expect(documentationStep).toContain(
    'gpu-dataframe-analysis.md#gpucoregionalizationfit-parameters'
  );
});

it('R4 demonstrates the wrong-format input without overstating the existing constructor test', () => {
  const context = createKernelAuditContext('authoring-wrong-format');
  try {
    const props = createSmallFitProps(context);
    expect(() => new GPUCoregionalizationFit(props)).not.toThrow();
    const wrongFormat = context.view('uint32', props.distances.length);
    expect(
      () =>
        new GPUCoregionalizationFit({
          ...props,
          distances: wrongFormat as unknown as GPUCoregionalizationFitProps['distances']
        })
    ).toThrow(/coregionalization-fit distances must be packed, uint32-aligned float32 GPU data/);
    const testingStep = readAuthoringStep(8);
    expect(testingStep).toContain('does not exercise a wrong-format input');
    expect(testingStep).toContain("context.view('uint32', props.distances.length)");
  } finally {
    context.device.destroy();
  }
});

it('R5 distinguishes the configured sweep budget from completed sweeps', () => {
  const runner = readRepositoryFile(
    'modules/lenses/src/computation-lenses/scenarios/gpu-dataframe/coregionalization-fit.run.ts'
  );
  const lensSpec = readRepositoryFile(
    'modules/lenses/test/computation-lenses/coregionalization-fit.spec.ts'
  );
  expect(runner).toMatch(/maximumIterations: 8\b/);
  expect(lensSpec).toContain("expect(getValues('iterationCount')).toEqual([2])");
  expect(readAuthoringStep(10)).toContain('sets an eight-sweep budget');
});
